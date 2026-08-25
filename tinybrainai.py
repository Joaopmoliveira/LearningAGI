import collections
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from PIL import Image
import cv2

import os
import numpy as np
import matplotlib.pyplot as plt
from fpdf import FPDF
from PIL import Image

def generate_pdf_report(episode_rewards, eval_frames, filename="learning_report.pdf"):
    """
    Generates a PDF report containing training reward metrics and captured evaluation frames.
    """
    # 1. Render and save the learning curve plot
    plt.figure(figsize=(6, 3))
    plt.plot(episode_rewards, label="Episode Reward", color="#2b5c8f", alpha=0.6)
    
    # Calculate 20-episode moving average
    if len(episode_rewards) >= 20:
        ma = [np.mean(episode_rewards[max(0, i-20):i+1]) for i in range(len(episode_rewards))]
        plt.plot(ma, label="20-Ep Moving Avg", color="#d95f02", linewidth=2)
        
    plt.title("Agent Learning Progress")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    
    plot_path = "temp_reward_plot.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()

    # 2. Extract 4 evenly spaced evaluation frames
    frame_paths = []
    step_size = max(1, len(eval_frames) // 4)
    sampled_frames = eval_frames[::step_size][:4]

    for idx, frame in enumerate(sampled_frames):
        img_path = f"temp_frame_{idx}.png"
        if isinstance(frame, np.ndarray):
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
            Image.fromarray(frame).save(img_path)
        frame_paths.append(img_path)

    # 3. Assemble PDF layout
    pdf = FPDF()
    pdf.add_page()
    
    # Header
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 10, "Brain Model Training & Behavior Report", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(5)

    # Summary Statistics
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 6, f"Total Training Episodes: {len(episode_rewards)}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"Final Moving Avg Reward: {np.mean(episode_rewards[-20:]):.2f}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"Peak Episode Reward: {max(episode_rewards):.2f}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    # Section 1: Learning Curve
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "1. Learning Performance", new_x="LMARGIN", new_y="NEXT")
    pdf.image(plot_path, x=15, w=180)
    pdf.ln(5)

    # Section 2: Behavior Frames
    pdf.cell(0, 8, "2. Replicated Behavior (Evaluation Rollout)", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    # Render a 2x2 grid of evaluation frames
    x_coords = [15, 105, 15, 105]
    y_start = pdf.get_y()
    for i, fpath in enumerate(frame_paths):
        y_pos = y_start if i < 2 else y_start + 55
        pdf.image(fpath, x=x_coords[i], y=y_pos, w=85)

    pdf.output(filename)

    # Clean up temporary image files
    os.remove(plot_path)
    for fpath in frame_paths:
        if os.path.exists(fpath):
            os.remove(fpath)

## The world is an abstraction over what our agents can see:
# the way I see this is that we can observe the world, get rewards, etc
# doubts: 
# should I have a method to query the potential reward without advancing the world?
# this seems to go against the RL literature, 
# where the agent/brain should explore all of these things by itself
class World:
    def __init__(self, env_name, img_size=64, render=False):
        self.render_frame_in_window = render
        self.img_size = img_size
        self.frame_stack = 1
        self.env = gym.make(env_name, render_mode="rgb_array")
        self.obs_shape = (self.frame_stack, img_size, img_size)
        self.world_frame = None
        self.n_actions = self.env.action_space.n

        if render:
            cv2.namedWindow("World", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("World", 512, 512)

    def reset(self):
        self.env.reset()
        self.world_frame = self._grab_frame()


    def _grab_frame(self):
        frame = self.env.render()  # (H, W, 3) 
        img = Image.fromarray(frame).convert("L").resize((self.img_size, self.img_size))
        img_np = np.asarray(img, dtype=np.uint8)

        if self.render_frame_in_window:
            display_frame = cv2.resize(img_np, (512, 512), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("World", display_frame)
            cv2.waitKey(1)

        return img_np.astype(np.float32) / 255.0


    def observe(self):
        return torch.as_tensor(self.world_frame, dtype=torch.float32).unsqueeze(0)
 
    def act(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.world_frame = self._grab_frame()
        return obs, reward, terminated or truncated, info
 
    def close(self):
        self.env.close()
        if self.render_frame_in_window:
            cv2.destroyAllWindows()

# the proto-memory and corresponding action are concatenated into a
# structure. I dislike this in general, but I can't really articilate why.
# Here it goes: these rewards will vary in between games. 
# Storing the rewards seems like it looses the generalization inbetween games
# the rewards are just a signal for us to store memories, at least thats my feeling
# but for now I need to store a signal to indicate to the attention mechanism 
# what is good and what is bad
class Association:
    def __init__(self, proto, action, reward, reward_prediction_error):
        self.proto = proto
        self.action = action
        self.reward = reward
        self.reward_prediction_error = reward_prediction_error

# The memory is just a list of Associations, 
# with some hardcoded heuristics for the moment.
class Memory:
    def __init__(self, short_capacity=8, long_capacity=64):
        self.short_term = collections.deque(maxlen=short_capacity)
        self.long_term = []
        self.long_capacity = long_capacity

    def add(self, transition):
        self.short_term.append(transition)
        if len(self.long_term) < self.long_capacity:
            self.long_term.append(transition)
        else:
            min_idx = min(range(len(self.long_term)), key=lambda i: abs(self.long_term[i].reward_prediction_error))
            if abs(transition.reward_prediction_error) > abs(self.long_term[min_idx].reward_prediction_error):
                self.long_term[min_idx] = transition

    def snapshot(self, k=8):
        pool = list(self.short_term) + self.long_term
        if not pool:
            return None
        
        sample = random.sample(pool, min(k, len(pool)))
        return torch.stack([t.proto for t in sample])

    def add(self, transition):
        self.short_term.append(transition)
        if len(self.long_term) < self.long_capacity:
            self.long_term.append(transition)
        else:
            min_idx = min(
                range(len(self.long_term)),
                key=lambda i: abs(self.long_term[i].reward_prediction_error)
            )
            if abs(transition.reward_prediction_error) > abs(self.long_term[min_idx].reward_prediction_error):
                self.long_term[min_idx] = transition

    def snapshot(self, k_longterm=4):
        if not self.short_term:
            return None
        ## this is stupid but for the moment I don't know a better way. I think the 
        # atention mechanism should look at all memories and decide
        sequential_protos = [t.proto for t in self.short_term]
        long_protos = []
        if self.long_term:
            k = min(k_longterm, len(self.long_term))
            long_protos = [t.proto for t in random.sample(self.long_term, k)]
        return torch.stack(sequential_protos + long_protos)

## The conv encoder is supposed to be a backbone 
# of some YOLO famous network, the goal is that this is a 
# thing that generates an embbedding of the world
class ConvEncoder(nn.Module):
    def __init__(self, in_channels, img_size, d_model):
        super().__init__()
        self.in_channels = in_channels
        self.img_size = img_size
        self.d_model = d_model
 
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2), nn.ReLU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, img_size, img_size)
            conv_out = self.conv(dummy)
        self.conv_out_shape = conv_out.shape[1:]  # (C, H, W) after the conv stack
        conv_out_dim = conv_out.flatten(1).shape[1]
        self.fc = nn.Linear(conv_out_dim, d_model)
 
    def forward(self, obs):
        single = obs.dim() == 3
        if single:
            obs = obs.unsqueeze(0)  # add batch dim
        x = self.conv(obs).flatten(1)
        latent = self.fc(x)
        return latent.squeeze(0) if single else latent


## The conv dencoder is supposed to be something that can 
# interpret the embedding of the encoder and recreate the original image
# from the world. This should be used to understand if the model is correctly 
# simulating the world
class ConvDecoder(nn.Module):
    def __init__(self, encoder: ConvEncoder):
        super().__init__()
        c, h, w = encoder.conv_out_shape
        self._unflatten_shape = (c, h, w)
        self.img_size = encoder.img_size
        self.fc = nn.Linear(encoder.d_model, c * h * w)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(c, 16, kernel_size=4, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(16, encoder.in_channels, kernel_size=8, stride=4),
            nn.Sigmoid(),  # outputs land in [0, 1], matching the normalized frames
        )
 
    def forward(self, latent):
        single = latent.dim() == 1
        if single:
            latent = latent.unsqueeze(0)
        x = self.fc(latent)
        x = x.view(-1, *self._unflatten_shape)
        recon = self.deconv(x)
        # Transposed-conv output size doesn't land exactly on img_size due to
        # kernel/stride rounding, so snap it back to match for clean losses.
        if recon.shape[-1] != self.img_size:
            recon = F.interpolate(recon, size=(self.img_size, self.img_size),
                                   mode="bilinear", align_corners=False)
        return recon.squeeze(0) if single else recon

class Cortex(nn.Module):
    def __init__(self, d_model, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_actions),
        )
 
    def forward(self, fused):
        return self.net(fused)
 
 
class RewardCenter(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )
 
    def forward(self, fused):
        return self.net(fused).squeeze(-1)
 
 
class Brain(nn.Module):
    def __init__(self, world, d_model=32, num_heads=2):
        super().__init__()
        self.memory = Memory()
        frame_stack, img_size, _ = world.obs_shape
        self.perception = ConvEncoder(frame_stack, img_size, d_model)
        self.decoder = ConvDecoder(self.perception) 
 
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.cortex = Cortex(d_model, world.n_actions)
        self.reward_center = RewardCenter(d_model)
        self.d_model = d_model
 
    def reconstruct(self, latent):
        return self.decoder(latent)

    def remember(self, proto, action, reward, reward_prediction_error):
        self.memory.add(Association(proto, action, reward, reward_prediction_error))
 
    def forward(self, obs):
        mem_snap = self.memory.snapshot()
        protomemory = self.perception(obs)                     # (d_model,)
        query = protomemory.unsqueeze(0).unsqueeze(0)           # (1, 1, d_model)
 
        if mem_snap is not None:
            kv = mem_snap.unsqueeze(0)             # (1, k, d_model)
            attended, _ = self.attention(query, kv, kv)
            attended = attended.squeeze(0).squeeze(0)      # (d_model,)
        else:
            attended = torch.zeros_like(protomemory)
 
        fused = torch.cat([protomemory, attended], dim=-1)       # (2*d_model,)
        action = self.cortex(fused)
        predicted_reward = self.reward_center(fused)
        return action, predicted_reward, protomemory.detach()


class Agent:
    def __init__(self):
        pass

    def observe_and_act(self):
        pass


def train(num_episodes=1000, gamma=0.99, lr=1e-3, entropy_coef=0.01, log_every=20,render=False):
    world = World("CartPole-v1", render=render)
    brain = Brain(world)

    optimizer = torch.optim.Adam(brain.parameters(), lr=lr)
 
    episode_rewards = []
 
    for episode in range(num_episodes):
        world.reset()
        done = False
        ep_reward = 0.0
        log_probs, predicted_rewards, rewards, entropies = [], [], [], []
 
        while not done:
            viewed_world = world.observe()
            action, predicted_reward, protomemory = brain(viewed_world)
 
            dist = torch.distributions.Categorical(logits=action)
            action = dist.sample()
 
            next_obs, reward, done, _ = world.act(action.item())
 
            log_probs.append(dist.log_prob(action))
            predicted_rewards.append(predicted_reward)
            rewards.append(reward)
            entropies.append(dist.entropy())
 
            # crude "surprise" signal used purely to decide what's worth
            # remembering; the real learning signal is the loss below
            reward_prediction_error = reward - predicted_reward.item()
            brain.remember(proto=protomemory, action=action.item(), reward=reward, reward_prediction_error=reward_prediction_error)
 
            ep_reward += reward
 
        # ---- discounted returns and advantages ----
        returns = []
        R = 0.0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns, dtype=torch.float32)
        values_t = torch.stack(predicted_rewards)
        advantages = returns - values_t.detach()
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
 
        log_probs_t = torch.stack(log_probs)
        entropies_t = torch.stack(entropies)
 
        actor_loss = -(log_probs_t * advantages).mean()
        critic_loss = F.mse_loss(values_t, returns)
        entropy_bonus = entropies_t.mean()
        loss = actor_loss + 0.5 * critic_loss - entropy_coef * entropy_bonus
 
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(brain.parameters(), max_norm=1.0)
        optimizer.step()
 
        episode_rewards.append(ep_reward)
        if (episode + 1) % log_every == 0:
            avg = np.mean(episode_rewards[-log_every:])
            print(f"Episode {episode + 1:4d} | avg reward (last {log_every}): {avg:.1f}")
 
    return brain, episode_rewards


def train_single(num_episodes=1000, gamma=0.99, lr=1e-3, entropy_coef=0.01, log_every=20, render=False):
    world = World("CartPole-v1", render=render)
    brain = Brain(world)
    optimizer = torch.optim.Adam(brain.parameters(), lr=lr)
 
    episode_rewards = []
 
    for episode in range(num_episodes):
        world.reset()
        done = False
        ep_reward = 0.0
 
        while not done:
            viewed_world = world.observe()
            action_logits, value, protomemory = brain(viewed_world)
 
            dist = torch.distributions.Categorical(logits=action_logits)
            action = dist.sample()
 
            _, reward, done, _ = world.act(action.item())
            ep_reward += reward

            # --- 1. Compute 1-Step TD Target & Advantage ---
            if done:
                next_value = torch.tensor(0.0)
            else:
                with torch.no_grad():
                    next_view = world.observe()
                    _, next_value, _ = brain(next_view)

            target = reward + gamma * next_value
            advantage = (target - value).detach()

            # --- 2. Step-Level Loss Computation ---
            actor_loss = -dist.log_prob(action) * advantage
            critic_loss = F.mse_loss(value.unsqueeze(0), target.unsqueeze(0))
            entropy_bonus = dist.entropy()

            p = episode / num_episodes

            # Linear schedules
            w_recon  = max(0.05, 1.0 - p)        # Decays from 1.0 down to 0.05
            w_critic = min(0.5, 0.1 + 0.4 * p)   # Increases from 0.1 up to 0.5
            w_actor  = min(1.0, 0.05 + 0.95 * p) # Increases from 0.05 up to 1.0

            # Calculate total loss for current step
            recon_loss = F.mse_loss(brain.reconstruct(protomemory), viewed_world)
            loss = (w_recon * recon_loss) + (w_actor * actor_loss) + (w_critic * critic_loss) - (entropy_coef * entropy_bonus)

            # --- 3. Immediate Gradient Update ---
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(brain.parameters(), max_norm=1.0)
            optimizer.step()

            # --- 4. Store Memory ---
            brain.remember(
                proto=protomemory, 
                action=action.item(), 
                reward=reward, 
                reward_prediction_error=advantage.item()
            )
 
        episode_rewards.append(ep_reward)
        if (episode + 1) % log_every == 0:
            avg = np.mean(episode_rewards[-log_every:])
            print(f"Episode {episode + 1:4d} | avg reward (last {log_every}): {avg:.1f}")
 
    return brain, episode_rewards
 
 
def watch(brain, num_episodes=100, greedy=True, render=True):
    world = World("CartPole-v1", render=render)
    captured_frames = []
    for episode in range(num_episodes):
        world.reset()
        done = False
        ep_reward = 0.0
        while not done:
            obs_t = world.observe()
            with torch.no_grad():
                logits, value, proto = brain(obs_t)
            dist = torch.distributions.Categorical(logits=logits)
            action = torch.argmax(logits) if greedy else dist.sample()
 
            next_obs, reward, done, _ = world.act(action.item())
            captured_frames.append(world.env.render())
            reward_prediction_error = reward - value.item()
            brain.remember(proto=proto, action=action.item(), reward=reward, reward_prediction_error=reward_prediction_error)
 
            ep_reward += reward
        print(f"[watch] Episode {episode + 1}: reward = {ep_reward:.0f}")
 
    world.close()
    return captured_frames
 
 
if __name__ == "__main__":
    # Train headless (fast), then pop open a window to watch the result.
    # Flip to obs_type="pixels" to make the same Brain code learn from
    # raw frames instead of CartPole's hand-crafted 4-number state.
    trained_brain, rewards = train_single(num_episodes=500, render=False)
    captured_frames = watch(trained_brain, num_episodes=3, render=True)
    generate_pdf_report(rewards,captured_frames)
    
