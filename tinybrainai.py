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
import matplotlib.colors as mcolors


def plot_metric_trend(list_of_lists, title, ylabel, save_path,
                       early_color="red", late_color="blue",
                       xlabel="Step within episode", legend_label="Episode"):
    """
    Generic version of the earlier reward-error plot: takes ANY list of
    lists (outer index = episode, inner list = a per-step metric for
    that episode) and plots every episode's curve on one axes, colored
    on a gradient from `early_color` (earliest episodes) to `late_color`
    (latest episodes). Used for prediction error, decoding error, and
    world-evolution error alike -- only the title/ylabel differ.
    """
    n_epochs = len(list_of_lists)
    if n_epochs == 0:
        return None
 
    cmap = mcolors.LinearSegmentedColormap.from_list("epoch_gradient", [early_color, late_color])
 
    plt.figure(figsize=(7, 4))
    for i, series in enumerate(list_of_lists):
        if not series:
            continue
        t = i / max(1, n_epochs - 1)  # 0.0 for the first episode, 1.0 for the last
        plt.plot(series, color=cmap(t), alpha=0.5, linewidth=1)
 
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.4)
 
    # Colorbar doubles as the legend for the red->blue = early->late gradient
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=n_epochs - 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca())
    cbar.set_label(legend_label)
 
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    return save_path
 
 
def plot_reward_error_trends(reward_error_history, save_path="temp_rpe_trend.png",
                              early_color="red", late_color="blue"):
    """Thin wrapper kept for backwards compatibility -- see plot_metric_trend."""
    return plot_metric_trend(
        reward_error_history,
        title="Reward Prediction Error per Step, Across Training",
        ylabel="Reward Prediction Error",
        save_path=save_path,
        early_color=early_color,
        late_color=late_color,
    )
 
 
def plot_reward_curve(rewards, save_path="temp_reward_plot.png", moving_avg_window=20):
    """
    Flat (not list-of-lists) reward-per-episode curve, with an optional
    moving average overlay. Factored out of generate_pdf_report so
    ErrorTracking.render_pdf can reuse the exact same plot.
    """
    plt.figure(figsize=(6, 3))
    plt.plot(rewards, label="Episode Reward", color="#2b5c8f", alpha=0.6)
 
    if len(rewards) >= moving_avg_window:
        ma = [np.mean(rewards[max(0, i - moving_avg_window):i + 1]) for i in range(len(rewards))]
        plt.plot(ma, label=f"{moving_avg_window}-Ep Moving Avg", color="#d95f02", linewidth=2)
 
    plt.title("Agent Learning Progress")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    return save_path


class ErrorTracking:
    def __init__(self):
        self.rewards = []
        self.episode_prediction_errors = []
        self.episode_decoding_errors = []
        self.episode_world_evolution_errors = []

        self.intra_episode_prediction_errors = []
        self.intra_episode_decoding_errors = []
        self.intra_episode_world_evolution_errors = []

        self.episode_actor_loss = []
        self.episode_entropy_loss = []

    def start_episode(self):
        self.intra_episode_prediction_errors = []
        self.intra_episode_decoding_errors = []
        self.intra_episode_world_evolution_errors = []

    def append_measurments(self,prediction_error,decoding_error,world_error):
        self.intra_episode_prediction_errors.append(prediction_error)
        self.intra_episode_decoding_errors.append(decoding_error)
        self.intra_episode_world_evolution_errors.append(world_error)

    def append_sleep(self,actor_loss,entropy_bonus):
        self.episode_actor_loss.append(actor_loss)
        self.episode_entropy_loss.append(entropy_bonus)
        
    def end_episode(self,real_reward):
        self.episode_prediction_errors.append(self.intra_episode_prediction_errors)
        self.episode_decoding_errors.append(self.intra_episode_decoding_errors)
        self.episode_world_evolution_errors.append(self.intra_episode_world_evolution_errors)
        self.rewards.append(real_reward)


    def render_pdf(self, eval_frames=None, filename="error_tracking_report.pdf"):
        """
        Assembles everything this class has tracked into one PDF:
        the reward curve, plus a red(early)->blue(late) trend plot for
        each of the three intra-episode error series, plus (optionally)
        a grid of evaluation frames if you pass some in.
        """
        sections = []  # (heading, image_path, caption) in render order
 
        if self.rewards:
            reward_plot = plot_reward_curve(self.rewards, save_path="temp_rewards.png")
            sections.append(("Learning Performance", reward_plot, None))
 
        error_series = [
            ("Prediction Error Trend", self.episode_prediction_errors,
             "Reward/next-state prediction error", "temp_prediction_error.png"),
            ("Decoding Error Trend", self.episode_decoding_errors,
             "Decoding (reconstruction) error", "temp_decoding_error.png"),
            ("World Evolution Error Trend", self.episode_world_evolution_errors,
             "World-evolution (dynamics) error", "temp_world_error.png"),
            ("Entropy Error Trend", self.episode_entropy_loss,
              "Entropy error evolution", "temp_entropy.png"),
            ("Actor Loss Trend", self.episode_actor_loss,
             "Actor loss evolution", "temp_actor_loss.png")
        ]
        caption = ("Each line is one episode's error over its steps. Color moves from "
                   "red (earliest episodes) to blue (latest episodes) to reveal whether "
                   "error is shrinking across training.")
        for heading, series, ylabel, path in error_series:
            if not series:
                continue
            plot_path = plot_metric_trend(
                series, title=heading, ylabel=ylabel, save_path=path,
            )
            sections.append((heading, plot_path, caption))
 
        # Optional evaluation frames, reusing the same 2x2 grid logic
        # generate_pdf_report uses.
        frame_paths = []
        if eval_frames:
            step_size = max(1, len(eval_frames) // 4)
            for idx, frame in enumerate(eval_frames[::step_size][:4]):
                img_path = f"temp_frame_{idx}.png"
                if isinstance(frame, np.ndarray):
                    if frame.dtype != np.uint8:
                        frame = (frame * 255).astype(np.uint8)
                    Image.fromarray(frame).save(img_path)
                frame_paths.append(img_path)
 
        pdf = FPDF()
        pdf.add_page()
 
        pdf.set_font("Helvetica", "B", 18)
        pdf.cell(0, 10, "Brain Model Error Tracking Report", new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.ln(5)
 
        pdf.set_font("Helvetica", "", 11)
        pdf.cell(0, 6, f"Total Episodes: {len(self.rewards)}", new_x="LMARGIN", new_y="NEXT")
        if self.rewards:
            pdf.cell(0, 6, f"Final Reward: {self.rewards[-1]:.2f}", new_x="LMARGIN", new_y="NEXT")
            pdf.cell(0, 6, f"Peak Reward: {max(self.rewards):.2f}", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(5)
 
        for i, (heading, image_path, section_caption) in enumerate(sections, start=1):
            if i > 1:
                pdf.add_page()
            pdf.set_font("Helvetica", "B", 13)
            pdf.cell(0, 8, f"{i}. {heading}", new_x="LMARGIN", new_y="NEXT")
            if section_caption:
                pdf.ln(2)
                pdf.set_font("Helvetica", "", 10)
                pdf.multi_cell(0, 5, section_caption)
            pdf.ln(2)
            pdf.image(image_path, x=15, w=180)
 
        if frame_paths:
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 13)
            pdf.cell(0, 8, f"{len(sections) + 1}. Replicated Behavior (Evaluation Rollout)",
                     new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)
            x_coords = [15, 105, 15, 105]
            y_start = pdf.get_y()
            for i, fpath in enumerate(frame_paths):
                y_pos = y_start if i < 2 else y_start + 55
                pdf.image(fpath, x=x_coords[i], y=y_pos, w=85)
 
        pdf.output(filename)
 
        # Clean up temp images
        for _, image_path, _ in sections:
            if os.path.exists(image_path):
                os.remove(image_path)
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

    def snapshot(self, k_longterm=4):
        if not self.short_term:
            return None

        sequential_protos = [t.proto for t in self.short_term]
        long_protos = []
        if self.long_term:
            k = min(k_longterm, len(self.long_term))
            long_protos = [t.proto for t in random.sample(self.long_term, k)]
        return torch.stack(sequential_protos + long_protos)

    def random_long_term_memory(self) :
        return random.sample(self.long_term, 1)[0]

    def __len__(self):
        return len(self.long_term)


## The conv encoder is supposed to be a backbone 
# of some YOLO famous network, the goal is that this is a 
# thing that generates an embbedding of the world
class Perception(nn.Module):
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

    def freeze_weights(self):
        for param in self.parameters():
            param.requires_grad = False

    def activate_weights(self):
        for param in self.parameters():
            param.requires_grad = True   

## The conv dencoder is supposed to be something that can 
# interpret the embedding of the encoder and recreate the original image
# from the world. This should be used to understand if the model is correctly 
# simulating the world
class Projection(nn.Module):
    def __init__(self, encoder: Perception):
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

    def freeze_weights(self):
        for param in self.parameters():
            param.requires_grad = False

    def activate_weights(self):
        for param in self.parameters():
            param.requires_grad = True   

## The cortex is the magic center, it takes the attention, 
# memories, protomemory and generates an action

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

    def freeze_weights(self):
        for param in self.parameters():
            param.requires_grad = False

    def activate_weights(self):
        for param in self.parameters():
            param.requires_grad = True   
 
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

    def freeze_weights(self):
        for param in self.parameters():
            param.requires_grad = False

    def activate_weights(self):
        for param in self.parameters():
            param.requires_grad = True   

class DreamerCenter(nn.Module):
    def __init__(self, d_model=32, n_actions=2, action_dim=16, hidden_dim=128):
        super().__init__()
        self.d_model = d_model
        self.action_embed = nn.Embedding(n_actions, action_dim)
        
        input_dim = d_model + action_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, d_model)  # Outputs delta_z
        )
 
    def forward(self, z, action):
        single = z.dim() == 1
        if single:
            z = z.unsqueeze(0)
        if action.dim() == 0:
            action = action.unsqueeze(0)
 
        a_emb = self.action_embed(action)
 
        x = torch.cat([z, a_emb], dim=-1)

        delta_z = self.net(x)
        z_next = z + delta_z
 
        return z_next.squeeze(0) if single else z_next


    def freeze_weights(self):
        for param in self.parameters():
            param.requires_grad = False

    def activate_weights(self):
        for param in self.parameters():
            param.requires_grad = True   

class Agent(nn.Module):
    def __init__(self, world, d_model=32, num_heads=2, lr=1e-3, gamma=0.99, entropy_coef=0.01):
        super().__init__()
        self.d_model = d_model
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.is_asleep = False
        
        frame_stack, img_size, _ = world.obs_shape
        self.memory = Memory()
        self.perception = Perception(frame_stack, img_size, d_model)
        self.decoder = Projection(self.perception) 
        self.dreamer = DreamerCenter(d_model, world.n_actions) # Expects (z, action)
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        
        self.cortex = Cortex(d_model, world.n_actions)
        self.reward_center = RewardCenter(d_model)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def waking_up(self):
        self.cortex.freeze_weights()
        self.dreamer.activate_weights()
        self.reward_center.activate_weights()
        self.perception.activate_weights()
        self.decoder.activate_weights()
        self.is_asleep = False

    def awake_cycle(self, worldview):
        mem_snap = self.memory.snapshot()
        protomemory = self.perception(worldview)  
        query = protomemory.unsqueeze(0).unsqueeze(0) 

        if mem_snap is not None:
            kv = mem_snap.unsqueeze(0)
            attended, _ = self.attention(query, kv, kv)
            attended = attended.squeeze(0).squeeze(0)
        else:
            attended = torch.zeros_like(protomemory)

        fused = torch.cat([protomemory, attended], dim=-1)
        action_logits = self.cortex(fused)
        predicted_reward = self.reward_center(fused)
        recreated_world_view = self.decoder(protomemory)

        dist = torch.distributions.Categorical(logits=action_logits)
        action = dist.sample()

        predicted_next_embedding = self.dreamer(protomemory.detach(), action.detach())

        return (action_logits.detach(), action.detach(), predicted_reward,
                recreated_world_view, protomemory.detach(), predicted_next_embedding)


    def going_to_sleep(self):
        self.cortex.activate_weights()
        self.dreamer.freeze_weights()
        self.reward_center.freeze_weights()
        self.perception.freeze_weights()
        self.decoder.freeze_weights()
        self.is_asleep = True

    def sleep(self, number_of_dreams, error_reporting):
        log_probs, predicted_rewards, rewards, entropies = [], [], [], []
        mem_snap = self.memory.snapshot()
        dream = self.memory.random_long_term_memory().proto

        for i in range(number_of_dreams):
            query = dream.unsqueeze(0).unsqueeze(0) 

            if mem_snap is not None:
                kv = mem_snap.unsqueeze(0)
                attended, _ = self.attention(query, kv, kv)
                attended = attended.squeeze(0).squeeze(0)
            else:
                attended = torch.zeros_like(dream)

            fused = torch.cat([dream, attended], dim=-1)
            action_logits = self.cortex(fused)

            dist = torch.distributions.Categorical(logits=action_logits)
            action = dist.sample()

            predicted_reward = self.reward_center(fused)
            
            dream = self.dreamer(dream, action)

            log_probs.append(dist.log_prob(action))
            predicted_rewards.append(predicted_reward)
            rewards.append(predicted_reward.detach()) # Use predicted reward in dreams
            entropies.append(dist.entropy())

        # ---- Discounted returns & Advantage calculation ----
        returns, R = [], 0.0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns, dtype=torch.float32)
        values_t = torch.stack(predicted_rewards).squeeze()
        advantages = returns - values_t.detach()
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        log_probs_t = torch.stack(log_probs)
        entropies_t = torch.stack(entropies)

        actor_loss = -(log_probs_t * advantages).mean()
        entropy_bonus = entropies_t.mean()
        loss = actor_loss - self.entropy_coef * entropy_bonus

        error_reporting.append_sleep(actor_loss.detach(),entropy_bonus.detach())

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()

    def remember(self, proto, action, reward, reward_prediction_error):
        self.memory.add(Association(proto, action, reward, reward_prediction_error))

    def freeze_all(self):
        self.cortex.freeze_weights()
        self.dreamer.freeze_weights()
        self.reward_center.freeze_weights()
        self.perception.freeze_weights()
        self.decoder.freeze_weights()

def train(num_episodes=1000, number_of_dreams=20, lr=1e-3, log_every=20, render=False):
    world = World("CartPole-v1", render=render)
    agent = Agent(world, d_model=32, lr=lr)
    error_tracker = ErrorTracking()
 
    for episode in range(num_episodes):
        agent.waking_up()
        error_tracker.start_episode()
 
        awake_optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, agent.parameters()), lr=lr
        )
 
        world.reset()
        done = False
        ep_reward = 0.0

        while not done:
            viewed_world = world.observe()
 
            (action_logits, action, pred_reward, recon_world,
             proto, pred_next_proto) = agent.awake_cycle(viewed_world)
 
            _, reward, done, _ = world.act(action.item())
 
            # 3. Compute ground truth target for next frame's embedding
            # (world.act() already updated world.world_frame; observe() is
            # what turns that into the same kind of tensor perception expects)
            next_obs = world.observe()
            with torch.no_grad():
                next_proto_target = agent.perception(next_obs)
 
            recon_loss = F.mse_loss(recon_world, viewed_world)
            reward_loss = F.mse_loss(pred_reward.squeeze(), torch.tensor(reward, dtype=torch.float32))
            dynamics_loss = F.mse_loss(pred_next_proto, next_proto_target)
 
            awake_loss = recon_loss + reward_loss + dynamics_loss
            error_tracker.append_measurments(prediction_error=reward_loss.detach(),
                                            decoding_error=recon_loss.detach(),
                                            world_error=dynamics_loss.detach())

            # 5. Backpropagate & step perception/world model weights
            awake_optimizer.zero_grad()
            awake_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.0)
            awake_optimizer.step()
 
            agent.remember(
                proto=proto,
                action=action.item(),
                reward=reward,
                reward_prediction_error=abs(reward - pred_reward.item())
            )
 
            ep_reward += reward

        error_tracker.end_episode(ep_reward)
        agent.going_to_sleep()

        if len(agent.memory) > 0:
            for _ in range(number_of_dreams):
                agent.sleep(number_of_dreams=20,error_reporting=error_tracker)

        if (episode + 1) % log_every == 0:
            avg = np.mean(error_tracker.rewards[-log_every:])
            print(f"Episode {episode + 1:4d} | Avg Real Reward (last {log_every}): {avg:.1f}")
 
    return agent, error_tracker

 
def watch(agent, num_episodes=100, render=True):
    world = World("CartPole-v1", render=render)
    captured_frames = []
    for episode in range(num_episodes):
        world.reset()
        done = False
        ep_reward = 0.0
        while not done:
            obs_t = world.observe()
            with torch.no_grad():
                (action_logits, action, pred_reward, recon_world,proto, pred_next_proto) = agent.awake_cycle(obs_t)
            action = torch.argmax(action_logits)
            next_obs, reward, done, _ = world.act(action.item())
            captured_frames.append(world.env.render())
            reward_prediction_error = reward - pred_reward.item()
            agent.remember(proto=proto, action=action.item(), reward=reward, reward_prediction_error=reward_prediction_error)
 
            ep_reward += reward
        print(f"[watch] Episode {episode + 1}: reward = {ep_reward:.0f}")
 
    world.close()
    return captured_frames
 
 
if __name__ == "__main__":
    # The loop is simple:
    # we train it for a number of episodes
    trained_brain, error_tracker = train(num_episodes=500, render=False)
    trained_brain.freeze_all()
    # we then just force the brain to move in the world
    captured_frames = watch(trained_brain, num_episodes=10, render=True)
    error_tracker.render_pdf(captured_frames)    
