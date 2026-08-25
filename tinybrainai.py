import collections
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from PIL import Image
import cv2

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
        proto = self.perception(obs)                     # (d_model,)
        query = proto.unsqueeze(0).unsqueeze(0)           # (1, 1, d_model)
 
        if mem_snap is not None:
            kv = mem_snap.unsqueeze(0)             # (1, k, d_model)
            attended, _ = self.attention(query, kv, kv)
            attended = attended.squeeze(0).squeeze(0)      # (d_model,)
        else:
            attended = torch.zeros_like(proto)
 
        fused = torch.cat([proto, attended], dim=-1)       # (2*d_model,)
        action_logits = self.cortex(fused)
        value = self.reward_center(fused)
        return action_logits, value, proto.detach()


def train(num_episodes=1000, gamma=0.99, lr=1e-3, entropy_coef=0.01, log_every=20,render=False):
    world = World("CartPole-v1", render=render)
    brain = Brain(world)

    optimizer = torch.optim.Adam(brain.parameters(), lr=lr)
 
    episode_rewards = []
 
    for episode in range(num_episodes):
        world.reset()
        done = False
        ep_reward = 0.0
        log_probs, values, rewards, entropies = [], [], [], []
 
        while not done:
            viewed_world = world.observe()
            logits, value, proto = brain(viewed_world)
 
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
 
            next_obs, reward, done, _ = world.act(action.item())
 
            log_probs.append(dist.log_prob(action))
            values.append(value)
            rewards.append(reward)
            entropies.append(dist.entropy())
 
            # crude "surprise" signal used purely to decide what's worth
            # remembering; the real learning signal is the loss below
            reward_prediction_error = reward - value.item()
            brain.remember(proto=proto, action=action.item(), reward=reward, reward_prediction_error=reward_prediction_error)
 
            ep_reward += reward
 
        # ---- discounted returns and advantages ----
        returns = []
        R = 0.0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32)
        values_t = torch.stack(values)
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
 
 
def watch(brain, num_episodes=100, greedy=True, render=True):
    world = World("CartPole-v1", render=render)
 
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
            reward_prediction_error = reward - value.item()
            brain.remember(proto=proto, action=action.item(), reward=reward, reward_prediction_error=reward_prediction_error)
 
            ep_reward += reward
        print(f"[watch] Episode {episode + 1}: reward = {ep_reward:.0f}")
 
    world.close()
 
 
if __name__ == "__main__":
    # Train headless (fast), then pop open a window to watch the result.
    # Flip to obs_type="pixels" to make the same Brain code learn from
    # raw frames instead of CartPole's hand-crafted 4-number state.
    trained_brain, rewards = train(num_episodes=500, render=False)
    watch(trained_brain, num_episodes=3, render=True)
