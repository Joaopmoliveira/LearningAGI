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
                       colors=("#d62728", "#ff7f0e", "#bcbd22", "#2ca02c", "#1f77b4", "#9467bd"),
                       xlabel="Step within episode", legend_label="Episode"):
    """
    Generic version of the reward-error plot: takes ANY list of
    lists (outer index = episode, inner list = a per-step metric for
    that episode) and plots every episode's curve on one axes, colored
    on a gradient from `early_color` (earliest episodes) to `late_color`
    (latest episodes). Used for prediction error, decoding error, and
    world-evolution error alike.
    """
    n_epochs = len(list_of_lists)
    if n_epochs == 0:
        return None
 
    #cmap = mcolors.LinearSegmentedColormap.from_list("epoch_gradient", [early_color, late_color])
    cmap = mcolors.LinearSegmentedColormap.from_list("epoch_gradient", list(colors))
 
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


def plot_simple_series_curve(series, label, title, save_path="temp_reward_plot.png"):
    """
    Flat (not list-of-lists) reward-per-episode curve, with an optional
    moving average overlay. Factored out of generate_pdf_report so
    ErrorTracking.render_pdf can reuse the exact same plot.
    """
    plt.figure(figsize=(6, 3))
    plt.plot(series, label=label, color="#2b5c8f", alpha=0.6)
 
    plt.title(title)
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
        self.dream_actor_loss = []
        self.dream_entropy_loss = []
        self.dream_value_loss = []

        self.episodict_time_series = {}

    def register_variable_query(self,name,callback,reportdata):
        self.episodict_time_series[name] = {"callback":callback,"pdfdata": reportdata,"list_of_timeseries":[],"intra_timeseries":[]}

    def start_episode(self):
        for key in self.episodict_time_series:
            self.episodict_time_series[key]["intra_timeseries"] = []

    def query_measurments(self):
        for key in self.episodict_time_series:
            value = self.episodict_time_series[key]["callback"]()
            self.episodict_time_series[key]["intra_timeseries"].append(value)

    def end_episode(self,real_reward):
        self.rewards.append(real_reward)
        for key in self.episodict_time_series:
            self.episodict_time_series[key]["list_of_timeseries"].append(self.episodict_time_series[key]["intra_timeseries"])

    def append_sleep(self,actor_loss,entropy_bonus,value_loss):
        self.dream_actor_loss.append(actor_loss)
        self.dream_entropy_loss.append(entropy_bonus)
        self.dream_value_loss.append(value_loss)
        
    def render_pdf(self, eval_frames=None, filename="error_tracking_report.pdf"):
        """
        Assembles everything this class has tracked into one PDF:
        the reward curve, the sleep-phase losses, plus a red(early)->
        purple(late) trend plot for every metric registered via
        register_variable_query (each using its own section/title/
        description, so adding a new tracked metric needs no changes
        here -- just a register_variable_query call), and (optionally)
        a grid of evaluation frames if you pass some in.
        """
        sections = []  # (heading, image_path, caption) in render order
 
        if self.rewards:
            reward_plot = plot_reward_curve(self.rewards, save_path="temp_rewards.png")
            sections.append(("Learning Performance", reward_plot, None))
 
        path = plot_simple_series_curve(self.dream_actor_loss, "actor loss", "actor loss", save_path="temp_actorloss_plot.png")
        sections.append(("Actor loss", path, None))
 
        path = plot_simple_series_curve(self.dream_value_loss, "Value loss", "Value loss", save_path="temp_valueloss_plot.png")
        sections.append(("Value loss", path, None))
 
        path = plot_simple_series_curve(self.dream_entropy_loss, "Entropy loss", "Entropy loss", save_path="temp_entropyloss_plot.png")
        sections.append(("Entropy loss", path, None))
 
        for name, data in self.episodict_time_series.items():
            series = data["list_of_timeseries"]
            if not series:
                continue
            reportdata = data["pdfdata"]
            heading = reportdata.get("section", name)
            title = reportdata.get("title", heading)
            ylabel = reportdata.get("ylabel", title)
            caption = reportdata.get("description")
 
            plot_path = plot_metric_trend(
                series, title=title, ylabel=ylabel, save_path=f"temp_{name}.png",
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
    def __init__(self, proto, action, reward, priority):
        self.proto = proto
        self.action = action
        self.reward = reward
        self.priority = priority


# The memory is just a list of Associations and an attention layer to select the best memories, 
# with some hardcoded heuristics for the moment.
class Memory(nn.Module):
    def __init__(self, d_model,n_actions, query_short_memories = 5, query_long_memories = 5,num_heads=2, short_capacity=8, long_capacity=64):
        super().__init__()
        self.short_term = collections.deque(maxlen=short_capacity)
        self.long_term = []
        self.long_capacity = long_capacity
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.v_proj = nn.Linear(d_model + n_actions + 1, d_model)
        self.query_short_memories = query_short_memories
        self.query_long_memories = query_long_memories
        self.d_model = d_model
        self.n_actions = n_actions
 
    def add(self, transition):
        self.short_term.append(transition)
        if len(self.long_term) < self.long_capacity:
            self.long_term.append(transition)
        else:
            min_idx = min(range(len(self.long_term)), key=lambda i: abs(self.long_term[i].priority))
            if abs(transition.priority) > abs(self.long_term[min_idx].priority):
                self.long_term[min_idx] = transition


    def _stack_or_empty(self, items, width=None):
        if not items:
            return torch.zeros(0, width) if width is not None else torch.zeros(0, dtype=torch.long)
        return torch.stack(items) if torch.is_tensor(items[0]) else torch.tensor(items)

    def retrieve(self, query_proto):
        """
        The retrieved memories are a dictionary composed of short term and long term memories.
        Short term memories are just compacted sequencially, while long term memories are selected by
        the an attention network to retrieve the most important memories for this particular moment.
        """
        memories = {"short": {}, "long": {}}

        short_items = list(self.short_term)[-self.query_short_memories:]
        pad_size = max(0, self.query_short_memories - len(short_items))
 
        short_protos = self._stack_or_empty([t.proto for t in short_items], width=self.d_model)
        short_actions = self._stack_or_empty([t.action for t in short_items])
        short_rewards = self._stack_or_empty([t.reward for t in short_items], width=1)
        if short_rewards.dim() == 1 and short_rewards.numel() > 0:
            short_rewards = short_rewards.unsqueeze(-1)
 
        memories["short"]["protos"] = torch.cat([torch.zeros(pad_size, self.d_model), short_protos], dim=0)
        memories["short"]["actions"] = torch.cat([torch.zeros(pad_size, dtype=torch.long), short_actions.long()], dim=0)
        memories["short"]["rewards"] = torch.cat([torch.zeros(pad_size, 1), short_rewards], dim=0)
 
        if len(self.long_term) <= self.query_long_memories:
            pad_size = self.query_long_memories - len(self.long_term)
            long_protos = self._stack_or_empty([t.proto for t in self.long_term], width=self.d_model)
            long_actions = self._stack_or_empty([t.action for t in self.long_term])
            long_rewards = self._stack_or_empty([t.reward for t in self.long_term], width=1)
            if long_rewards.dim() == 1 and long_rewards.numel() > 0:
                long_rewards = long_rewards.unsqueeze(-1)
 
            memories["long"]["protos"] = torch.cat([long_protos, torch.zeros(pad_size, self.d_model)], dim=0)
            memories["long"]["actions"] = torch.cat([long_actions.long(), torch.zeros(pad_size, dtype=torch.long)], dim=0)
            memories["long"]["rewards"] = torch.cat([long_rewards, torch.zeros(pad_size, 1)], dim=0)
            return memories
            
        k_protos = torch.stack([t.proto for t in self.long_term]) 
 
        actions_onehot = F.one_hot(
            torch.tensor([t.action for t in self.long_term]), num_classes=self.n_actions
        ).float()      
        rewards = torch.tensor([t.reward for t in self.long_term]).float().unsqueeze(-1)
        
        v_raw = torch.cat([k_protos, actions_onehot, rewards], dim=-1)
        v_embeds = self.v_proj(v_raw)
 
        query = query_proto.view(1, 1, self.d_model)
        key = k_protos.unsqueeze(0)
        values = v_embeds.unsqueeze(0)
 
        _, attn_weights = self.attention(query, key, values)
        attn_weights = attn_weights.squeeze(0).squeeze(0)
 
        actual_k = min(self.query_long_memories, len(self.long_term))
        _, top_indices = torch.topk(attn_weights, k=actual_k)
        
        memories["long"] = {
                        "protos": k_protos[top_indices],
                        "actions": torch.tensor([self.long_term[i].action for i in top_indices], dtype=torch.long),
                        "rewards": torch.tensor([self.long_term[i].reward for i in top_indices], dtype=torch.float32).unsqueeze(-1)
                    }
        return memories


    def random_long_term_memory(self) :
        return random.sample(self.long_term, 1)[0]

    def freeze_weights(self):
        for param in self.parameters():
            param.requires_grad = False

    def activate_weights(self):
        for param in self.parameters():
            param.requires_grad = True 

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
            nn.Sigmoid(), 
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
# Shared building block for Cortex and ValueCenter: both need to see the
# current protomemory (the "now" state), a temporal-conv summary of the
# short-term window (proto+action+reward per recent step -> velocity /
# acceleration-like features), and the attention-selected long-term payload.
# Kept as two separate instances (not shared weights) since actor and critic
# optimize different objectives and sharing a trunk is a known source of
# actor/critic optimization interference.
class DualBranchHead(nn.Module):
    def __init__(self, d_model, n_actions, query_short_memory, query_long_memory, out_dim=32):
        super().__init__()
        step_dim = d_model + n_actions + 1  # proto + one-hot action + reward, per step/memory
        assert query_short_memory >= 3, "temporal_conv needs at least 3 short-term steps (two kernel-2 convs)"
 
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(in_channels=step_dim, out_channels=64, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * (query_short_memory - 2), out_dim),
            nn.ReLU(),
        )
 
        long_term_dim = query_long_memory * step_dim
        self.fused_dim = d_model + out_dim + long_term_dim
 
    def fuse(self, protomemory, short_term_window, long_term_payload):
        single = protomemory.dim() == 1
        if single:
            protomemory = protomemory.unsqueeze(0)
            short_term_window = short_term_window.unsqueeze(0)
            long_term_payload = long_term_payload.unsqueeze(0)
 
        x = short_term_window.transpose(1, 2)     # (batch, step_dim, seq_len) for Conv1d
        motion_features = self.temporal_conv(x)   # (batch, out_dim)
 
        fused = torch.cat([protomemory, motion_features, long_term_payload], dim=-1)
        return fused, single
 
    def freeze_weights(self):
        for param in self.parameters():
            param.requires_grad = False
 
    def activate_weights(self):
        for param in self.parameters():
            param.requires_grad = True


class Cortex(DualBranchHead):
    def __init__(self, d_model, n_actions, query_short_memory, query_long_memory, out_dim=32):
        super().__init__(d_model, n_actions, query_short_memory, query_long_memory, out_dim)
        self.net = nn.Sequential(
            nn.Linear(self.fused_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_actions),
        )
 
    def forward(self, protomemory, short_term_window, long_term_payload):
        fused, single = self.fuse(protomemory, short_term_window, long_term_payload)
        out = self.net(fused)
        return out.squeeze(0) if single else out
 
 
class ValueCenter(DualBranchHead):
    def __init__(self, d_model, n_actions, query_short_memory, query_long_memory, out_dim=32):
        super().__init__(d_model, n_actions, query_short_memory, query_long_memory, out_dim)
        self.net = nn.Sequential(
            nn.Linear(self.fused_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )
 
    def forward(self, protomemory, short_term_window, long_term_payload):
        fused, single = self.fuse(protomemory, short_term_window, long_term_payload)
        out = self.net(fused).squeeze(-1)
        return out.squeeze(0) if single else out
 

class DreamerCenter(nn.Module):
    def __init__(self, d_model=32, n_actions=2, action_dim=16, hidden_dim=128):
        super().__init__()
        self.d_model = d_model
        self.action_embed = nn.Embedding(n_actions, action_dim)

        input_dim = d_model + action_dim
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        ## this predicts the next state given the actions and the previous world state
        self.next_state_head = nn.Linear(hidden_dim, d_model)
        ## this predicts the expected reward from the world
        self.reward_head = nn.Linear(hidden_dim, 1)
        ## this predicts the probability of losing the game 
        self.world_over_head = nn.Linear(hidden_dim, 1)
 

    def forward(self, z, action):
        single = z.dim() == 1
        if single:
            z = z.unsqueeze(0)
        if action.dim() == 0:
            action = action.unsqueeze(0)

        a_emb = self.action_embed(action)
        h = self.trunk(torch.cat([z, a_emb], dim=-1))

        delta_z = self.next_state_head(h)
        z_next = z + delta_z
        reward_pred = self.reward_head(h).squeeze(-1)
        done_logit = self.world_over_head(h).squeeze(-1)

        if single:
            return z_next.squeeze(0), reward_pred.squeeze(0),done_logit.squeeze(0)
        return z_next, reward_pred, done_logit


    def freeze_weights(self):
        for param in self.parameters():
            param.requires_grad = False

    def activate_weights(self):
        for param in self.parameters():
            param.requires_grad = True   

class Agent(nn.Module):
    def __init__(self, world, d_model=32, num_heads=2, lr=1e-3, gamma=0.99, entropy_coef=0.05):
        super().__init__()
        self.d_model = d_model
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.is_asleep = False
        
        frame_stack, img_size, _ = world.obs_shape
        self.memory = Memory(d_model=d_model, query_short_memories=5, query_long_memories=5,
                             n_actions=world.n_actions, num_heads=num_heads)
        self.perception = Perception(frame_stack, img_size, d_model)
        self.decoder = Projection(self.perception) 
        self.dreamer = DreamerCenter(d_model, world.n_actions)
 
        self.n_actions = world.n_actions
        
        self.cortex = Cortex(d_model, world.n_actions, self.memory.query_short_memories, self.memory.query_long_memories)
        self.value_center = ValueCenter(d_model, world.n_actions, self.memory.query_short_memories, self.memory.query_long_memories)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
 
    def waking_up(self):
        self.cortex.freeze_weights()
        self.value_center.freeze_weights()
        self.dreamer.activate_weights()
        self.perception.activate_weights()
        self.decoder.activate_weights()
        self.is_asleep = False
 
    def _memory_features(self, retrieved):
        # Turns Memory.retrieve()'s {"short": {...}, "long": {...}} dict into
        # the two tensors Cortex/ValueCenter expect: a (seq_len, d_model+n_actions+1)
        # short-term window for the temporal conv, and a flattened long-term payload.
        short = retrieved["short"]
        short_actions_onehot = F.one_hot(short["actions"], num_classes=self.n_actions).float()
        short_window = torch.cat([short["protos"], short_actions_onehot, short["rewards"]], dim=-1)
 
        long = retrieved["long"]
        long_actions_onehot = F.one_hot(long["actions"], num_classes=self.n_actions).float()
        long_payload = torch.cat([long["protos"], long_actions_onehot, long["rewards"]], dim=-1).view(-1)
 
        return short_window, long_payload
 
    def awake_cycle(self, worldview):
        protomemory = self.perception(worldview)  
        retrieved = self.memory.retrieve(protomemory)
        short_window, long_payload = self._memory_features(retrieved)
 
        action_logits = self.cortex(protomemory, short_window, long_payload)
        predicted_value = self.value_center(protomemory, short_window, long_payload)
        recreated_world_view = self.decoder(protomemory)
 
        dist = torch.distributions.Categorical(logits=action_logits)
        action = dist.sample()
 
        predicted_next_embedding,predicted_reward,predicted_losing_probability = self.dreamer(protomemory, action)
 
        return (action_logits.detach(), action.detach(), predicted_value, predicted_reward,predicted_losing_probability,
                recreated_world_view, protomemory.detach(), predicted_next_embedding)
 
    @torch.no_grad()
    def get_value(self, worldview):
        proto = self.perception(worldview)
        retrieved = self.memory.retrieve(proto)
        short_window, long_payload = self._memory_features(retrieved)
        return self.value_center(proto, short_window, long_payload).squeeze()
 
    def going_to_sleep(self):
        self.cortex.activate_weights()
        self.value_center.activate_weights()
        self.dreamer.freeze_weights()
        self.perception.freeze_weights()
        self.decoder.freeze_weights()
        self.is_asleep = True
 
    def sleep(self, number_of_dreams,lambda_=0.95):
        log_probs, values, rewards, entropies, continue_prob = [], [], [], [], []
        
        dreamed_next_embedding = self.memory.random_long_term_memory().proto
 
        for i in range(number_of_dreams):
            retrieved = self.memory.retrieve(dreamed_next_embedding)
            short_window, long_payload = self._memory_features(retrieved)
 
            action_logits = self.cortex(dreamed_next_embedding, short_window, long_payload)
 
            dist = torch.distributions.Categorical(logits=action_logits)
            action = dist.sample()
 
            predicted_value = self.value_center(dreamed_next_embedding, short_window, long_payload)
            dreamed_next_embedding, predicted_reward,predicted_losing_probability = self.dreamer(dreamed_next_embedding, action)
            predicted_continue_prob = torch.sigmoid(-predicted_losing_probability)
 
            continue_prob.append(predicted_continue_prob.detach())
            log_probs.append(dist.log_prob(action))
            values.append(predicted_value.squeeze())
            rewards.append(predicted_reward.detach()) 
            entropies.append(dist.entropy())
 
        with torch.no_grad():
            retrieved = self.memory.retrieve(dreamed_next_embedding)
            short_window, long_payload = self._memory_features(retrieved)
            R = self.value_center(dreamed_next_embedding, short_window, long_payload).squeeze()
 
        returns = [R]
        for t in reversed(range(len(rewards) - 1)):
            r_t = rewards[t].detach()
            c_t = continue_prob[t].detach()
            v_next = values[t + 1].detach()
            
            R = r_t + self.gamma * c_t * ((1 - lambda_) * v_next + lambda_ * R)
            returns.insert(0, R)
 
        returns_t = torch.stack(returns[:-1])
        values_t = torch.stack(values[:-1])
        advantages = (returns_t - values_t).detach()
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        actor_loss = -(torch.stack(log_probs[:-1]) * advantages).mean()
        entropy_bonus = torch.stack(entropies[:-1]).mean()
        value_loss = F.mse_loss(values_t, returns_t.detach())
    
        loss = actor_loss - self.entropy_coef * entropy_bonus + value_loss
 
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()
 
        return actor_loss.detach(),entropy_bonus.detach(),value_loss.detach()
 
    def remember(self, proto, action, reward, priority):
        self.memory.add(Association(proto, action, reward, priority))
 
    def freeze_all(self):
        self.cortex.freeze_weights()
        self.dreamer.freeze_weights()
        self.value_center.freeze_weights()
        self.perception.freeze_weights()
        self.decoder.freeze_weights()
 
def train(num_episodes=1000, number_of_dreams=20, lr=1e-3, log_every=20, render=False):
    world = World("CartPole-v1", render=render)
    agent = Agent(world, d_model=32, lr=lr)
    error_tracker = ErrorTracking()

    temp_episodic_metric_storage = {"reward_error" : 0.0,
                                    "encoder_decoder_error" : 0.0,
                                    "predict_next_word_state": 0.0,
                                    "done_loss" : 0.0
                                    }


    error_tracker.register_variable_query(name = "reward_error",
                                          callback= lambda: temp_episodic_metric_storage["reward_error"],
                                          reportdata={
                                              "section":"Reward Prediction Error",
                                              "title":"Reward Prediction Error",
                                              "description":"The plot shows the error in prediction the reward as time progresses for each episode. The color coding represents the passage of episodes"
                                              }
                                          )
    error_tracker.register_variable_query(name = "encoder_decoder_error",
                                          callback= lambda: temp_episodic_metric_storage["encoder_decoder_error"],
                                          reportdata={
                                              "section":"Encoder-Decoder Error",
                                              "title":"Decorder Error",
                                              "description":"The plot shows the error in transforming the latent space into the real world observation. The color coding represents the passage of episodes"
                                              }
                                          )
    error_tracker.register_variable_query(name = "predict_next_word_state",
                                          callback= lambda: temp_episodic_metric_storage["predict_next_word_state"],
                                          reportdata={
                                              "section":"Predicting World State from Action-Previous State",
                                              "title":"World State Error (Action-Previous State)",
                                              "description":"The plot shows the error in predicting the next latent spate (condensed world representation) given the action and the previous world representation. The color coding represents the passage of episodes"
                                              }
                                          )
    error_tracker.register_variable_query(name = "done_loss",
                                          callback= lambda: temp_episodic_metric_storage["done_loss"],
                                          reportdata={
                                              "section":"Predicting Termination of Game",
                                              "title":"Done Losss",
                                              "description":"The plot shows the error of the model in predicting termination. The color coding represents the passage of episodes"
                                              }
                                          )

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
 
            (_, action, predicted_value , predicted_reward,predicted_losing_probability,
                            recreated_world_view, protomemory, predicted_next_embedding) = agent.awake_cycle(viewed_world)

           
            _, reward, done, _ = world.act(action.item())
            next_obs = world.observe()
            with torch.no_grad():
                next_proto_target = agent.perception(next_obs)
                next_value = 0.0 if done else agent.get_value(next_obs).item()
                td_error = reward + agent.gamma * next_value - predicted_value.item()

            recon_loss = F.mse_loss(recreated_world_view, viewed_world)
            reward_loss = F.mse_loss(predicted_reward, torch.tensor(reward, dtype=torch.float32))
            dynamics_loss = F.mse_loss(predicted_next_embedding, next_proto_target)
            done_loss = F.binary_cross_entropy_with_logits(
                predicted_losing_probability, torch.tensor(float(done), dtype=torch.float32)
            )
            awake_loss = recon_loss + reward_loss + dynamics_loss + done_loss

            temp_episodic_metric_storage["reward_error"] = reward_loss.detach()
            temp_episodic_metric_storage["encoder_decoder_error"] = recon_loss.detach()
            temp_episodic_metric_storage["predict_next_word_state"] = dynamics_loss.detach()
            temp_episodic_metric_storage["done_loss"] = done_loss.detach()
            error_tracker.query_measurments()
 
            awake_optimizer.zero_grad()
            awake_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.0)
            awake_optimizer.step()
 
            agent.remember(
                proto=protomemory,
                action=action.item(),
                reward=reward,
                priority = abs(td_error)
            )
 
            ep_reward += reward
        #print("!")
        error_tracker.end_episode(ep_reward)
        agent.going_to_sleep()
 
        if len(agent.memory) > 0:
            actor_loss_list = 0
            entropy_bonus_list = 0 
            value_loss_list = 0
            for _ in range(5):
                actor_loss,entropy_bonus,value_loss = agent.sleep(number_of_dreams=5)
                actor_loss_list += actor_loss
                entropy_bonus_list += entropy_bonus
                value_loss_list += value_loss
            
            error_tracker.append_sleep(actor_loss_list,entropy_bonus_list,value_loss)
        if (episode + 1) % log_every == 0:
            avg = np.mean(error_tracker.rewards[-log_every:])
            print(f"Episode {episode + 1:4d} | Avg Real Reward (last {log_every}): {avg:.1f}")
 
    return agent, error_tracker
 
 
def watch(agent, num_episodes=1, render=True):
    world = World("CartPole-v1", render=render)
    captured_frames = []
    for episode in range(num_episodes):
        world.reset()
        done = False
        ep_reward = 0.0
        while not done:
            obs_t = world.observe()
            with torch.no_grad():
                (action_logits, action, predicted_value, predicted_reward,predicted_losing_probability,
                                            recreated_world_view, protomemory, predicted_next_embedding) = agent.awake_cycle(obs_t)
                 
            action = torch.argmax(action_logits)
            _, reward, done, _ = world.act(action.item())
            next_obs = world.observe()
            with torch.no_grad():
                next_proto_target = agent.perception(next_obs)
                next_value = 0.0 if done else agent.get_value(next_obs).item()
                td_error = reward + agent.gamma * next_value - predicted_value.item()
            captured_frames.append(world.env.render())
            agent.remember(proto=protomemory, action=action.item(), reward=reward, priority=abs(td_error))
 
            ep_reward += reward
        print(f"[watch] Episode {episode + 1}: reward = {ep_reward:.0f}")
 
    world.close()
    return captured_frames
 
 
if __name__ == "__main__":
    # The loop is simple:
    # we train it for a number of episodes
    trained_brain, error_tracker = train(num_episodes=1000, render=False)
    trained_brain.freeze_all()
    # we then just force the brain to move in the world
    captured_frames = watch(trained_brain, num_episodes=1, render=True)
    ## Because we are cool we generate a pdf file with 
    # a report detailing the statistics during training
    error_tracker.render_pdf(captured_frames)    