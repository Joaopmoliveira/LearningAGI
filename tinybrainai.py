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
        self.dream_continue_prob_var = []
        self.dream_continue_prob_mean = []

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

    def append_sleep(self,actor_loss,entropy_bonus,value_loss,continue_prob_var, continue_prob_mean):
        self.dream_actor_loss.append(actor_loss)
        self.dream_entropy_loss.append(entropy_bonus)
        self.dream_value_loss.append(value_loss)
        self.dream_continue_prob_var.append(continue_prob_var)
        self.dream_continue_prob_mean.append(continue_prob_mean)

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

        path = plot_simple_series_curve(self.dream_continue_prob_var, "continue_prob variance",
                                        "Continue-Prob Variance (dream)", save_path="temp_cpvar_plot.png")
        sections.append(("Continue-Prob Variance", path, None))

        path = plot_simple_series_curve(self.dream_continue_prob_mean, "continue_prob mean",
                                        "Continue-Prob Mean (dream)", save_path="temp_cpmean_plot.png")
        sections.append(("Continue-Prob Mean", path, None))

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
    def __init__(self, proto, action_embedding, reward, priority):
        self.proto = proto
        self.action_embedding = action_embedding
        self.reward = float(reward)
        self.priority = priority

    def compact(self):
        reward_t = torch.tensor([self.reward], dtype=self.proto.dtype)
        return torch.cat([self.proto, self.action_embedding, reward_t], dim=-1)

# The memory is just a list of Associations and an attention layer to select the best memories, 
# with some hardcoded heuristics for the moment.
# When should it be trained? 
# Unclear. My instinct is to train it at night only
class Memory(nn.Module):
    def __init__(self, d_model, d_action, query_short_memories=5, query_long_memories=5,
                 num_heads=2, short_capacity=8, long_capacity=64):
        super().__init__()
        self.short_term = collections.deque(maxlen=short_capacity)
        self.long_term = []
        self.long_capacity = long_capacity

        step_dim = d_model + d_action + 1
        self.attn_dim = d_model
        assert self.attn_dim % num_heads == 0, (
            f"attn_dim ({self.attn_dim}) must be divisible by num_heads ({num_heads})"
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=self.attn_dim, num_heads=num_heads, kdim=d_model, vdim=step_dim, batch_first=True
        )
        self.q_proj = nn.Linear(d_model, self.attn_dim)

        self.query_short_memories = query_short_memories
        self.query_long_memories = query_long_memories
        self.d_model = d_model
        self.d_action = d_action
 
    def add(self, transition):
        self.short_term.append(transition)
        if len(self.long_term) < self.long_capacity:
            self.long_term.append(transition)
        else:
            min_idx = min(range(len(self.long_term)), key=lambda i: abs(self.long_term[i].priority))
            if abs(transition.priority) > abs(self.long_term[min_idx].priority):
                self.long_term[min_idx] = transition

    def add_short_term(self,transition):
        self.short_term.append(transition)

    def compact_short_memories(self):
        short_items = list(self.short_term)[-self.query_short_memories:]
        pad_size = max(0, self.query_short_memories - len(short_items))
        width = self.d_model + self.d_action + 1
        pieces = [torch.zeros(width) for _ in range(pad_size)] + [t.compact() for t in short_items]
        return torch.stack(pieces, dim=0)   # (query_short_memories, step_dim) — not flattened

    def compact_long_memories(self, query):
        if len(self.long_term) == 0:
            return torch.zeros(self.attn_dim), torch.zeros(0)
        k_protos = torch.stack([t.proto for t in self.long_term])   # (N, d_model)
        v_raw = torch.stack([t.compact() for t in self.long_term])  # (N, step_dim)

        q_embed = self.q_proj(query).view(1, 1, -1)   # (1, 1, attn_dim)
        key = k_protos.unsqueeze(0)                    # (1, N, d_model)
        values = v_raw.unsqueeze(0)                    # (1, N, step_dim)

        output, attn_weights = self.attention(q_embed, key, values)
        return output.view(-1), attn_weights.view(-1)  # (attn_dim,), (N,)

    def long_memory_kv(self):
        """
        Build the attention keys/values ONCE per sleep call. long_term does not
        change while dreaming, so rebuilding this every step (as the current code
        does) is pure waste.
        Returns (k, v) or None if long-term memory is empty.
        """
        if len(self.long_term) == 0:
            return None
        k = torch.stack([t.proto for t in self.long_term])          # (N, d_model)
        v = torch.stack([t.compact() for t in self.long_term])      # (N, step_dim)
        return k, v
    
    def attend_long(self, query, kv):
        """
        Batched version of compact_long_memories.
        query: (B, d_model)  ->  (B, attn_dim)
        """
        B = query.shape[0]
        if kv is None:
            return query.new_zeros(B, self.attn_dim)
        k, v = kv
        q = self.q_proj(query).unsqueeze(1)                          # (B, 1, attn_dim)
        out, _ = self.attention(
            q,
            k.unsqueeze(0).expand(B, -1, -1),                        # (B, N, d_model)
            v.unsqueeze(0).expand(B, -1, -1),                        # (B, N, step_dim)
        )
        return out.squeeze(1)                                        # (B, attn_dim)
    
    def empty_short_window(self, B, ref):
        """Zero-padded short-term window, one per rollout: (B, K, step_dim)."""
        width = self.d_model + self.d_action + 1
        return ref.new_zeros(B, self.query_short_memories, width)
    
    
    def sample_long_term_protos(self, B):
        idx = torch.randint(len(self.long_term), (B,))
        return torch.stack([self.long_term[i].proto for i in idx])

    def raw_long_memories(self):
        return self.long_term

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
        self.norm = nn.LayerNorm(d_model)
 
    def forward(self, obs):
        single = obs.dim() == 3
        if single:
            obs = obs.unsqueeze(0)  # add batch dim
        x = self.conv(obs).flatten(1)
        latent = self.norm(self.fc(x))
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
    def __init__(self, d_model, d_action, query_short_memory, long_term_dim, out_dim=32):
        super().__init__()
        step_dim = d_model + d_action + 1
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

class ActionMapper(nn.Module):
    def __init__(self, n_actions: int, d_action: int, temperature: float = 0.1):
        super().__init__()
        self.n_actions = n_actions
        self.d_action = d_action
        self.temperature = temperature
        
        # Codebook mapping each discrete action index (0..n_actions-1) to a d_action vector
        self.codebook = nn.Embedding(n_actions, d_action)
        nn.init.normal_(self.codebook.weight, std=0.02)

    def compute_scores(self, proto_action: torch.Tensor) -> torch.Tensor:
        # Normalize vectors for cosine similarity stability if desired
        proto_norm = F.normalize(proto_action, p=2, dim=-1)
        codebook_norm = F.normalize(self.codebook.weight, p=2, dim=-1)
        
        # Similarity score: (..., d_action) x (d_action, n_actions) -> (..., n_actions)
        scores = (proto_norm @ codebook_norm.t()) / self.temperature
        return scores

    def distribution(self, proto_action: torch.Tensor) -> torch.distributions.Categorical:
        scores = self.compute_scores(proto_action)
        return torch.distributions.Categorical(logits=scores)

    def sample(self, proto_action: torch.Tensor):
        dist = self.distribution(proto_action)
        action_idx = dist.sample()
        return action_idx, dist

    def best(self, proto_action: torch.Tensor) -> torch.Tensor:
        scores = self.compute_scores(proto_action)
        return torch.argmax(scores, dim=-1)

    def encode(self, action_idx: torch.Tensor) -> torch.Tensor:
        return self.codebook(action_idx)

    def freeze_weights(self):
        for param in self.parameters():
            param.requires_grad = False

    def activate_weights(self):
        for param in self.parameters():
            param.requires_grad = True

class Cortex(DualBranchHead):
    def __init__(self, d_model, d_action, query_short_memory,
                 long_term_dim, out_dim=32):
        super().__init__(d_model, d_action, query_short_memory, long_term_dim, out_dim)
        self.net = nn.Sequential(
            nn.Linear(self.fused_dim, d_model),
            nn.ELU(),
            nn.Linear(d_model, d_action),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, protomemory, short_term_window, long_term_payload):
        fused, single = self.fuse(protomemory, short_term_window, long_term_payload)
        logits = self.net(fused)
        return logits.squeeze(0) if single else logits
 
class ValueCenter(DualBranchHead):
    def __init__(self, d_model, d_action, query_short_memory, long_term_dim, out_dim=32):
        super().__init__(d_model, d_action, query_short_memory, long_term_dim, out_dim)
        self.net = nn.Sequential(
            nn.Linear(self.fused_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )
 
    def forward(self, protomemory, short_term_window, long_term_payload):
        fused, single = self.fuse(protomemory, short_term_window, long_term_payload)
        out = self.net(fused).squeeze(-1)
        return out.squeeze(0) if single else out

class DreamerCenter(DualBranchHead):
    def __init__(self, d_model: int, d_action: int, query_short_memories: int, long_term_dim: int = 32, hidden_dim: int = 132):
        super().__init__(d_model, d_action, query_short_memories, long_term_dim)

        # Fused memory/perception context + continuous action embedding vector (d_action)
        input_dim = self.fused_dim + d_action
        
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        # Predicts next latent state given action embedding and context
        self.next_state_head = nn.Sequential(
            nn.Linear(hidden_dim, d_model),
            nn.LayerNorm(d_model),
        )
        # Predicts expected reward
        self.reward_head = nn.Linear(hidden_dim, 1)
        # Predicts termination probability (logits)
        self.world_over_head = nn.Linear(hidden_dim, 1)

    def forward(self, protomemory: torch.Tensor, short_term_window: torch.Tensor, long_term_payload: torch.Tensor, action_embedding: torch.Tensor):
        fused, single = self.fuse(protomemory, short_term_window, long_term_payload)

        if single and action_embedding.dim() == 1:
            action_embedding = action_embedding.unsqueeze(0)

        out = self.trunk(torch.cat([fused, action_embedding], dim=-1))
        
        next_world_state = self.next_state_head(out)
        expected_next_reward = self.reward_head(out).squeeze(-1)
        continuation_prob = self.world_over_head(out).squeeze(-1)

        return (
            next_world_state.squeeze(0) if single else next_world_state,
            expected_next_reward.squeeze(0) if single else expected_next_reward,
            continuation_prob.squeeze(0) if single else continuation_prob
        )

class Agent(nn.Module):
    def __init__(self, world, d_model=128, d_action = 16, num_heads=2, lr=1e-3, gamma=0.99, entropy_coef=0.05):
        super().__init__()
        self.d_model = d_model
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.is_asleep = False
        
        frame_stack, img_size, _ = world.obs_shape
        self.memory = Memory(d_model=d_model, query_short_memories=5, query_long_memories=5,
                             d_action=d_action, num_heads=num_heads)
        self.perception = Perception(frame_stack, img_size, d_model)
        self.decoder = Projection(self.perception) 
 
        self.n_actions = world.n_actions
        
        self.cortex = Cortex(d_model, d_action, self.memory.query_short_memories, self.memory.attn_dim)
        self.dreamer = DreamerCenter(d_model=d_model, d_action=d_action,
                                    query_short_memories=self.memory.query_short_memories,
                                    long_term_dim=self.memory.attn_dim)
        self.value_center = ValueCenter(d_model, d_action, self.memory.query_short_memories, self.memory.attn_dim)
        self.action_mapper = ActionMapper(n_actions=world.n_actions, d_action=d_action)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

 
    def waking_up(self):
        self.memory.short_term.clear()
        self.cortex.freeze_weights()
        self.value_center.freeze_weights()
        self.dreamer.activate_weights()
        self.action_mapper.activate_weights()
        self.perception.activate_weights()
        self.decoder.activate_weights()
        self.memory.activate_weights()
        self.is_asleep = False

    @torch.no_grad()
    def get_value(self, worldview):
        proto = self.perception(worldview)
        short_window = self.memory.compact_short_memories()
        long_payload, _ = self.memory.compact_long_memories(proto)
        return self.value_center(proto, short_window, long_payload).squeeze()
 
    def going_to_sleep(self):
        self.memory.short_term.clear()
        self.cortex.activate_weights()
        self.value_center.activate_weights()
        self.dreamer.freeze_weights()
        self.action_mapper.freeze_weights()
        self.memory.freeze_weights()
        self.perception.freeze_weights()
        self.decoder.freeze_weights()
        self.is_asleep = True


    def sleep(self,horizon=15, dream_batch=32, lambda_=0.95,
          seed_protos=None, seed_windows=None):

        if seed_protos is None:
            z = self.memory.sample_long_term_protos(dream_batch)
            window = self.memory.empty_short_window(dream_batch, z)
        else:
            idx = torch.randint(seed_protos.shape[0], (dream_batch,))
            z = seed_protos[idx].detach()                            # (B, d_model)
            if seed_windows is None:
                window = self.memory.empty_short_window(dream_batch, z)
            else:
                window = seed_windows[idx].detach()                  # (B, K, step_dim)

        B = z.shape[0]
        kv = self.memory.long_memory_kv()


        values, rewards, entropies, conts, logps = [], [], [], [], []

        for _ in range(horizon):
            long_payload = self.memory.attend_long(z, kv)            # (B, attn_dim)
    
            proto_action = self.cortex(z, window, long_payload)      # (B, d_action)
            action_idx, dist = self.action_mapper.sample(proto_action)   # (B,)
            probs = dist.probs                                       # (B, n_actions)
            onehot = F.one_hot(action_idx, self.n_actions).float()
            st = onehot + probs - probs.detach()                     # straight-through
            action_embedding = st @ self.action_mapper.codebook.weight    # (B, d_action)
    
            v = self.value_center(z, window, long_payload)           # (B,)
            z_next, r, done_logit = self.dreamer(z, window, long_payload, action_embedding)
            c = torch.sigmoid(-done_logit)                           # (B,) P(continue)
    
            logps.append(dist.log_prob(action_idx))
            values.append(v)
            rewards.append(r)
            conts.append(c)
            entropies.append(dist.entropy())
    
            # roll the per-rollout window. Layout must match Association.compact():
            # [proto, action_embedding, reward]
            new_step = torch.cat([z, action_embedding, r.unsqueeze(-1)], dim=-1).detach()
            window = torch.cat([window[:, 1:], new_step.unsqueeze(1)], dim=1)
    
            z = z_next          # no detach: gradient flows through the dynamics chain


        # ---- bootstrap ---------------------------------------------------------
        long_payload = self.memory.attend_long(z, kv)
        R = self.value_center(z, window, long_payload)               # (B,)
    
        returns = []
        for t in reversed(range(horizon)):
            v_next = values[t + 1] if t + 1 < horizon else R
            R = rewards[t] + self.gamma * conts[t] * ((1 - lambda_) * v_next + lambda_ * R)
            returns.insert(0, R)
    
        returns_t = torch.stack(returns)                             # (T, B)
        values_t = torch.stack(values)                               # (T, B)
        logps_t = torch.stack(logps)                                 # (T, B)
        ent_t = torch.stack(entropies)                               # (T, B)
        cont_t = torch.stack(conts)                                  # (T, B)
    
        # ---- termination-aware weighting ---------------------------------------
        # Step t only "exists" with probability prod(c_0..c_{t-1}). Without this,
        # deep steps of rollouts the model believes are already dead get full
        # weight in the loss. Matters much more at horizon 15+ than at 10.
        with torch.no_grad():
            ones = torch.ones_like(cont_t[:1])
            w = torch.cumprod(torch.cat([ones, cont_t[:-1]], dim=0), dim=0)   # (T, B)
            w = w / w.sum().clamp(min=1e-8)
    
        adv = (returns_t - values_t).detach()
        S = (torch.quantile(returns_t.detach(), 0.95) -
            torch.quantile(returns_t.detach(), 0.05)).clamp(min=1.0)
    
        actor_loss = -(w * (logps_t * adv / S)).sum()
        entropy_bonus = (w * ent_t).sum()
        actor_loss = actor_loss - self.entropy_coef * entropy_bonus
    
        value_loss = (w * (values_t - returns_t.detach()).pow(2)).sum()
    
        loss = actor_loss + value_loss
    
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()
    
        return (actor_loss.detach(),
                (-self.entropy_coef * entropy_bonus).detach(),
                value_loss.detach(),
                cont_t.var().item(),
                cont_t.mean().item())
 
    def remember(self, proto, action, reward, priority):
        self.memory.add(Association(proto, action, reward, priority))
 
    def freeze_all(self):
        self.cortex.freeze_weights()
        self.dreamer.freeze_weights()
        self.value_center.freeze_weights()
        self.perception.freeze_weights()
        self.decoder.freeze_weights()

@torch.no_grad()
def advance_single_episode(agent,error_tracker,world):
    # we forward the agent a full day with the "policy" encoded in the cortex 
    agent.waking_up()
    error_tracker.start_episode()
    world.reset()
    ep_reward = 0.0
    done = False
    day_buffer = []

    while not done:
        viewed_world = world.observe()

        protomemory = agent.perception(viewed_world)  
        short_window = agent.memory.compact_short_memories()
        long_payload,_ = agent.memory.compact_long_memories(protomemory)

        predicted_value = agent.value_center(protomemory, short_window, long_payload)
        proto_action = agent.cortex(protomemory, short_window, long_payload)
        action_idx, dist = agent.action_mapper.sample(proto_action)

        _, reward, done, _ = world.act(action_idx.item())
        next_obs = world.observe()

        next_value = 0.0 if done else agent.get_value(next_obs).item()
        td_error = reward + agent.gamma * next_value - predicted_value.item()

        agent.remember(
            proto=protomemory,
            action=proto_action,
            reward=reward,
            priority = abs(td_error)
        )

        day_buffer.append({
            "obs": viewed_world,
            "reward": torch.tensor(reward, dtype=torch.float32),
            "next_obs": next_obs,
            "done": torch.tensor(float(done), dtype=torch.float32),
            "short": short_window,
            "long": long_payload,
            "action_idx": action_idx
        })

        ep_reward += reward

    return day_buffer, ep_reward
 
 
def train(day_batch_size = 32, sleep_batch_size = 32 ,num_episodes=50, sleep_updates=20, lr=1e-3, log_every=20, render=False):
    world = World("CartPole-v1", render=render)
    agent = Agent(world, d_model=128, lr=lr)
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

    for batches in range(num_episodes):
        concatenated_daybuffer = []
        buffer_of_day_buffers = []
        reward_buffer = []
        for episode in range(day_batch_size):
            day_buffer, ep_reward = advance_single_episode(agent=agent, error_tracker=error_tracker, world=world)
            concatenated_daybuffer += day_buffer
            buffer_of_day_buffers.append(day_buffer)
            reward_buffer.append(ep_reward)

        # values to be used
        obs_batch = torch.stack([step["obs"] for step in concatenated_daybuffer])
        next_obs_batch = torch.stack([step["next_obs"] for step in concatenated_daybuffer])
        rewards_batch = torch.stack([step["reward"] for step in concatenated_daybuffer])
        dones_batch = torch.stack([step["done"] for step in concatenated_daybuffer])

        short_batch = torch.stack([step["short"] for step in concatenated_daybuffer])
        long_batch = torch.stack([step["long"] for step in concatenated_daybuffer])
        action_idx_batch = torch.stack([step["action_idx"] for step in concatenated_daybuffer])

        action_embedding_batch = agent.action_mapper.encode(action_idx_batch)

        proto_batch = agent.perception(obs_batch)          # grad flows through perception
        recon_views = agent.decoder(proto_batch)

        with torch.no_grad():
            next_proto_targets = agent.perception(next_obs_batch)

        pred_next_proto, pred_reward, pred_done_logits = agent.dreamer(
            proto_batch, short_batch, long_batch, action_embedding_batch
        )

        recon_loss_per_step = F.mse_loss(recon_views, obs_batch, reduction='none').mean(
            dim=tuple(range(1, obs_batch.dim()))
        ) 
        dynamics_loss_per_step = F.mse_loss(pred_next_proto, next_proto_targets, reduction='none').mean(dim=-1)  # (T,)
        reward_loss_per_step = F.mse_loss(pred_reward.squeeze(-1), rewards_batch, reduction='none')  # (T,)

        num_done = (dones_batch == 1).sum()
        num_not_done = (dones_batch == 0).sum()
        pos_weight = (num_not_done / num_done.clamp(min=1)).clamp(max=20).to(pred_done_logits.device)

        done_loss_per_step = F.binary_cross_entropy_with_logits(
            pred_done_logits.squeeze(-1), dones_batch,pos_weight=pos_weight,reduction='none'
        )  # (T,)

        recon_loss = recon_loss_per_step.mean()
        dynamics_loss = dynamics_loss_per_step.mean()
        reward_loss = reward_loss_per_step.mean()
        done_loss = done_loss_per_step.mean()

        awake_loss = recon_loss + reward_loss + dynamics_loss + done_loss

        agent.optimizer.zero_grad()
        awake_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.0)
        agent.optimizer.step()

        t = 0
        for day_buffer in buffer_of_day_buffers:
            for _ in range(len(day_buffer)):
                temp_episodic_metric_storage["reward_error"] = reward_loss_per_step[t].detach()
                temp_episodic_metric_storage["encoder_decoder_error"] = recon_loss_per_step[t].detach()
                temp_episodic_metric_storage["predict_next_word_state"] = dynamics_loss_per_step[t].detach()
                temp_episodic_metric_storage["done_loss"] = done_loss_per_step[t].detach()
                error_tracker.query_measurments()
                t += 1

        error_tracker.end_episode(ep_reward)
        agent.going_to_sleep()
 
        if len(agent.memory) > 0:
            a_sum = e_sum = v_sum = 0.0
            cp_var, cp_mean = [], []
            seed_protos = proto_batch.detach()
            for _ in range(sleep_updates):              # e.g. 20 -- cheap now
                a, e, v, cv, cm = agent.sleep(
                    horizon=15,
                    dream_batch=sleep_batch_size,
                    seed_protos=seed_protos,
                    seed_windows=short_batch,
                )
                a_sum += a; e_sum += e; v_sum += v
                cp_var.append(cv); cp_mean.append(cm)
            error_tracker.append_sleep(a_sum / sleep_updates,
                                    e_sum / sleep_updates,
                                    v_sum / sleep_updates,
                                    float(np.mean(cp_var)),
                                    float(np.mean(cp_mean)))



        avg = np.mean(error_tracker.rewards[-log_every:])
        print(f"Episode {batches + 1:4d} | Avg Real Reward (last {log_every}): {avg:.1f}")
 
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
                protomemory = agent.perception(obs_t)
                short_window = agent.memory.compact_short_memories()
                long_payload, _ = agent.memory.compact_long_memories(protomemory)

                predicted_value = agent.value_center(protomemory, short_window, long_payload)
                proto_action = agent.cortex(protomemory, short_window, long_payload)
                action_idx, _ = agent.action_mapper.sample(proto_action)

            _, reward, done, _ = world.act(action_idx.item())
            next_obs = world.observe()
            with torch.no_grad():
                next_value = 0.0 if done else agent.get_value(next_obs).item()
                td_error = reward + agent.gamma * next_value - predicted_value.item()

            captured_frames.append(world.env.render())
            agent.remember(proto=protomemory, action=proto_action.detach(), reward=reward, priority=abs(td_error))

            ep_reward += reward
        print(f"[watch] Episode {episode + 1}: reward = {ep_reward:.0f}")

    world.close()
    return captured_frames
 
 
if __name__ == "__main__":
    # The loop is simple:
    # we train it for a number of episodes
    trained_brain, error_tracker = train(num_episodes=50, render=False)
    trained_brain.freeze_all()
    # we then just force the brain to move in the world
    captured_frames = watch(trained_brain, num_episodes=1, render=True)
    ## Because we are cool we generate a pdf file with 
    # a report detailing the statistics during training
    error_tracker.render_pdf(captured_frames)   
