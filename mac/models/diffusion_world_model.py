"""Conditional diffusion world model for interactive traffic.

The model learns p(neighbour futures | scene history, ego plan): a distribution
over how the surrounding drivers respond to a *candidate* ego trajectory. This
is what turns prediction into negotiation, because the ego can compare the
response distributions induced by different plans of its own.

A latent intention head is trained on the same encoder so the policy can reason
over compact behavioural hypotheses (yield / contest / undecided) instead of raw
coordinates.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class SceneEncoder(nn.Module):
    """Encodes scene history and the ego's candidate plan into one context vector."""

    def __init__(self, history_len, n_neighbors, plan_len, hidden=128):
        super().__init__()
        self.n_neighbors = n_neighbors
        self.agent_mlp = nn.Sequential(
            nn.Linear(history_len * 5, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.plan_mlp = nn.Sequential(
            nn.Linear(plan_len * 3, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.fuse = nn.Sequential(
            nn.Linear(hidden * 3, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),
        )

    def forward(self, history, ego_plan, independent=False):
        b, h, a, d = history.shape
        flat = history.permute(0, 2, 1, 3).reshape(b, a, h * d)
        tokens = self.agent_mlp(flat)
        ego_token = tokens[:, 0]
        neighbor_tokens = tokens[:, 1:]
        plan = self.plan_mlp(ego_plan.reshape(b, -1))
        if independent:
            k = neighbor_tokens.shape[1]
            ego_exp = ego_token[:, None, :].expand(-1, k, -1)
            plan_exp = plan[:, None, :].expand(-1, k, -1)
            ctx = self.fuse(torch.cat([ego_exp, neighbor_tokens, plan_exp], dim=-1))
            return ctx, neighbor_tokens
        # Permutation-invariant pooling over neighbours (joint model).
        pooled = neighbor_tokens.max(dim=1).values
        return self.fuse(torch.cat([ego_token, pooled, plan], dim=-1)), neighbor_tokens


class DenoiserNet(nn.Module):
    def __init__(self, future_len, n_neighbors, hidden=256, context_dim=128):
        super().__init__()
        self.future_len = future_len
        self.n_neighbors = n_neighbors
        self.out_dim = future_len * n_neighbors * 2
        self.time_dim = 128
        self.net = nn.Sequential(
            nn.Linear(self.out_dim + context_dim + self.time_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, self.out_dim),
        )

    def forward(self, x, t, context, agent_mask=None):
        temb = timestep_embedding(t, self.time_dim)
        return self.net(torch.cat([x, context, temb], dim=-1))


class TokenDenoiserNet(nn.Module):
    """Per-neighbour denoiser with masked joint interaction attention."""

    def __init__(self, future_len, n_neighbors, hidden=256, context_dim=128):
        super().__init__()
        self.future_len = future_len
        self.n_neighbors = n_neighbors
        self.out_dim = future_len * n_neighbors * 2
        self.time_dim = 128
        token_dim = future_len * 2
        self.x_proj = nn.Linear(token_dim, hidden)
        self.context_proj = nn.Linear(context_dim, hidden)
        self.time_proj = nn.Linear(self.time_dim, hidden)
        self.attn = nn.MultiheadAttention(
            hidden, num_heads=4, batch_first=True)
        self.ff = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden))
        self.out = nn.Linear(hidden, token_dim)

    def forward(self, x, t, context, agent_mask=None):
        b = x.shape[0]
        tokens = (x.view(b, self.future_len, self.n_neighbors, 2)
                  .permute(0, 2, 1, 3)
                  .reshape(b, self.n_neighbors, self.future_len * 2))
        temb = self.time_proj(timestep_embedding(t, self.time_dim))[:, None]
        hidden = self.x_proj(tokens) + self.context_proj(context) + temb
        padding = None if agent_mask is None else ~agent_mask.bool()
        attended, _ = self.attn(
            hidden, hidden, hidden, key_padding_mask=padding,
            need_weights=False)
        hidden = hidden + attended
        hidden = hidden + self.ff(hidden)
        if agent_mask is not None:
            hidden = hidden * agent_mask[..., None]
        return (self.out(hidden)
                .view(b, self.n_neighbors, self.future_len, 2)
                .permute(0, 2, 1, 3)
                .reshape(b, -1))


class DiffusionWorldModel(nn.Module):
    def __init__(self, history_len=5, future_len=10, n_neighbors=5, plan_len=10,
                 n_steps=50, hidden=256, context_dim=128, n_types=2,
                 independent=False, token_conditioned=False):
        super().__init__()
        self.history_len = history_len
        self.future_len = future_len
        self.n_neighbors = n_neighbors
        self.n_steps = n_steps
        self.independent = bool(independent)
        self.token_conditioned = bool(token_conditioned)

        self.encoder = SceneEncoder(history_len, n_neighbors, plan_len, hidden=context_dim)
        denoise_k = 1 if self.independent else n_neighbors
        if self.token_conditioned:
            if self.independent:
                raise ValueError("token_conditioned and independent are mutually exclusive")
            self.denoiser = TokenDenoiserNet(
                future_len, n_neighbors, hidden=hidden,
                context_dim=context_dim)
        else:
            self.denoiser = DenoiserNet(
                future_len, denoise_k, hidden=hidden,
                context_dim=context_dim)
        # Predicting the latent driver type from the same context gives the
        # policy an interpretable summary of the future distribution.
        self.intention_head = nn.Sequential(
            nn.Linear(context_dim * 2, 128), nn.ReLU(),
            nn.Linear(128, n_types),
        )

        # A cosine schedule drives the cumulative alpha to ~0 even with the few
        # steps this model uses; a linear schedule would leave the "fully noised"
        # sample still carrying signal, which the sampler could never match.
        steps = torch.arange(n_steps + 1, dtype=torch.float32) / n_steps
        s = 0.008
        f = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
        alphas_cumprod = (f / f[0])[1:].clamp(min=1e-5)
        alphas_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])
        betas = (1 - alphas_cumprod / alphas_prev).clamp(max=0.999)
        alphas = 1.0 - betas
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_acp", alphas_cumprod.sqrt())
        self.register_buffer("sqrt_one_minus_acp", (1.0 - alphas_cumprod).sqrt())

    def encode(self, history, ego_plan):
        return self.encoder(
            history, ego_plan,
            independent=self.independent or self.token_conditioned)

    def intention_logits(self, context, neighbor_tokens):
        b, k, d = neighbor_tokens.shape
        if context.dim() == 2:
            ctx = context[:, None, :].expand(b, k, context.shape[-1])
        else:
            ctx = context
        return self.intention_head(torch.cat([ctx, neighbor_tokens], dim=-1))

    def _labelled_slot_weight(self, types, future, weight):
        """Up-weight neighbour slots that have an intention label.

        On cross/roundabout most of K=5 is an unlabeled queue. Fitting FDE
        uniformly there drowns the committers whose response to ``u`` is the
        quantity the planner actually needs.
        """
        if types is None or weight == 1.0:
            return None
        slot = torch.ones(types.shape, device=types.device, dtype=future.dtype)
        slot = torch.where(types >= 0, slot * weight, slot)
        if self.independent:
            return slot.reshape(-1, 1)
        return slot[:, None, :, None].expand(
            -1, future.shape[1], -1, 2).reshape(types.shape[0], -1)

    def loss(self, history, ego_plan, future, types=None, type_weight=0.2,
             intent_mask=None, trajectory_mask=None, labelled_traj_weight=1.0):
        b = history.shape[0]
        context, neighbor_tokens = self.encode(history, ego_plan)

        agent_mask = history[:, -1, 1:, 4] > 0
        slot_w = self._labelled_slot_weight(types, future, labelled_traj_weight)
        if self.independent:
            k = self.n_neighbors
            target = future[..., :2].permute(0, 2, 1, 3).reshape(b * k, -1)
            mask = future[..., 2:].expand(-1, -1, -1, 2).permute(0, 2, 1, 3).reshape(b * k, -1)
            ctx = context.reshape(b * k, -1)
            t = torch.randint(0, self.n_steps, (b * k,), device=history.device)
        else:
            target = future[..., :2].reshape(b, -1)
            mask = future[..., 2:].expand(-1, -1, -1, 2).reshape(b, -1)
            ctx = context
            t = torch.randint(0, self.n_steps, (b,), device=history.device)
            if trajectory_mask is not None:
                mask = mask * trajectory_mask.float().view(b, 1)

        noise = torch.randn_like(target)
        noisy = self.sqrt_acp[t][:, None] * target + self.sqrt_one_minus_acp[t][:, None] * noise
        denoise_mask = None
        if self.independent:
            if trajectory_mask is not None:
                mask = mask * trajectory_mask.float().repeat_interleave(
                    self.n_neighbors).view(-1, 1)
        elif self.token_conditioned:
            denoise_mask = agent_mask
        if slot_w is not None:
            mask = mask * slot_w
        pred = self.denoiser(noisy, t, ctx, denoise_mask)

        # Vehicles that are absent for part of the horizon are masked out.
        denom = mask.sum().clamp(min=1.0)
        diffusion_loss = (((pred - noise) ** 2) * mask).sum() / denom

        total = diffusion_loss
        type_loss = torch.zeros((), device=history.device)
        if types is not None:
            logits = self.intention_logits(context, neighbor_tokens)
            valid = types >= 0
            if intent_mask is not None:
                # Restricting supervision to episodes whose plan was chosen
                # open-loop makes u independent of h, so the head fits
                # p(theta | h, do(u)). Fitting it on reactive episodes as well
                # mixes in the observational conditional, where the plan is a
                # function of the state and its effect is attributed to the
                # state instead.
                valid = valid & intent_mask.view(-1, 1)
            if valid.any():
                type_loss = F.cross_entropy(logits[valid], types[valid])
                total = total + type_weight * type_loss
        return total, {"diffusion": diffusion_loss.item(),
                       "intention": float(type_loss.detach())}

    def counterfactual_delta_loss(self, history_a, plan_a, future_a,
                                  history_b, plan_b, future_b, types_a=None,
                                  types_b=None, labelled_traj_weight=1.0):
        """Supervise response differences for matched do(u) branches."""
        if self.independent:
            raise ValueError("paired delta loss requires a joint denoiser")
        if history_a.shape != history_b.shape:
            raise ValueError("paired batches must have matching shapes")
        b = history_a.shape[0]
        ctx_a, _ = self.encode(history_a, plan_a)
        ctx_b, _ = self.encode(history_b, plan_b)
        target_a = future_a[..., :2].reshape(b, -1)
        target_b = future_b[..., :2].reshape(b, -1)
        mask = (future_a[..., 2:] * future_b[..., 2:]).expand(
            -1, -1, -1, 2).reshape(b, -1)
        types = types_a if types_a is not None else types_b
        slot_w = self._labelled_slot_weight(types, future_a, labelled_traj_weight)
        if slot_w is not None:
            mask = mask * slot_w
        # Avoid the final near-pure-noise step where x0 reconstruction divides
        # by an extremely small cumulative alpha and destabilises this auxiliary
        # contrast objective.
        t = torch.randint(
            0, max(self.n_steps - 1, 1), (b,), device=history_a.device)
        noise = torch.randn_like(target_a)
        sqrt_acp = self.sqrt_acp[t][:, None]
        sqrt_om = self.sqrt_one_minus_acp[t][:, None]
        noisy_a = sqrt_acp * target_a + sqrt_om * noise
        noisy_b = sqrt_acp * target_b + sqrt_om * noise
        agent_mask = history_a[:, -1, 1:, 4] > 0 if self.token_conditioned else None
        eps_a = self.denoiser(noisy_a, t, ctx_a, agent_mask)
        eps_b = self.denoiser(noisy_b, t, ctx_b, agent_mask)
        x0_a = ((noisy_a - sqrt_om * eps_a) / sqrt_acp).clamp(-4.0, 4.0)
        x0_b = ((noisy_b - sqrt_om * eps_b) / sqrt_acp).clamp(-4.0, 4.0)
        error = ((x0_b - x0_a) - (target_b - target_a)) ** 2
        return (error * mask).sum() / mask.sum().clamp(min=1.0)

    @torch.no_grad()
    def sample(self, history, ego_plan, n_samples=8, steps=None, eta=1.0, x0_clip=4.0,
               common_noise=False, generator=None):
        """Draw ``n_samples`` plausible neighbour futures per scene.

        ``steps`` selects a strided DDIM schedule, which is what makes the model
        cheap enough to query inside the reinforcement learning loop.

        ``common_noise`` reuses one latent draw across the batch dimension. When
        the batch enumerates counterfactual plans for a *single* scene this is
        the coupling that defines a counterfactual: the same exogenous noise
        under a different action. Differences between batch rows then isolate
        the effect of the plan instead of also carrying two independent
        Monte-Carlo errors. Do not enable it when the batch holds distinct
        scenes, since their futures would share a latent.
        """
        b = history.shape[0]
        context, _ = self.encode(history, ego_plan)
        agent_mask = history[:, -1, 1:, 4] > 0
        rows = self.n_neighbors if self.independent else 1
        if self.independent:
            context = context.reshape(b * rows, -1)
        context = context.repeat_interleave(n_samples, dim=0)

        def draw(like=None):
            shape = (rows * n_samples, self.denoiser.out_dim)
            if not common_noise:
                sample_shape = ((b * rows * n_samples, self.denoiser.out_dim)
                                if like is None else like.shape)
                return torch.randn(
                    sample_shape, device=history.device,
                    dtype=history.dtype, generator=generator)
            # Layout is (batch, rows, samples) with samples varying fastest, so
            # tiling a single (rows * samples) block aligns it across the batch.
            return torch.randn(
                shape, device=history.device, dtype=history.dtype,
                generator=generator).repeat(b, 1)

        x = draw()
        steps = self.n_steps if steps is None else min(steps, self.n_steps)
        schedule = torch.linspace(self.n_steps - 1, 0, steps).long().tolist()

        for i, step in enumerate(schedule):
            t = torch.full((x.shape[0],), step, device=x.device, dtype=torch.long)
            denoise_mask = agent_mask.repeat_interleave(
                n_samples, dim=0) if self.token_conditioned else None
            eps = self.denoiser(x, t, context, denoise_mask)
            acp = self.alphas_cumprod[step]
            # At high noise levels acp is tiny, so recovering x0 divides by a
            # very small number; without clipping the sampler diverges.
            x0 = ((x - (1 - acp).sqrt() * eps) / acp.sqrt()).clamp(-x0_clip, x0_clip)

            if i == len(schedule) - 1:
                x = x0
                continue

            prev = schedule[i + 1]
            acp_prev = self.alphas_cumprod[prev]
            sigma = eta * ((1 - acp_prev) / (1 - acp)).sqrt() * (1 - acp / acp_prev).sqrt()
            direction = (1 - acp_prev - sigma**2).clamp(min=0).sqrt() * eps
            x = acp_prev.sqrt() * x0 + direction + sigma * draw(x)

        if self.independent:
            k = self.n_neighbors
            return (x.view(b, k, n_samples, self.future_len, 2)
                    .permute(0, 2, 3, 1, 4).contiguous())
        return x.view(b, n_samples, self.future_len, self.n_neighbors, 2)

    @torch.no_grad()
    def predict_intentions(self, history, ego_plan, guidance=1.0):
        """``guidance`` extrapolates along the plan-conditional direction.

        Cross-entropy on a finite sample shrinks the plan term toward the
        history-only solution, so the head reports the right sign but too small
        a change in intention. Writing the logits as the history-only term plus
        a plan contribution and scaling that contribution by ``w`` undoes the
        shrinkage; ``w = 1`` is the plain conditional. The weight is a property
        of the fitted model and is calibrated on held-out open-loop episodes.
        """
        context, neighbor_tokens = self.encode(history, ego_plan)
        logits = self.intention_logits(context, neighbor_tokens)
        if guidance != 1.0:
            base_context, base_tokens = self.encode(history, torch.zeros_like(ego_plan))
            base = self.intention_logits(base_context, base_tokens)
            logits = base + guidance * (logits - base)
        return F.softmax(logits, dim=-1)
