"""Entity-Transformer policy/value network for Slay the Spire 2.

Architecture (one network for every decision in the run):

  * Every game entity (cards in each pile, creatures, powers, relics, potions, orbs,
    map nodes, offered rewards, run-history summary, upcoming bosses) is a token:
        embed(token_type) + embed(vocab_id) + embed(aux1) + embed(aux2) + Linear(numeric)
  * `n_memory` learned "global memory" tokens are appended. With `recurrent=True` they are
    carried across decisions inside a run (Recurrent-Memory-Transformer style, trained with
    truncated BPTT over rollout chunks); otherwise they act as learned scratch slots.
  * A pre-LN Transformer encoder mixes all tokens (full self-attention: the deck can attend
    to the boss, the card reward can attend to relics, etc.).
  * Policy: every legal action is a token built from its own embedding plus pointers to the
    encoded entities it references (card, target). A small MLP scores each action against the
    global summary -> masked softmax. Works for any number/kind of actions.
  * Value: several heads on the global token
        - v_return : expected (shaped) return, used for GAE
        - v_win    : logit of P(win this run)          (strategic signal, shown in the overlay)
        - v_floor  : expected final floor reached / 50  (auxiliary)
        - v_hp     : expected HP fraction at end of this combat (auxiliary, combat only)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import spec


class Block(nn.Module):
    """Pre-LN Transformer block on fused attention (F.scaled_dot_product_attention).

    nn.TransformerEncoder in training mode materializes the full [B, heads, S, S] attention matrix
    for every layer (~5 GB per layer at B=4096, S=200); the fused kernels never do."""

    def __init__(self, d: int, n_heads: int, d_ff: int, n_experts: int = 0):
        super().__init__()
        self.h = n_heads
        self.n_experts = n_experts
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        if n_experts:
            # one feed-forward per character (expert e = blocks.i.ffe*[e]; sliced to ff1/ff2 by export --character)
            self.ffe1_w = nn.Parameter(nn.init.trunc_normal_(torch.empty(n_experts, d_ff, d), std=0.02))
            self.ffe1_b = nn.Parameter(torch.zeros(n_experts, d_ff))
            self.ffe2_w = nn.Parameter(nn.init.trunc_normal_(torch.empty(n_experts, d, d_ff), std=0.02))
            self.ffe2_b = nn.Parameter(torch.zeros(n_experts, d))
        else:
            self.ff1 = nn.Linear(d, d_ff)
            self.ff2 = nn.Linear(d_ff, d)

    def forward(self, x, keep, splits=None):
        """splits (expert blocks): samples per expert; the batch is sorted by expert (Sts2Net.forward)."""
        B, S, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, S, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=keep)
        x = x + self.proj(a.transpose(1, 2).reshape(B, S, D))
        y = self.ln2(x)
        if not self.n_experts:
            return x + self.ff2(F.gelu(self.ff1(y)))
        parts = []
        for e, ye in enumerate(torch.split(y, splits, 0)):
            if ye.shape[0]:
                ye = F.linear(F.gelu(F.linear(ye, self.ffe1_w[e], self.ffe1_b[e])), self.ffe2_w[e], self.ffe2_b[e])
            parts.append(ye)
        return x + torch.cat(parts, 0)


@dataclass
class ModelConfig:
    vocab_size: int = 4096
    d_model: int = 384
    n_layers: int = 8
    n_heads: int = 8
    d_ff: int = 1536
    n_memory: int = 8
    recurrent: bool = False
    dropout: float = 0.0
    # character-routed mixture of experts: the top `moe_layers` blocks have one feed-forward per character,
    # chosen by the Global token's id (the run's character); attention, embeddings and heads are shared
    moe_layers: int = 0
    expert_ids: tuple = ()       # vocab ids of the characters routed to experts 0..n-1 (filled from the vocab)

    def param_estimate(self) -> int:
        d = self.d_model
        per_layer = 4 * d * d + 2 * d * self.d_ff
        emb = self.vocab_size * d + (spec.TokenType.COUNT + spec.ActionType.COUNT + 3 * spec.AUX_VOCAB) * d
        return self.n_layers * per_layer + emb


PRESETS = {
    # ~3M params: smoke tests / CPU
    "tiny": ModelConfig(d_model=128, n_layers=3, n_heads=4, d_ff=512, n_memory=4),
    # ~13M params: rollout-friendly default
    "base": ModelConfig(d_model=256, n_layers=6, n_heads=8, d_ff=1024, n_memory=8),
    # ~22M params: final model
    "large": ModelConfig(d_model=384, n_layers=8, n_heads=8, d_ff=1536, n_memory=8),
    # ~17M params: "base" grown in place (sts2ai.grow: 6 -> 12 layers, d_ff 1024 -> 2048, function-preserving)
    "grown": ModelConfig(d_model=256, n_layers=12, n_heads=8, d_ff=2048, n_memory=8),
    # "base" with per-character feed-forward experts in the top 3 blocks (sts2ai.moe converts a base checkpoint)
    "moe": ModelConfig(d_model=256, n_layers=6, n_heads=8, d_ff=1024, n_memory=8, moe_layers=3),
}

CHARACTER_KEYS = ["CHARACTER.IRONCLAD", "CHARACTER.SILENT", "CHARACTER.DEFECT", "CHARACTER.NECROBINDER",
                  "CHARACTER.REGENT"]  # expert order = RunDriver.CharacterNames


def expert_ids_from_vocab(vocab_ids: list[str]) -> tuple:
    """Vocab ids (index + 2, as in Vocab.cs) of the characters, in expert order."""
    return tuple(vocab_ids.index(k) + 2 for k in CHARACTER_KEYS)


class Sts2Net(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.tok_type = nn.Embedding(spec.TokenType.COUNT, d)
        self.vocab = nn.Embedding(cfg.vocab_size, d, padding_idx=0)
        # secondary id (card enchantment, creature's current move, ...): separate table so it is
        # not confused with the token's primary identity
        self.vocab2 = nn.Embedding(cfg.vocab_size, d // 2, padding_idx=0)
        self.vocab2_proj = nn.Linear(d // 2, d, bias=False)
        self.aux1 = nn.Embedding(spec.AUX_VOCAB, d)
        self.aux2 = nn.Embedding(spec.AUX_VOCAB, d)
        self.tok_num = nn.Sequential(nn.Linear(spec.F_TOK, d), nn.GELU(), nn.Linear(d, d))
        self.memory = nn.Parameter(torch.randn(cfg.n_memory, d) * 0.02)
        self.mem_type = nn.Parameter(torch.zeros(d))
        if cfg.recurrent:
            self.mem_update = nn.GRUCell(d, d)

        self.n_experts = len(cfg.expert_ids) if cfg.moe_layers else 0
        if cfg.moe_layers and not self.n_experts:
            raise ValueError("moe_layers > 0 needs expert_ids (expert_ids_from_vocab)")
        self.blocks = nn.ModuleList(
            Block(d, cfg.n_heads, cfg.d_ff, self.n_experts if i >= cfg.n_layers - cfg.moe_layers else 0)
            for i in range(cfg.n_layers))
        if self.n_experts:
            eo = torch.zeros(cfg.vocab_size, dtype=torch.long)  # unknown characters -> expert 0
            for e, vid in enumerate(cfg.expert_ids):
                eo[vid] = e
            self.register_buffer("expert_of", eo, persistent=False)
        self.final_norm = nn.LayerNorm(d)

        self.act_type = nn.Embedding(spec.ActionType.COUNT, d)
        self.act_aux = nn.Embedding(spec.AUX_VOCAB, d)
        self.act_num = nn.Sequential(nn.Linear(spec.F_ACT, d), nn.GELU(), nn.Linear(d, d))
        self.ref_proj = nn.Linear(2 * d, d)
        self.policy_mlp = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

        self.pool_q = nn.Parameter(torch.randn(d) * 0.02)
        self.value_trunk = nn.Sequential(nn.Linear(2 * d, d), nn.GELU())
        self.v_return = nn.Linear(d, 1)
        self.v_win = nn.Linear(d, 1)
        self.v_floor = nn.Linear(d, 1)
        self.v_hp = nn.Linear(d, 1)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, token_cat, token_num, action_cat, action_ref, action_num, memory_in=None):
        """
        token_cat  [B,T,5] long, token_num [B,T,F_TOK]
        action_cat [B,A,3] long, action_ref [B,A,2] long (-1 = none), action_num [B,A,F_ACT]
        memory_in  [B,M,d] or None
        returns dict(logits[B,A] (masked to -inf), v_return[B], v_win[B], v_floor[B], v_hp[B], memory_out[B,M,d])
        """
        B, T, _ = token_cat.shape
        cfg = self.cfg
        ttype = token_cat[..., 0]
        tok_mask = ttype == spec.TokenType.PAD  # True = padding
        x = (self.tok_type(ttype) + self.vocab(token_cat[..., 1].clamp(0, cfg.vocab_size - 1))
             + self.vocab2_proj(self.vocab2(token_cat[..., 2].clamp(0, cfg.vocab_size - 1)))
             + self.aux1(token_cat[..., 3].clamp(0, spec.AUX_VOCAB - 1))
             + self.aux2(token_cat[..., 4].clamp(0, spec.AUX_VOCAB - 1))
             + self.tok_num(token_num))
        if memory_in is None:
            mem = self.memory.unsqueeze(0).expand(B, -1, -1)
        else:
            mem = memory_in
        mem = mem + self.mem_type
        h = torch.cat([x, mem], dim=1)
        mask = torch.cat([tok_mask, torch.zeros(B, cfg.n_memory, dtype=torch.bool, device=h.device)], dim=1)
        keep = (~mask)[:, None, None, :]  # [B,1,1,S] True = attend to this key
        splits = None
        if self.n_experts:
            # sort the batch by expert so each expert runs on one contiguous slice (one host sync per forward)
            route = self.expert_of[token_cat[:, 0, 1].clamp(0, cfg.vocab_size - 1)]
            order = torch.argsort(route, stable=True)
            splits = torch.bincount(route, minlength=self.n_experts).tolist()
            h, keep = h[order], keep[order]
        for blk in self.blocks:
            h = blk(h, keep, splits)
        if splits is not None:
            h = h[torch.argsort(order)]
        h = self.final_norm(h)
        ent = h[:, :T]
        mem_out = h[:, T:]

        g = ent[:, 0]  # GLOBAL token
        # attention pooling over all live tokens (entities + memory)
        scores = (h @ self.pool_q) / math.sqrt(cfg.d_model)
        scores = scores.masked_fill(mask, float("-inf"))
        pooled = torch.einsum("bt,btd->bd", scores.softmax(-1), h)
        summary = torch.cat([g, pooled], dim=-1)

        # ---- policy over legal actions ----
        atype = action_cat[..., 0]
        act_mask = atype == spec.ActionType.PAD
        a = (self.act_type(atype) + self.vocab(action_cat[..., 1].clamp(0, cfg.vocab_size - 1))
             + self.act_aux(action_cat[..., 2].clamp(0, spec.AUX_VOCAB - 1)) + self.act_num(action_num))
        ref = action_ref.clamp(min=0)
        ref_valid = (action_ref >= 0).unsqueeze(-1).to(ent.dtype)
        r0 = torch.gather(ent, 1, ref[..., 0:1].expand(-1, -1, cfg.d_model)) * ref_valid[:, :, 0]
        r1 = torch.gather(ent, 1, ref[..., 1:2].expand(-1, -1, cfg.d_model)) * ref_valid[:, :, 1]
        a = a + self.ref_proj(torch.cat([r0, r1], dim=-1))
        A = a.shape[1]
        logits = self.policy_mlp(torch.cat([a, summary.unsqueeze(1).expand(-1, A, -1)], dim=-1)).squeeze(-1)
        logits = logits.masked_fill(act_mask, float("-inf"))

        v = self.value_trunk(summary)
        out = {
            "logits": logits,
            "v_return": self.v_return(v).squeeze(-1),
            "v_win": self.v_win(v).squeeze(-1),
            "v_floor": self.v_floor(v).squeeze(-1),
            "v_hp": self.v_hp(v).squeeze(-1),
        }
        if cfg.recurrent:
            flat = self.mem_update(mem_out.reshape(-1, cfg.d_model), mem.reshape(-1, cfg.d_model))
            out["memory_out"] = flat.view(B, cfg.n_memory, cfg.d_model)
        return out


def dense_for_character(model: Sts2Net, character: int) -> Sts2Net:
    """The MoE network restricted to one character's experts, as a plain dense network (same outputs for that
    character's observations); this is what the C# runtime loads."""
    import re
    from dataclasses import replace
    dense = Sts2Net(replace(model.cfg, moe_layers=0, expert_ids=()))
    sd = {}
    for k, v in model.state_dict().items():
        m = re.fullmatch(r"blocks\.(\d+)\.ffe([12])_([wb])", k)
        sd[f"blocks.{m[1]}.ff{m[2]}.{'weight' if m[3] == 'w' else 'bias'}" if m else k] = v[character] if m else v
    dense.load_state_dict(sd)
    return dense.eval()


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    for name, cfg in PRESETS.items():
        m = Sts2Net(cfg)
        print(f"{name}: {count_params(m) / 1e6:.2f}M params")
