import io
import json
import re
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def bytes_to_unicode() -> Dict[int, str]:
    
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word: Tuple[str, ...]) -> set:
    pairs = set()
    prev_char = word[0]
    for ch in word[1:]:
        pairs.add((prev_char, ch))
        prev_char = ch
    return pairs


class BPECLIPTokenizer:
   
    def __init__(
        self,
        vocab_json_path: str,
        merges_txt_path: str,
        context_length: int = 77,
        pad_token: Optional[str] = None,
        bos_token: Optional[str] = None,
        eos_token: Optional[str] = "</s>",
    ):
        self.context_length = context_length

        # load vocab
        with open(vocab_json_path, "r", encoding="utf-8") as f:
            self.encoder: Dict[str, int] = json.load(f)
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.vocab_size = len(self.encoder)

        # load merges and build ranks
        merges: List[Tuple[str, str]] = []
        with open(merges_txt_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue  # skip header
                pair = line.strip().split()
                if len(pair) == 2:
                    merges.append((pair[0], pair[1]))
        self.bpe_ranks: Dict[Tuple[str, str], int] = {pair: i for i, pair in enumerate(merges)}

        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        # special token ids
        self.pad_id = self.encoder.get(pad_token) if pad_token is not None else 0
        self.bos_id = self.encoder.get(bos_token) if bos_token is not None else None
        self.eos_id = self.encoder.get(eos_token) if eos_token is not None else None

        self.cache: Dict[str, List[str]] = {}
        self._pat = re.compile(
            r"'s|'t|'re|'ve|'m|'ll|'d|[\w]+|[^\s\w]", re.IGNORECASE
        )

    def bpe(self, token: str) -> List[str]:
        if token in self.cache:
            return self.cache[token]

        word = list(token)
        word = tuple(word)
        pairs = get_pairs(word)

        if not pairs:
            self.cache[token] = [token]
            return [token]

        while True:
            bigram = min(pairs, key=lambda p: self.bpe_ranks.get(p, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word: List[str] = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                new_word.extend(word[i:j])
                i = j

                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)

        tokens = list(word)
        self.cache[token] = tokens
        return tokens

    def encode(self, text: str) -> List[int]:
        bpe_tokens: List[int] = []
        for token in re.findall(self._pat, text.strip()):
            token_bytes = token.encode("utf-8")
            token_translated = "".join(self.byte_encoder[b] for b in token_bytes)
            bpe_toks = self.bpe(token_translated)
            for bpe_tok in bpe_toks:
                if bpe_tok in self.encoder:
                    bpe_tokens.append(self.encoder[bpe_tok])
                else:
                    continue
        return bpe_tokens

    def __call__(self, texts: List[str]) -> torch.Tensor:
        B = len(texts)
        ids = torch.full((B, self.context_length), self.pad_id, dtype=torch.long)
        for i, s in enumerate(texts):
            pieces = self.encode(s)
            if self.bos_id is not None:
                pieces = [self.bos_id] + pieces
            if self.eos_id is not None:
                pieces = pieces + [self.eos_id]
            pieces = pieces[: self.context_length]
            ids[i, : len(pieces)] = torch.tensor(pieces, dtype=torch.long)
        return ids


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class MultiheadSelfAttention(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        self.width = width
        self.heads = heads
        self.head_dim = width // heads
        assert self.head_dim * heads == width, "width must be divisible by heads"

        self.in_proj_weight = nn.Parameter(torch.empty(3 * width, width))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * width))
        self.out_proj = nn.Linear(width, width)
        self.scale = self.head_dim ** -0.5

        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.zeros_(self.in_proj_bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, C = x.shape
        qkv = F.linear(x, self.in_proj_weight, self.in_proj_bias)  # (B,L,3C)
        q, k, v = qkv.chunk(3, dim=-1)

        def reshape_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, L, self.heads, self.head_dim).transpose(1, 2)

        q = reshape_heads(q)
        k = reshape_heads(k)
        v = reshape_heads(v)

        attn_scores = (q @ k.transpose(-2, -1)) * self.scale  # (B,H,L,L)
        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask  # additive mask
        attn = attn_scores.softmax(dim=-1)

        out = attn @ v  # (B,H,L,head_dim)
        out = out.transpose(1, 2).contiguous().view(B, L, C)
        out = self.out_proj(out)
        return out


class ResidualAttentionBlock(nn.Module):
    def __init__(self, width: int, heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.ln_1 = nn.LayerNorm(width)
        self.attn = MultiheadSelfAttention(width, heads)
        self.ln_2 = nn.LayerNorm(width)
        self.c_fc = nn.Linear(width, int(width * mlp_ratio))
        self.act = QuickGELU()
        self.c_proj = nn.Linear(int(width * mlp_ratio), width)

        nn.init.xavier_uniform_(self.c_fc.weight)
        nn.init.zeros_(self.c_fc.bias)
        nn.init.xavier_uniform_(self.c_proj.weight)
        nn.init.zeros_(self.c_proj.bias)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), attn_mask)
        x = x + self.c_proj(self.act(self.c_fc(self.ln_2(x))))
        return x


class CLIPTextEncoder(nn.Module):
    
    def __init__(
        self,
        vocab_size: int = 49408,
        context_length: int = 77,
        width: int = 512,
        layers: int = 12,
        heads: int = 8,
        output_dim: Optional[int] = None, 
        eos_id: Optional[int] = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.width = width
        self.layers = layers
        self.heads = heads
        self.eos_id = eos_id

        self.token_embedding = nn.Embedding(vocab_size, width)
        self.positional_embedding = nn.Parameter(torch.empty(context_length, width))
        nn.init.normal_(self.positional_embedding, std=0.01)

        self.transformer = nn.ModuleList(
            [ResidualAttentionBlock(width, heads) for _ in range(layers)]
        )
        self.ln_final = nn.LayerNorm(width)

        self.output_dim = output_dim or width
        self.text_projection = nn.Parameter(torch.empty(width, self.output_dim))
        nn.init.normal_(self.text_projection, std=width ** -0.5)

    @staticmethod
    def build_causal_mask(L: int, device: torch.device) -> torch.Tensor:
        mask = torch.empty(L, L, device=device)
        mask.fill_(-float("inf"))
        mask = torch.triu(mask, diagonal=1)
        return mask  # (L, L)

    def select_global(self, x: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        if self.eos_id is None:
            return x[:, -1]

        eos = (token_ids == self.eos_id)
        idx = eos.int().argmax(dim=1)  
        has_eos = eos.any(dim=1)
        idx = torch.where(has_eos, idx, torch.full_like(idx, token_ids.shape[1] - 1))
        batch = torch.arange(token_ids.shape[0], device=token_ids.device)

        return x[batch, idx]

    def forward(self, token_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L = token_ids.shape
        x = self.token_embedding(token_ids) + self.positional_embedding[:L]

        attn_mask = self.build_causal_mask(L, x.device)
        attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # (1,1,L,L)

        for blk in self.transformer:
            x = blk(x, attn_mask)

        x = self.ln_final(x)
        global_tok = self.select_global(x, token_ids)
        global_emb = global_tok @ self.text_projection
        return x, global_emb

    # weight loading 
    def load_openai_state(self, sd: Dict[str, torch.Tensor]) -> None:
        """Load weights from OpenAI CLIP text encoder state dict."""

        own = self.state_dict()
        remap: Dict[str, str] = {}

        # mappings
        direct = {
            "token_embedding.weight": "token_embedding.weight",
            "positional_embedding": "positional_embedding",
            "ln_final.weight": "ln_final.weight",
            "ln_final.bias": "ln_final.bias",
            "text_projection": "text_projection",
        }
        for k_src, k_dst in direct.items():
            if k_src in sd and own[k_dst].shape == sd[k_src].shape:
                own[k_dst] = sd[k_src]

        # blocks
        for i in range(self.layers):
            prefix_src = f"transformer.resblocks.{i}."
            prefix_dst = f"transformer.{i}."
            mapping = {
                f"{prefix_src}ln_1.weight": f"{prefix_dst}ln_1.weight",
                f"{prefix_src}ln_1.bias": f"{prefix_dst}ln_1.bias",
                f"{prefix_src}attn.in_proj_weight": f"{prefix_dst}attn.in_proj_weight",
                f"{prefix_src}attn.in_proj_bias": f"{prefix_dst}attn.in_proj_bias",
                f"{prefix_src}attn.out_proj.weight": f"{prefix_dst}attn.out_proj.weight",
                f"{prefix_src}attn.out_proj.bias": f"{prefix_dst}attn.out_proj.bias",
                f"{prefix_src}ln_2.weight": f"{prefix_dst}ln_2.weight",
                f"{prefix_src}ln_2.bias": f"{prefix_dst}ln_2.bias",
                f"{prefix_src}mlp.c_fc.weight": f"{prefix_dst}c_fc.weight",
                f"{prefix_src}mlp.c_fc.bias": f"{prefix_dst}c_fc.bias",
                f"{prefix_src}mlp.c_proj.weight": f"{prefix_dst}c_proj.weight",
                f"{prefix_src}mlp.c_proj.bias": f"{prefix_dst}c_proj.bias",
            }
            for k_src, k_dst in mapping.items():
                if k_src in sd and own[k_dst].shape == sd[k_src].shape:
                    own[k_dst] = sd[k_src]

        self.load_state_dict(own)

    def load_state_dict_path(self, path: str) -> None:
        """
        Load weights from a local file.
        """
        state = torch.load(path, map_location="cpu")

        if isinstance(state, dict) and "state_dict" in state and all(
            k.startswith("transformer") or k in {"token_embedding.weight", "positional_embedding", "ln_final.weight", "ln_final.bias", "text_projection"}
            for k in state["state_dict"].keys()
        ):
            state = state["state_dict"]
        if not isinstance(state, dict):
            raise ValueError("Loaded object is not a state_dict mapping")
        self.load_openai_state(state)


def build_text_stack(
    use_bpe: bool,
    context_length: int = 77,
    vocab_json_path: Optional[str] = None,
    merges_txt_path: Optional[str] = None,
    pad_token: Optional[str] = None,
    bos_token: Optional[str] = None,
    eos_token: Optional[str] = "</s>",
    model_width: int = 512,
    layers: int = 12,
    heads: int = 8,
) -> Tuple[nn.Module, object]:
   
    if use_bpe:
        assert vocab_json_path and merges_txt_path, "Provide local vocab.json and merges.txt"
        tok = BPECLIPTokenizer(
            vocab_json_path=vocab_json_path,
            merges_txt_path=merges_txt_path,
            context_length=context_length,
            pad_token=pad_token,
            bos_token=bos_token,
            eos_token=eos_token,
        )
        enc = CLIPTextEncoder(
            vocab_size=tok.vocab_size,
            context_length=context_length,
            width=model_width,
            layers=layers,
            heads=heads,
            eos_id=tok.eos_id,
        )
        return enc, tok
    else:
        raise ValueError("Only BPE tokenizer is supported.")

