import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import json

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@dataclass
class ModelConfig:
    vocab_size:   int   = 256   # overwritten after building vocabulary
    embed_dim:    int   = 128
    num_heads:    int   = 4
    num_layers:   int   = 4
    num_experts:  int   = 3
    k:            int   = 1     # top-k experts per token
    max_seq_len:  int   = 256
    dropout:      float = 0.1

model_cfg = ModelConfig()

class CharTokenizer:
    """Character-level tokenizer built from a training corpus."""

    def __init__(self, corpus: str):
        chars = sorted(set(corpus))
        self.stoi: Dict[str, int] = {ch: i for i, ch in enumerate(chars)}
        self.itos: Dict[int, str] = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(chars)

    def encode(self, text: str) -> List[int]:
        return [self.stoi[ch] for ch in text if ch in self.stoi]

    def decode(self, ids: List[int]) -> str:
        return ''.join(self.itos.get(i, '') for i in ids)

class Expert(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.w1 = nn.Linear(input_dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.w1(x)
        x = F.relu(x)
        x = self.w2(x)
        return x
    
class Gate(nn.Module):
    """
    Gate to select the Top-K experts with highest logits for each token.

    Args:
      input_dim:   Dimension of the input
      num_experts: Number of experts
      k:           Number of experts to select

    Returns:
      Tuple of (dense weight matrix, indices of chosen experts)
    """
    def __init__(self, input_dim: int, num_experts: int = 3, k: int = 1):
        super().__init__()
        self.k = k
        self.linear = nn.Linear(input_dim, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # compute logits
        logits = self.linear(x)

        # select Top-k experts
        topK_logits, topK_indices = torch.topk(logits, self.k, dim=-1)

        # apply softmax to Top-K logits
        topK_probs = F.softmax(topK_logits, dim=-1)

        # sparse weight matrix for combining outputs
        full_weights = torch.zeros_like(logits)
        full_weights.scatter_(1, topK_indices, topK_probs)

        return full_weights, topK_indices
    
class MoE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_experts: int, k: int = 1):
        super().__init__()
        self.num_experts = num_experts
        self.k           = k
        self.input_dim   = input_dim
        self.hidden_dim  = hidden_dim
        self.output_dim  = output_dim

        self.gate = Gate(self.input_dim, self.num_experts, self.k)

        # Each expert: input_dim → hidden_dim*4 → output_dim  (standard FFN ratio)
        self.Experts = nn.ModuleList([
            Expert(self.input_dim, self.hidden_dim * 4, self.output_dim)
            for _ in range(self.num_experts)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x shape: [batch_size, seq_len, input_dim]
        original_shape = x.shape
        x = x.view(-1, self.input_dim)  # flatten to [N, input_dim], N = batch * seq_len

        # Get gating weights and expert indices
        # gating_weights: [N, num_experts]  (sparse — only top-k slots are nonzero)
        # top_k_indices:  [N, k]
        gating_weights, top_k_indices = self.gate(x)

        # Initialize final output tensor
        final_output = torch.zeros(x.shape[0], self.output_dim,
                                   device=x.device, dtype=x.dtype)

        # flat_top_k_indices: [N * k]
        flat_top_k_indices = top_k_indices.view(-1)

        # flat_x: [N * k, input_dim] — repeat each token once per selected expert
        flat_x = x.repeat_interleave(self.k, dim=0)

        # Dispatch tokens to experts and compute outputs
        expert_outputs = []
        for i in range(self.num_experts):
            # idx: positions in flat_x that were assigned to expert i
            idx = torch.where(flat_top_k_indices == i)[0]

            if idx.numel() > 0:
                expert_input  = flat_x[idx]
                expert_output = self.Experts[i](expert_input)
                expert_outputs.append((idx, expert_output))

        # Combine expert outputs using gating weights
        for idx, output in expert_outputs:
            original_indices = idx // self.k            # token index in [0, N)
            selected_expert  = flat_top_k_indices[idx]  # which expert (0 … num_experts-1)

            weights         = gating_weights[original_indices, selected_expert].unsqueeze(1)
            weighted_output = output * weights

            # Scatter-add weighted outputs back to the correct token positions
            final_output.index_add_(0, original_indices, weighted_output)

        # Reshape back to original shape [batch_size, seq_len, output_dim]
        final_output = final_output.view(original_shape[0], original_shape[1], self.output_dim)

        return final_output, gating_weights  # gating_weights returned for aux loss
    
class TransformerBlockWithMoE(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, num_experts: int,
                 k: int = 1, dropout: float = 0.1):
        super().__init__()
        self.attention  = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1      = nn.LayerNorm(embed_dim)
        self.norm2      = nn.LayerNorm(embed_dim)
        self.moe_layer  = MoE(embed_dim, embed_dim, embed_dim, num_experts, k)
        self.dropout    = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Multi-Head Attention part
        attn_output, _ = self.attention(x, x, x, attn_mask=mask)
        x = x + self.dropout(attn_output)   # residual connection
        x = self.norm1(x)

        # MoE Layer part
        moe_output, gate_weights = self.moe_layer(x)
        x = x + self.dropout(moe_output)    # residual connection
        x = self.norm2(x)

        # Return x and gate_weights for auxiliary loss calculation
        return x, gate_weights

class MoELanguageModel(nn.Module):
    """
    Decoder-only transformer language model whose FFN sub-layers are replaced
    by Mixture-of-Experts (MoE) blocks.

    Args:
        config: ModelConfig instance
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.token_embedding    = nn.Embedding(config.vocab_size, config.embed_dim)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.embed_dim)

        self.blocks = nn.ModuleList([
            TransformerBlockWithMoE(
                config.embed_dim,
                config.num_heads,
                config.num_experts,
                config.k,
                config.dropout
            )
            for _ in range(config.num_layers)
        ])

        self.norm   = nn.LayerNorm(config.embed_dim)
        self.output = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

        # Weight tying: share the token embedding and output projection matrices.
        # Halves parameter count and usually improves perplexity.
        self.output.weight = self.token_embedding.weight

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular additive mask to prevent attending to future tokens."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        # x: [batch_size, seq_len]
        B, T = x.shape
        assert T <= self.config.max_seq_len, \
            f"Sequence length {T} exceeds max_seq_len {self.config.max_seq_len}"

        positions = torch.arange(T, device=x.device).unsqueeze(0)  # [1, T]
        h = self.token_embedding(x) + self.position_embedding(positions)

        causal_mask      = self._causal_mask(T, x.device)
        all_gate_weights: List[torch.Tensor] = []

        for block in self.blocks:
            h, gate_weights = block(h, mask=causal_mask)
            all_gate_weights.append(gate_weights)

        h      = self.norm(h)
        logits = self.output(h)   # [B, T, vocab_size]

        return logits, all_gate_weights

    @torch.no_grad()
    def generate(self, prompt_ids: List[int], max_new_tokens: int = 200,
                 temperature: float = 1.0, top_k: int = 40) -> List[int]:
        """Autoregressive text generation with top-k sampling."""
        self.eval()
        device = next(self.parameters()).device
        ids    = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)

        for _ in range(max_new_tokens):
            ids_cond  = ids[:, -self.config.max_seq_len:]
            logits, _ = self(ids_cond)
            logits    = logits[:, -1, :] / temperature   # [1, vocab_size]

            if top_k > 0:
                topk_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < topk_vals[:, [-1]]] = float('-inf')

            probs   = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            ids     = torch.cat([ids, next_id], dim=1)

        return ids[0].tolist()

def call_llm(prompt: str, tokenizer: CharTokenizer, model: MoELanguageModel) -> str:
    prompt_ids = tokenizer.encode(prompt)
    generated  = model.generate(
        prompt_ids, max_new_tokens=200, temperature=0.8, top_k=40
    )
    return tokenizer.decode(generated)

def load_model(model_path: str, tokenizer_path: str) -> Tuple[MoELanguageModel, CharTokenizer]:
    tokenizer = load_tokenizer(tokenizer_path)
    model_cfg = ModelConfig(vocab_size = tokenizer.vocab_size)
    model = MoELanguageModel(model_cfg).to(DEVICE)

    # quantize the model
    model = torch.quantization.quantize_dynamic(
        model,
        qconfig_spec = {nn.Linear},
        dtype = torch.qint8
    )

    # load the state
    model.load_state_dict(torch.load(model_path, map_location = 'cpu'))
    model.eval()

    return model, tokenizer

def load_tokenizer(path: str) -> CharTokenizer:
    with open(path) as f:
        stoi = json.load(f)

    tokenizer = CharTokenizer.__new__(CharTokenizer)
    tokenizer.stoi = stoi
    tokenizer.itos = {i: ch for ch, i in stoi.items()}
    tokenizer.vocab_size = len(stoi)
    return tokenizer

model, tokenizer = load_model("moe_quantized.pt", "tokenizer.json")