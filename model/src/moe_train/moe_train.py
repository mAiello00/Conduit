import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple
from datasets import load_dataset
import math

from model import ModelConfig, CharTokenizer, MoELanguageModel, DEVICE

@dataclass
class TrainConfig:
    batch_size:      int   = 64
    learning_rate:   float = 3e-4
    num_epochs:      int   = 5
    grad_clip:       float = 1.0
    eval_interval:   int   = 500   # steps between evaluations
    aux_loss_alpha:  float = 0.01  # weight for load-balancing loss
    checkpoint_path: str   = "moe_checkpoint.pt"
    quantized_path:  str   = "moe_quantized.pt"

train_cfg = TrainConfig()
model_cfg = ModelConfig()

class TextDataset(Dataset):
    """Sliding-window token dataset for autoregressive next-token prediction."""

    def __init__(self, token_ids: List[int], seq_len: int, stride: int = None):
        self.data    = torch.tensor(token_ids, dtype=torch.long)
        self.seq_len = seq_len
        self.stride = stride or seq_len

    def __len__(self) -> int:
        return (len(self.data) - self.seq_len) // self.stride

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.stride
        x = self.data[start : start + self.seq_len]
        y = self.data[start + 1 : start + self.seq_len + 1]
        return x, y


def load_wikitext2(seq_len: int, stride: int = None) -> Tuple[TextDataset, TextDataset, CharTokenizer]:
    raw = load_dataset("wikitext", "wikitext-2-raw-v1")

    train_text = "\n".join(raw["train"]["text"])
    val_text   = "\n".join(raw["validation"]["text"])

    tokenizer = CharTokenizer(train_text)   # build vocab from training data only

    train_ids = tokenizer.encode(train_text)
    val_ids   = tokenizer.encode(val_text)

    return TextDataset(train_ids, seq_len, stride), TextDataset(val_ids, seq_len, stride), tokenizer


print("Loading WikiText-2 …")
train_dataset, val_dataset, tokenizer = load_wikitext2(model_cfg.max_seq_len, stride=model_cfg.max_seq_len)
model_cfg.vocab_size = tokenizer.vocab_size

print(
    f"Vocab size:   {model_cfg.vocab_size}\n"
    f"Train tokens: {len(train_dataset.data):,}\n"
    f"Val tokens:   {len(val_dataset.data):,}"
)

# num_workers=0 avoids multiprocessing issues in Colab / notebooks
train_loader = DataLoader(
    train_dataset, batch_size=train_cfg.batch_size, shuffle=True,  num_workers=0
)
val_loader = DataLoader(
    val_dataset,   batch_size=train_cfg.batch_size, shuffle=False, num_workers=0
)

def calculate_load_balancing_loss(gate_weights: torch.Tensor, num_experts: int) -> torch.Tensor:
    """Calculates the Switch Transformer load-balancing auxiliary loss.

    Encourages uniform token distribution across experts, preventing collapse
    where all tokens route to a single expert.

    Args:
        gate_weights: Tensor of shape [N, num_experts] (sparse, post top-k softmax)
        num_experts:  Total number of experts.

    Returns:
        Scalar loss tensor.
    """
    num_tokens = gate_weights.shape[0]

    # Fraction of tokens routed to each expert  (f_i)
    tokens_per_expert = torch.sum(gate_weights, dim=0)      # [num_experts]
    f_i = tokens_per_expert / num_tokens

    # Average routing probability per expert  (P_i)
    P_i = torch.mean(gate_weights, dim=0)                   # [num_experts]

    # Switch Transformer: alpha * num_experts * sum(f_i * P_i)
    # alpha is applied by the caller
    loss = num_experts * torch.sum(f_i * P_i)
    return loss

def evaluate(model: MoELanguageModel, dataloader: DataLoader,
             criterion: nn.Module,
             train_cfg: TrainConfig, model_cfg: ModelConfig) -> float:
    """Returns the mean total loss (main + aux) over the validation set."""
    model.eval()
    total_loss  = 0.0
    num_batches = 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            logits, all_gate_weights = model(x)

            loss_main = criterion(logits.view(-1, model_cfg.vocab_size), y.view(-1))
            loss_aux  = sum(
                calculate_load_balancing_loss(gw, model_cfg.num_experts)
                for gw in all_gate_weights
            )
            total_loss  += (loss_main + train_cfg.aux_loss_alpha * loss_aux).item()
            num_batches += 1

    model.train()
    return total_loss / max(num_batches, 1)


def train(model: MoELanguageModel, train_loader: DataLoader, val_loader: DataLoader,
          optimizer: torch.optim.Optimizer, criterion: nn.Module,
          train_cfg: TrainConfig, model_cfg: ModelConfig):
    """Full training loop with evaluation, gradient clipping, and checkpointing."""
    model.train()
    global_step   = 0
    best_val_loss = float('inf')

    # Cosine learning-rate decay over the full training run
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=train_cfg.num_epochs * len(train_loader)
    )

    for epoch in range(train_cfg.num_epochs):
        running_loss = 0.0

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(DEVICE), y.to(DEVICE)

            logits, all_gate_weights = model(x)

            # Main language-modelling loss
            loss_main = criterion(logits.view(-1, model_cfg.vocab_size), y.view(-1))

            # Auxiliary load-balancing loss summed across all transformer blocks
            loss_aux = sum(
                calculate_load_balancing_loss(gw, model_cfg.num_experts)
                for gw in all_gate_weights
            )

            loss = loss_main + train_cfg.aux_loss_alpha * loss_aux

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            optimizer.step()
            scheduler.step()

            running_loss += loss_main.item()
            global_step  += 1

            if global_step % train_cfg.eval_interval == 0:
                val_loss       = evaluate(model, val_loader, criterion, train_cfg, model_cfg)
                train_loss_avg = running_loss / (batch_idx + 1)
                print(
                    f"Epoch {epoch+1} | Step {global_step:>6} | "
                    f"Train loss: {train_loss_avg:.4f} | "
                    f"Val loss: {val_loss:.4f} | "
                    f"Val perplexity: {math.exp(val_loss):.2f}"
                )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), train_cfg.checkpoint_path)
                    print(f"  → Checkpoint saved (val loss {val_loss:.4f})")

        avg_epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1} complete | Avg train loss: {avg_epoch_loss:.4f}")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")


def quantize_model(model: MoELanguageModel, save_path: str) -> MoELanguageModel:
    """
    Applies dynamic INT8 quantization to all Linear layers.

    Args:
        model:     Trained MoELanguageModel (moved to CPU internally).
        save_path: Path to save the quantized model state dict.

    Returns:
        Quantized model ready for CPU inference.
    """
    model_cpu = model.to('cpu')
    model_cpu.eval()

    quantized = torch.quantization.quantize_dynamic(
        model_cpu,
        qconfig_spec={nn.Linear},
        dtype=torch.qint8
    )

    torch.save(quantized.state_dict(), save_path)

    # Report size comparison
    def _param_bytes(m: nn.Module) -> int:
        return sum(p.numel() * p.element_size() for p in m.parameters())

    orig_mb  = _param_bytes(model_cpu) / 1024 / 1024
    q_buf_mb = sum(b.numel() * b.element_size() for b in quantized.buffers()) / 1024 / 1024
    print(f"Quantized model saved to: {save_path}")
    print(f"  Parameter size: {orig_mb:.2f} MB  →  Buffer (INT8) size: {q_buf_mb:.2f} MB")

    return quantized

model = MoELanguageModel(model_cfg).to(DEVICE)

optimizer = torch.optim.AdamW(
    model.parameters(), lr=train_cfg.learning_rate, weight_decay=0.1
)
criterion = nn.CrossEntropyLoss()

print(f"Training on {DEVICE} …\n")
train(model, train_loader, val_loader, optimizer, criterion, train_cfg, model_cfg)

print("Quantize the model \n")
model.load_state_dict(torch.load(train_cfg.checkpoint_path, map_location='cpu'))
quantized_model = quantize_model(model, train_cfg.quantized_path)