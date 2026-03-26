import torch
import torch.nn as nn

'''
    @param vocal_size:  number of unique tokens in our vocabulary
    @param d_model:     size of the vector that represents each token after embedding
    @param n_experts:   number of experts we will have
'''
class GatingNetwork(nn.Model):
    def __init__(self, vocab_size: int, d_model: int, n_experts: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx = 1)
        self.FFN == nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLu(),
            nn.Linear(d_model//2, n_experts)
        )

    def forward(self):
        pass