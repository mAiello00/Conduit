import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    @param vocal_size:  number of unique tokens in our vocabulary
    @param d_model:     size of the vector that represents each token after embedding
    @param n_experts:   number of experts we will have
    @param top_k:       number of experts we are routing to
"""
class GatingNetwork(nn.Model):
    def __init__(self, vocab_size: int, d_model: int, n_experts: int = 3, top_k: int = 1):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_experts = n_experts
        self.top_k = top_k

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx = 1)
        self.FFN == nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLu(),
            nn.Linear(d_model//2, n_experts)
        )

    def forward(self, x):
        # compute routing logits
        logits = self.FFN(x)

        # compute probabilities using softmax
        router_probs = F.softmax(logits, dim = -1)

        # select top k experts
        routing_weights, selected_experts = torch.topk(router_probs, self.top_k, dim = -1)

        # renormalize weights 
        routing_weights = routing_weights / routing_weights.sum(dim = -1, keepdim = True)

        return routing_weights, selected_experts, router_probs
    
def train():
    pass