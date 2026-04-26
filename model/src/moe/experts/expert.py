import torch
from torch.nn import nn
import torch.nn.functional as F
import math
import sys

# Hyperparameters
batch_size = 64 # number of independent sequences being processed in parallel
block_size = 256 # maximum context length for predictions
max_iterations = 5000
eval_interval = 500
learning_rate = 3e-4
epochs = 100
n_embd = 384
n_head = 6
n_layer = 6
dropout  = 0.2
# ----------------

# make environment variable
vocab_size = 65
MODE = "TMP" 
assert MODE in ["TMP", "TRAIN", "TEST", "INFERENCE"]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(1337) # setting seed so we can reproduce our results

class Head(nn.Module):
    def __init__(self):
        pass

class SingleHeadAttention(nn.Module):
    def __init(self):
        super().__init__()

    def forward(self):
        pass

class MultiHeadAddtention(nn.Module):
    """ Multiple heads of self attention in parallel"""
    def __init__(self, num_heads, head_size):
        super().__init()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.projection = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim = -1) # concatenating over the channel dimension
        out = self.projection(out) # the projection is just a linear transformation of the outcome of the above layer
        return out

class TransformerBlock(nn.Module):
    """
        Communication followed by computation.
        Implementaiton of the Nx block in Attention is All You Need, but without the cross-attention
    """
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.self_attention = MultiHeadAddtention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.LayerNorm1 = nn.LayerNorm(n_embd)
        self.LayerNorm2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # LayerNorm done before x goes into Self Attention and Feed Forward
        # # Slight deviation from Attention paper, but this is the new way to do it 
        x = x + self.self_attention(self.LayerNorm1(x))
        x = x + self.ffwd(self.LayerNorm2(x))
        return x


class FeedForward(nn.Module):
    """ Simple FFN followed by a lon-linearity"""
    def __init__(self, n_embd):
        super().__init__()
        # the 4 * comes from a multiplier in the Attention paper (dff 512 -> 2048) in the Position Wise Feed-Forward Networks section
        self.FFN = nn.Seqential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), # this is the prjection layer going back into the residual pathway
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.FFN(x)

class ExpertLLM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table

        # creating an embedding tabel of size vocab_size x vocab_size
        # nn.Embedding is a think wrapper around a tensor of shape vocab_size x vocab_size
        # 
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets = None):
        # idx and target are both (B, T) tensors of integers
        # when we pass in idx, every integer in our input will refer to the embedding table and plucks out a row
        # of that embedding table corresponding to its index
        # pytorch then arranges that into a (B, T, C) tensor (channel here is the vocab_size)
        # interpret this as the logits - the scores for the next character in the sequence
        # predicting what comes next on the identity of a single token
        # are the predictions
        logits =  self.token_embedding_table(idx) # (B(batch), T (time), C (channel)) tensor

        # if we have targets we provide them and get a loss
        # is we have no targets we get the logits
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape()
            logits = logits.view(B*T, C) # logits now conforms to how cross_entropy expects inputs
            targets = targets.view(B*T, C)
            # cross_entropy is negative log likelihood loss
            loss = F.cross_entropy(logits, targets)

        # self attend
        # call FFN
        return logits, loss
    
    # idx is the current context of some characters in a batch
    def generate(self, idx, max_new_tokens):
        # idx is a (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # gets the loss predictions
            logits, loss = self(idx) # this goes to the 'forward' function. not providing targets

            # focus only on the last time step
            logits = logits[:, -1, :] # become (B, C)

            # apply softmax to get probabilities
            probs = F.softmax(logits, dim = 1) # (B, C)

            #sample from the distribution
            idx_next = torch.multinomial(probs, num_samples = 1) # (B, 1)

            # append sampled index to the running space

            idx = torch.cat((idx, idx_next), dim = 1) # (B, T + 1)

        return idx
    
model = ExpertLLM(vocab_size)
m = model.to(device)

def train():
    # train
    optimizer = torch.optim.AdamW(model.parameters(), learning_rate)

    for iter in range(epochs):
        # sample batch of data
        xb, yb = get_batch('trian')

        # evaluate the loss
        optimizer.zero_grad(set_to_none = True) # zero out gradients from the previous step
        logits, loss = model(xb, yb)
        loss.backward() # get gradients for all the parameters
        optimizer.step() # update hte parameters

    print(loss.item())

def Tokenize(input):
    pass

def Test():
    pass

# TODO: properly format strings
def ReadInput():
    input = []
    for line in sys.stdin:
        input.append(line)
        if line == "<END>":
            break
    return input.join("")
            
def main():

    prompt = ReadInput()

    if MODE == "TMP":
        Test(prompt)
    elif MODE == "TRAIN":
        pass
    elif MODE == "TEST":
        pass
    elif MODE == "INFERENCE":
        pass


if __name__ == "__main__":
    main()