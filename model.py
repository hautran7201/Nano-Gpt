import torch
import torch.nn as nn
from torch.nn import functional as F
from utils import get_batch

class Head(nn.Module):
    def __init__(self, embedded_size, head_size, block_size, dropout=0.2):
        super().__init__()
        self.key = nn.Linear(embedded_size, head_size, bias=False)
        self.query = nn.Linear(embedded_size, head_size, bias=False)
        self.value = nn.Linear(embedded_size, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.drop_out = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x) # (B, T, C)
        q = self.query(x) # (B, T, C)
        v = self.value(x) # (B, T, C)

        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T]==0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.drop_out(wei)

        out = wei @ v # (B, T, C)

        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_head, embedded_size, block_size, dropout=0.2):
        super().__init__()
        self.heads = [Head(embedded_size, embedded_size//num_head, block_size) for h in range(num_head)]
        self.proj = nn.Linear(embedded_size, embedded_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = torch.cat([h(x) for h in self.heads], dim=-1)        
        out = self.proj(x)
        out = self.dropout(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, embedded_size, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedded_size, 4*embedded_size),
            nn.ReLU(),
            nn.Linear(4*embedded_size, embedded_size),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, num_head, embedded_size, block_size):
        super().__init__()
        self.sa = MultiHeadAttention(num_head, embedded_size, block_size)
        self.ffn = FeedForward(embedded_size)
        self.ln1 = nn.LayerNorm(embedded_size)
        self.ln2 = nn.LayerNorm(embedded_size)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):
    def __init__(
        self, 
        args,
        vocab_size
    ):

        super().__init__()
        self.args = args
        self.token_embedding = nn.Embedding(vocab_size, args.embedded_size)
        self.position_embedding = nn.Embedding(args.block_size, args.embedded_size)
        self.blocks = nn.Sequential(
            *[Block(args.number_head, args.embedded_size, args.block_size) for _ in range(args.number_block)]
        )
        self.ln_f = nn.LayerNorm(args.embedded_size)
        self.ln_head = nn.Linear(args.embedded_size, vocab_size)

    def forward(self, idx, target=None):
        B, T = idx.shape

        # Logits and target (B, T)
        embedded_token = self.token_embedding(idx) # (B, T, C)
        embedded_position = self.position_embedding(torch.arange(T)) # (T, C)
        x = embedded_token + embedded_position # (B, T, C)
        x = self.blocks(x) # (B, T, C)
        x = self.ln_f(x)
        logits = self.ln_head(x) # (B, T, vocab_size)

        loss = None
        if target != None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            target = target.view(B*T)
            loss = F.cross_entropy(logits, target)

        return logits, loss

    def generate(self, idx, max_new_token):

        for t in range(max_new_token):
            idx_cond = idx[:, -self.args.block_size:]
            logits, loss = self(idx_cond)
            last_logits = logits[:, -1, :]
            prods = F.softmax(last_logits, dim=1)
            idx_next = torch.multinomial(prods, 1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx    

    def estimate_loss(self, data, eval_interval):
        self.eval()

        with torch.no_grad():
            losses = torch.zeros(eval_interval)
            for step in range(eval_interval):
                x, y = get_batch(data, self.args.block_size, self.args.batch_size)
                logits, loss = self(x, y)
                losses[step] = loss.item()
            mean_loss = losses.mean()

        self.train()
        return mean_loss    
