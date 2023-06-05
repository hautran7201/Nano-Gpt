import torch
import torch.nn as nn
from torch.nn import functional as F


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)

        if targets == None:
            loss = None

        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    def get_batch(self, data, block_size, batch_size):        
        ix = torch.randint(len(data)-block_size, (batch_size, ))
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])
        return x, y

    def train(self, data, block_size, batch_size, learning_rate, train_iteration=10000):
        
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate)
        training_loss = []

        for _ in range(train_iteration):
            xb, yb = self.get_batch(data, block_size, batch_size)
            logits, loss = self(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            training_loss.append(loss)

        return training_loss

    @torch.no_grad()
    def estimate_loss(self, train, val=None, eval_iteration=100, block_size=8, batch_size=4):
        out = {}
        self.eval()

        if val == None:
            datas = {'train': train}
        else:
            datas = {'train': train, 'val': val}

        for key in datas.keys():
            losses = torch.zeros(eval_iteration)

            for k in range(eval_iteration):
                xb, yb = self.get_batch(datas[key], block_size, batch_size)
                logits, loss = self(xb, yb)
                losses[k] = loss.item()
            
            out[key] = losses.mean()

        self.train()

        return out
    




