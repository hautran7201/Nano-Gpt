import torch

class TxtDataset:
    def __init__(self, dir_path, block_size=8, batch_size=4):
        self.text = open(dir_path, 'r', encoding='utf-8').read()
        self.chars = sorted(set(self.text))
        self.num_chars = len(self.chars)
        self.block_size = block_size
        self.batch_size = batch_size
        self.c2i = {c:i for i, c in enumerate(self.chars)}
        self.i2c = {i:c for i, c in enumerate(self.chars)}
        self.encode = lambda text: [self.c2i[char] for char in text]
        self.decode = lambda id_list: ''.join([self.i2c[id] for id in id_list])
        self.encoded_text = torch.tensor(self.encode(self.text), dtype=torch.long)

    def encode(self, text):
        return self.encode(text)

    def decode(self, idx):
        return self.decode(idx)

    def train_test_split(self, ratio):
        n = int(ratio*len(self.encoded_text))
        train_data = self.encoded_text[:n]
        val_data = self.encoded_text[n:]
        return train_data, val_data

    def get_batch(self, data):
        idx = torch.randint(len(data)-self.block_size, (self.batch_size,))
        x = torch.stack([data[i:i+self.block_size] for i in idx])
        y = torch.stack([data[i+1:i+self.block_size+1] for i in idx])
        return x, y