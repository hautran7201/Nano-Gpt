import torch
import numpy as np 

class DataPreprocessing():
    def __init__(self, dataset):
        self.char = sorted(set(dataset))
        self.vocab_size = len(self.char)
        
        self.dataset = dataset
        self.dataset_size = len(dataset)

        self.stoi = {v:i for i, v in enumerate(self.char)}
        self.itos = {i:v for i, v in enumerate(self.char)}

    def encode(self, string):
        return [self.stoi[s] for s in string]
    
    def decode(self, indices):
        return ''.join([self.itos[i] for i in indices])
    
    def train_val_split(self, ratio):
        n = int(self.dataset_size * ratio)
        data = torch.tensor(self.encode(self.dataset), dtype=torch.long)

        train = data[:n]
        val = data[n:]

        return train, val