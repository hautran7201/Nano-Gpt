import torch
from data.load_dataset import TxtDataset
from model import BigramLanguageModel
from opt import config_parser
from utils import get_batch

dir_path = r'config/input.txt'

# Dataset
dataset = TxtDataset(dir_path)

# Config
args = config_parser()

# Model
model = BigramLanguageModel(
    args,
    vocab_size=dataset.num_chars
)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

# Split data
train_data, val_data = dataset.train_test_split(0.9)

# Train and generate
if args.train_only == 1:
    path = '/content/drive/MyDrive/project/nano_gpt/check_point/ckpt.pth'
    for step in range(args.training_step):
        x, y = get_batch(train_data, args.block_size, args.batch_size)

        if step % args.eval_interval == 0:
            train_loss = model.estimate_loss(train_data, args.eval_interval)
            eval_loss = model.estimate_loss(val_data, args.eval_interval)
            print(f'Iter: {step}, Train: {train_loss:.4f}, eval: {eval_loss:.4f}')

        lotgits, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    torch.save(model, path)

elif args.generation == 1:
    x, y = dataset.get_batch(train_data)
    logits, loss = model(x, y)

    idx = torch.zeros((1, 1), dtype=torch.long)
    generated_text = dataset.decode(model.generate(idx, 100)[0].tolist())
    print(generated_text)