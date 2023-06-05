import plotly.express as px
from DataPreparation import DataPreprocessing
from BigramLanguageModel import BigramLanguageModel

# Load data
path = 'Data\input.txt'
file = open(path, 'r', encoding='utf-8').read()[:1000]

# Split data
ratio = 0.9
pre_data = DataPreprocessing(file)
train, val = pre_data.train_val_split(ratio)
print('Training data length:', len(train), '\nValidation data length:', len(val))

# Create model
model = BigramLanguageModel(pre_data.vocab_size)

# Training model
block_size = 8
batch_size = 12
learning_rate = 1e-3
train_iteration = 1000
losses = model.train(train, block_size, batch_size, learning_rate, train_iteration = 1000)

# Loss plot
fig = px.scatter([tensor.tolist() for tensor in losses], range(len(losses)))
fig.update_xaxes(title_text='Iteration')
fig.update_yaxes(title_text='Loss')
fig.show()
