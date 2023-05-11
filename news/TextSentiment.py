import torch
import torchtext
from torchtext.datasets import text_classification
import os

load_data_path = './data'
if not os.path.isdir(load_data_path):
    os.mkdir(load_data_path)

train_dataset, test_dataset = text_classification.DATASETS['AG_NEWS'](root=load_data_path)

import torch.nn as nn
import torch.nn.functional as F

BATCH_SIZE = 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TextSentiment(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super().__init__()
        self.embdding = nn.Embedding(vocab_size, embed_dim, num_class)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embdding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)

        self.fc.bias.data.zero_()
    
    def forward(self, text):
        embeded = self.embdding(text)

        c = embeded.size(0) // BATCH_SIZE
        embeded = embeded[:BATCH_SIZE*c]
        embeded = embeded.transpose(1, 0).unsqueeze(0)
        embeded = F.avg_pool1d(embeded, kernel_size=c)
        return self.fc(embeded[0].transpose(1, 0))


VOCAB_SIZE = len(train_dataset.get_vocab())

EMBED_DIM = 32

NUM_CLASS = len(train_dataset.get_labels())

model = TextSentiment(VOCAB_SIZE, EMBED_DIM, NUM_CLASS).to(device)

def generte_batch(batch):
    label = torch.tensor([entry[0] for entry in batch])
    text = [entry[1] for entry in batch]
    text = torch.cat(text)

    return text, label

from torch.utils.data import DataLoader
from torch import optim

criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=4.0)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)#衰减学习率

def train(train_data):
    train_loss = 0
    train_acc = 0
    # data = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    data = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=generte_batch)

    for i, (text, cls) in enumerate(data):
        optimizer.zero_grad()
        output = model(text)
        loss = criterion(output, cls)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        train_acc += (output.argmax(1) == cls).sum().item()

    scheduler.step()

    return train_loss / len(train_data), train_acc / len(train_data)


def valid(valid_data):
    loss = 0
    acc = 0
    data = DataLoader(valid_data, batch_size=BATCH_SIZE, collate_fn=generte_batch)
    for text, cls in data:
        with torch.no_grad():
            output = model(text)
            loss = criterion(output, cls)
            loss += loss.item()
            acc += (output.argmax(1) == cls).sum().item()
    return loss / len(valid_data), acc / len(valid_data)

import time
from torch.utils.data.dataset import random_split

N_EPOCHS = 10
min_valid_loss = float('inf')

train_len = int(len(train_dataset) * 0.95)

sub_train_, sub_valid_ = \
    random_split(train_dataset, [train_len, len(train_dataset) - train_len])

for epoch in range(N_EPOCHS):
    start_time = time.time()
    train_loss, train_acc = train(sub_train_)
    valid_loss, valid_acc = valid(sub_valid_)

    secs = int(time.time() - start_time)
    mins = secs /60
    secs = secs % 60

    print('Epoch: %d' %(epoch + 1),"| time in %d minutes, %d seconds" %(mins, secs))
    print(f'\tLoss: {train_loss:.4f}(train)\t|\tAcc:{train_acc * 100:.1f}%(train)')
    print(f'\tLoss: {valid_loss:.4f}(valid)\t|\tAcc:{valid_acc * 100:.1f}%(valid)')






    

