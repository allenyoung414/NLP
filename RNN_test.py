import numpy as np
import matplotlib.pyplot as plt
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torch.utils.data as Data
from torchvision import transforms
import hiddenlayer as hl
from sklearn.metrics import classification_report, accuracy_score 
# %matpotlib inline

train_data = torchvision.datasets.MNIST(root="./data", train=True, transform=transforms.ToTensor(), download=False)
train_loader = Data.DataLoader(dataset=train_data, batch_size=64, shuffle=True, num_workers=0)

test_data = torchvision.datasets.MNIST(root="./data", train=False, transform=transforms.ToTensor(), download=False)
test_loader = Data.DataLoader(dataset=test_data, batch_size=64, shuffle=True,num_workers=0)

for batch in train_loader:
    x, y = batch
    print(x.shape)
    print(y.shape)
    break

class RNNimc(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim) -> None:
        super(RNNimc, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first = True, nonlinearity = "relu")
        self.fc1 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        out, h_n = self.rnn(x, None)
        out = self.fc1(out[: , -1, :])
        return out 

input_dim = 28       # 输入维度
hidden_dim = 128     # RNN神经元个数
layer_dim = 1        # RNN的层数
output_dim = 10      # 输出维度
MyRNNimc = RNNimc(input_dim, hidden_dim, layer_dim, output_dim)
print(MyRNNimc)

optimizer = optim.RMSprop(MyRNNimc.parameters(), lr=3e-4)
criterion = nn.CrossEntropyLoss()
train_loss_all = []
train_acc_all = []
test_loss_all = []
test_acc_all = []
num_epoch = 30

for epoch in range(num_epoch):
    print("Epoch {}/{}".format(epoch, num_epoch - 1))
    MyRNNimc.train() # 模式设为训练模式
    train_loss = 0
    corrects = 0
    train_num = 0
    for step, (b_x, b_y) in enumerate(train_loader):
        output = MyRNNimc(b_x.view(-1, 28, 28))
        pre_lab = torch.argmax(output, 1)
        loss = criterion(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * b_x.size(0)
        corrects += torch.sum(pre_lab == b_y.data)
        train_num += b_x.size(0)
    train_loss_all.append(train_loss / train_num)
    train_acc_all.append(corrects.double().item() / train_num)
    print("{}, Train Loss: {:.4f} Train Acc: {:.4f}".format(epoch, train_loss_all[-1], train_acc_all[-1]))
    MyRNNimc.eval()
    corrects, test_num, test_loss = 0, 0, 0
    for step, (b_x, b_y) in enumerate(test_loader):
        output = MyRNNimc(b_x.view(-1, 28, 28))
        pre_lab = torch.argmax(output, 1)
        loss = criterion(output, b_y)
        test_loss += loss.item() * b_x.size(0)
        corrects += torch.sum(pre_lab == b_y)
        test_num += b_x.size(0)
    # 计算经过一个epoch的训练后再测试集上的损失和精度
    test_loss_all.append(test_loss / test_num)
    test_acc_all.append(corrects.double().item() / test_num)
    print("{} Test Loss: {:.4f} Test Acc: {:.4f}".format(epoch, test_loss_all[-1], test_acc_all[-1]))
torch.save(MyRNNimc, "./data/RNNimc.pkl")

plt.figure(figsize=[14, 5])
plt.subplot(1, 2, 1)
plt.plot(train_loss_all, "ro-", label="Train Loss")
plt.plot(test_loss_all, "bs-", label="Val Loss")
plt.legend()
plt.xlabel("epoch")
plt.ylabel("Loss")

plt.subplot(1, 2, 2)
plt.plot(train_acc_all, "ro-", label="Train Acc")
plt.plot(test_acc_all, "bs-", label="Test Acc")
plt.xlabel("epoch")
plt.ylabel("Acc")
plt.legend()

plt.show()
