#CIFAR10
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
show = ToPILImage()


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# (data, label) = trainset[66]  
# print(data.size())  # 验证某一张图片的维度 —— 3*32*32
# print(classes[label]) # label是一个0-9的数字
# (data + 1) / 2是为了还原被归一化的数据 （这部分计算是可以推算出来的）
# show((data + 1) / 2).resize((100, 100))

# dataiter = iter(trainloader)
# images, labels = dataiter.next()
#print(' '.join('%11s'%classes[labels[j]] for j in range(4)))
#show(torchvision.utils.make_grid((images+1)/2)).resize((400,100))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) 
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1   = nn.Linear(16 * 5 * 5, 120)  
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)  # 最后是一个十分类，所以最后的一个全连接层的神经元个数为10

    def forward(self, x): 
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2)) 
        x = F.max_pool2d(F.relu(self.conv2(x)), 2) 
        x = x.view(x.size()[0], -1)  # 展平  x.size()[0]是batch size
        #x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)        
        return x
net = Net()
# print(net)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
'''
torch.set_num_threads(4)

for epoch in range(2):

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()

        outputs = net(inputs)

        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999: # 每2000个batch打印一下训练状态
            print('[%d, %5d] loss: %.3f' \
                  % (epoch+1, i+1, running_loss / 2000))
            running_loss = 0.0
print('Finished')

PATH = './cifa_net.pth'
torch.save(net.state_dict(), PATH)
'''
PATH = './cifa_net.pth'
net = Net()
net.load_state_dict(torch.load(PATH))

correct = 0 # 预测正确的图片数
total = 0 # 总共的图片数

# 由于测试的时候不需要求导，可以暂时关闭autograd，提高速度，节约内存
with torch.no_grad():
    for data in testloader:      # data是个tuple
        images, labels = data    # image和label 都是tensor        
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)    # labels tensor([3, 8, 8, 0])            labels.size: torch.Size([4])
        correct += (predicted == labels).sum()

print('10000张测试集中的准确率为: %d %%' % (100 * correct / total))
