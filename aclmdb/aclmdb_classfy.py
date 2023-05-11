import collections
import os
import random
import tarfile
import torch
from torch import nn
import torchtext.vocab as Vocab
import torch.utils.data as Data
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#DATA_ROOT = "E:\data" # 数据集所在路径
DATA_ROOT = "C:\\Users\\87333\\Desktop\\NLP\\transformer" # 数据集所在路径

# 1、读取数据
#fname = os.path.join(DATA_ROOT, 'aclImdb_v1.tar.gz')
# 将压缩文件进行解压
#if not os.path.exists(os.path.join(DATA_ROOT, 'aclImdb')):
   # print("从压缩包解压...")
    #with tarfile.open(fname, 'r') as f:
     #   f.extractall(DATA_ROOT) # 解压文件到此指定路径

from tqdm import tqdm # 可查看读取数据的进程

def read_imdb(folder='train', data_root=r'C:\Users\87333\Desktop\NLP\transformer\aclImdb'):
    data = []
    for label in ['pos', 'neg']:
        folder_name = os.path.join(data_root, folder, label) # 拼接文件路径 如：E:\data\aclImdb\train\pos\
        for file in tqdm(os.listdir(folder_name)): # os.listdir(folder_name) 读取文件路径下的所有文件名，并存入列表中
            with open(os.path.join(folder_name, file), 'rb') as f:
                review = f.read().decode('utf-8').replace('\n', ' ').lower()
                data.append([review, 1 if label == 'pos' else 0]) # 将每个文本读取的内容和对应的标签存入data列表中
    random.shuffle(data) # 打乱data列表中的数据排列顺序
    return data

train_data, test_data =read_imdb('train'), read_imdb('test')

# 2、预处理数据
# 空格分词
def get_tokenized_imdb(data):
    '''
    :param data: list of [string, label]
    '''
    def tokenizer(text):
        return [tok.lower() for tok in text.split(' ')]
    return [tokenizer(review) for review,_ in data] # 只从data中读取review(评论)内容而不读取标签(label)，对review使用tokenizer方法进行分词

# 创建词典
def get_vocab_imdb(data):
    tokenized_data = get_tokenized_imdb(data) # 调用get_tokenized_imdb()空格分词方法获取到分词后的数据tokenized_data
    counter = collections.Counter([tk for st in tokenized_data for tk in st]) # 读取tokenized_data列表中每个句子的每个词，放入列表中。
                                                                              # collections.Counter()方法可计算出列表中所有不重复的词数总和
    return Vocab.Vocab(counter, min_freq=5) # 去掉词频小于5的词

vocab = get_vocab_imdb(train_data)
# print(len(vocab)) # 46152
# print(vocab.stoi['hello']) # 8950

# 对data列表中的每行数据进行处理，将词转换为索引，并使每行数据等长
def process_imdb(data, vocab):
    max_len = 500 # 每条评论通过截断或者补0，使得长度变成500

    def pad(x):
        return x[:max_len] if len(x) > max_len else x + [0]*(max_len - len(x)) # x[:max_len] 只获取前max_len个词
                                                                               # x + [0]*(max_len - len(x)) 词数小于max_len,用pad=0补长到max_len

    tokenized_data = get_tokenized_imdb(data) # 调用方法获取分词后的数据
    features = torch.tensor([pad([vocab.stoi[word] for word in words]) for words in tokenized_data]) # 将词转换为vocab词典中对应词的索引
    labels = torch.tensor([score for _, score in data])
    # print(features.size())# 25000*500
    # print(labels.size())# 500
    return features, labels

# 3、创建数据迭代器
batch_size = 64

train_set = Data.TensorDataset(*process_imdb(train_data, vocab))
test_set = Data.TensorDataset(*process_imdb(test_data, vocab))


train_iter = Data.DataLoader(train_set, batch_size, True)
test_iter = Data.DataLoader(test_set, batch_size)

for X, y in train_iter:
    print('X', X.shape, 'y', y.shape)
    break
# '#batches:', len(train_iter)
# X torch.Size([64, 500]) y torch.Size([64])

# 4、创建循环神经网络


# 4、创建循环神经网络
class BiRNN(nn.Module):
    def __init__(self, vocab, embed_size, num_hiddens, num_layers):
        super(BiRNN, self).__init__()
        self.embedding = nn.Embedding(len(vocab), embed_size)
        self.encoder = nn.LSTM(
            input_size=embed_size,
            hidden_size=num_hiddens,
            num_layers=num_layers,
            # batch_first=True,
            bidirectional=True
        )
        self.decoder = nn.Linear(4*num_hiddens, 2)

    def forward(self, inputs):
        # inputs: [batch_size, seq_len], LSTM需要将序列长度(seq_len)作为第一维，所以需要将输入转置后再提取词特征
        # 输出形状 outputs: [seq_len, batch_size, embedding_dim]
        embeddings = self.embedding(inputs.permute(1, 0))
        # rnn.LSTM只传入输入embeddings, 因此只返回最后一层的隐藏层在各时间步的隐藏状态。
        # outputs形状是(seq_len, batch_size, 2*num_hiddens)
        outputs, _ = self.encoder(embeddings)
        # 连结初始时间步和最终时间步的隐藏状态作为全连接层输入。
        # 它的形状为 : [batch_size, 4 * num_hiddens]
        encoding = torch.cat((outputs[0], outputs[-1]), dim=-1)
        outs = self.decoder(encoding)
        return outs

# 创建一个包含两个隐藏层的双向循环神经网络
embed_size, num_hiddens, num_layers = 100, 100, 2
net = BiRNN(vocab, embed_size, num_hiddens, num_layers)


# 5、加载预训练的词向量
# 为词典vocab中的每个词加载100维的GloVe词向量
glove_vocab = Vocab.GloVe(name='6B', dim=100, cache=os.path.join(DATA_ROOT, 'glove'))
# print(len(glove_vocab.stoi)) # 400000
print(glove_vocab[0].shape)
def load_pretrained_embedding(words, pretrained_vocab):
    '''从训练好的vocab中提取出words对应的词向量'''
    embed = torch.zeros(len(words), pretrained_vocab.vectors[0].shape[0]) # pretrained_vocab.vectors[0].shape # torch.Size([100])
    oov_count = 0 # out of vocabulary
    for i, word in enumerate(words):
        try:
            idx = pretrained_vocab.stoi[word]
            embed[i, :] = pretrained_vocab.vectors[idx] # 将第i行用预训练的单词向量替换
        except KeyError:
            oov_count += 1
    if oov_count > 0:
        print("There are %d oov words." % oov_count)
    return embed
net.embedding.weight.data.copy_(
    load_pretrained_embedding(vocab.itos, glove_vocab)
)
net.embedding.weight.requires_grad = False # 直接加载预训练好的，所以不需要更新它 

lr, num_epochs = 0.01, 5
# 要过滤掉不计算梯度的embedding参数
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
loss = nn.CrossEntropyLoss()


# 评估
def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval() # 评估模式, 这会关闭dropout
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                net.train() # 改回训练模式
            else: # 自定义的模型, 3.13节之后不会用到, 不考虑GPU
                if('is_training' in net.__code__.co_varnames): # 如果有is_training这个参数
                    # 将is_training设置成False
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item()
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
    return acc_sum / n

# 训练
def train(train_iter, test_iter, net, loss, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on ", device)
    batch_count = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))

# 开始训练
train(train_iter, test_iter, net, loss, optimizer, device, num_epochs)


