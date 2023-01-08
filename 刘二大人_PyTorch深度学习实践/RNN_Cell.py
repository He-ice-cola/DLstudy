import torch

# 一、准备数据

idx2char = ['e', 'h', 'l', 'o']  # 列表。作用：idx2char[0]=e idx2char[1]=h
x_data = [1, 0, 2, 2, 3]  # 'hello'
y_data = [3, 1, 2, 3, 2]  # 'ohlol'

batch_size = 1
input_size = 4
hidden_size = 4

# 通过 onehot编码 ，对每个字母进行向量化表示
one_hot_lookup = [[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]]
x_one_hot = [one_hot_lookup[x] for x in x_data]   # 将x_data中的每个元素 onehot编码 ，得到的是 list
# print(x_one_hot) [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

inputs = torch.Tensor(x_one_hot).view(-1, batch_size, input_size)
# view()重构张量 -1 表示一个不确定的数，它的大小根据数据推断出来。
labels = torch.LongTensor(y_data).view(-1, 1)  # labels 维度 (seqLen,1)
# Tensor 默认数据类型为浮点型  ， LongTensor 的数据类型是整数类型


# 二、 定义模型

class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, batch_size):  # __init__ 绑定属性
        super(Model, self).__init__()
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnncell = torch.nn.RNNCell(input_size=self.input_size,
                                        hidden_size=self.hidden_size)

    def forward(self, input, hidden):
        hidden = self.rnncell(input, hidden)
        return hidden

    def init_hidden(self):  #初始 h0
        return torch.zeros(self.batch_size, self.hidden_size)

net = Model(input_size, hidden_size, batch_size)


# 三、 优化
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1)

# 四、 训练
for epoch in range(15):
    loss = 0
    optimizer.zero_grad()
    hidden = net.init_hidden()   # 每一轮都要初始化 h0
    print('Predicted string: ', end='')

    for input, label in zip(inputs, labels):
        hidden = net(input, hidden)
        loss += criterion(hidden, label)
        _, idx = hidden.max(dim=1)
        print(idx2char[idx.item()], end='')
    loss.backward()
    optimizer.step()
    print(', Epoch[% d/15] loss = % .4f' % (epoch + 1, loss.item()))


