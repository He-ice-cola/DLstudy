import torch

# 一、准备数据
idx2char = ['e', 'h', 'l', 'o']  # idx2char[0]=e     idx2char[1]=h
x_data = [1, 0, 2, 2, 3]  # 'hello'
y_data = [3, 1, 2, 3, 2]  # 'ohlol'
# 通过 onehot编码 ，对每个字母进行向量化表示
one_hot_lookup = [[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]]
x_one_hot = [one_hot_lookup[x] for x in x_data]  # 将x_data中的每个元素 onehot编码 ，得到的是 list
inputs = torch.Tensor(x_one_hot).view(5, 1, 4)
labels = torch.LongTensor(y_data)  # torch.Size([5])


# 二、定义模型
class Model(torch.nn.Module):
    def __init__(self, input_size=4, hidden_size=4, batch_size=1, num_layers=1):
        super(Model, self).__init__()
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(input_size=self.input_size,
                                  hidden_size=self.hidden_size,
                                  num_layers=self.num_layers,
                                  )

    def forward(self, input):
        h_0 = torch.zeros(self.num_layers,
                          self.batch_size,
                          self.hidden_size)  # 构造 h_0
        c_0 = torch.zeros(self.num_layers,
                          self.batch_size,
                          self.hidden_size)  # 构造 c_0
        h_t, (h, c) = self.lstm(input, (h_0, c_0))  # output h_t = rnn(input h_0)
        return h_t.view(-1, 4)
        # output 旧的shape(seq_Len,batch_size,hidden_size * num_directions) view 成（-1, hidden_size） 即(5,4)
        # 这一步骤是因为 交叉熵损失函数中两个参数: input 大小为 (5,4)  Target 大小为 (5) (即labels大小)


net = Model(input_size=4, hidden_size=4, batch_size=1, num_layers=1)

# 三、优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.05)

# 四、训练
for epoch in range(15):
    optimizer.zero_grad()
    outputs = net(inputs)  # outputs 的size为 (5,4)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    _, idx = outputs.max(dim=1)
    # torch.max 返回的是 值，索引。 dim = 1 表示 从维度为1角度对比，因为 outputs 是(5,4) 从每一行中选出最大值 得到 torch.Size([5])
    # https://pytorch.org/docs/stable/generated/torch.max.html#torch.max
    idx = idx.data.numpy()
    print('预测结果：', ''.join([idx2char[x] for x in idx]), end='')
    print(', Epoch[% d/15] loss = % .4f' % (epoch + 1, loss.item()))
