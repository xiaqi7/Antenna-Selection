import torch
import torch.nn as nn
# from torchsummary import summary


class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim):
        super(RNN, self).__init__()
        # 因为序列长度固定为1，RNN实际上等同于一个全连接层  
        # 隐藏层维度就是RNN层的输出维度  
        self.rnn = nn.RNN(input_size, hidden_dim, batch_first=True)
        # 全连接层将隐藏层维度映射到输出维度  
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        # 因为batch_first=True，所以输入形状为(batch_size, seq_len, input_size)  
        # x的形状为(16, 1, 64)，其中seq_len=1  
        # 初始化隐藏状态，形状为(num_layers * num_directions, batch_size, hidden_dim)  
        # 因为是单向RNN且只有一层，所以num_layers=1, num_directions=1  
        h0 = torch.zeros(1, x.size(0), self.rnn.hidden_size).to(x.device)  # 假设使用CPU或GPU  

        # RNN处理输入x，但因为seq_len=1，RNN不会进行递归  
        out, _ = self.rnn(x, h0)
        # 取最后一个时间步的输出（因为只有一个时间步，所以就是out本身）  
        # out的形状为(batch_size, seq_len, hidden_dim)，即(16, 1, hidden_dim)  
        out = out.squeeze(1)  # 移除seq_len维度，形状变为(16, hidden_dim)  
        # 全连接层映射到输出维度  
        output = self.fc(out)
        # output的形状为(16, output_size)，需要确保output_size=784  
        return output

    # 实例化模型，确保output_size为784

if __name__ == '__main__':
    # 10分类
    model = RNN(input_size=64, output_size=784, hidden_dim=256)  # hidden_dim可以根据需要调整
    # 假设输入tensor x
    x = torch.randn(16, 1, 64)  # 形状为(batch_size, seq_len, input_size)

    # 通过模型获取输出
    output = model(x)
    print(output.shape)  # 输出形状应为torch.Size([16, 784])

