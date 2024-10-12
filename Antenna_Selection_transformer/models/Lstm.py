import torch
import torch.nn as nn

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)

        out, _ = self.lstm(x.to(DEVICE), (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

if __name__ == '__main__':
        # 10分类
        model = LSTM(input_size=64,hidden_size=256, num_layers=2,  output_size=785).to(DEVICE)  # hidden_dim可以根据需要调整
        # 假设输入tensor x
        x = torch.randn(16, 1, 64)  # 形状为(batch_size, seq_len, input_size)

        # 通过模型获取输出
        output = model(x)
        print(output.shape)  # 输出形状应为torch.Size([16, 784])