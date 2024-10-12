import torch

from torch import nn
# from torchsummary import summary


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 定义模型
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(3, 3), stride=1, padding=2),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(3, 3), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=256, out_features=120),
            nn.ReLU(),
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(),
            nn.Linear(in_features=84, out_features=785)
        )

    def forward(self, x):
        x = x.reshape(-1, 1, 8, 8)
        # 定义前向算法
        x = self.features(x)
        # print(x.shape)
        x = torch.flatten(x, 1)
        # print(x.shape)
        result = self.classifier(x)
        return result



# if __name__ == "__main__":
# # Test the modified model with random input
#     model = LeNet()
#     input_tensor = torch.randn(16, 1, 64)  # Example input tensor
#     output = model(input_tensor)
#     print("Output shape:", output.shape)

# if __name__ == '__main__':
#     # 10分类
#     model = LeNet().to('cuda:0')
#     summary(model, (16, 1, 64))