# 创建AlexNet模型 227*227
import torch
from torch import nn
# from torchsummary import summary


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        # 特征提取
        self.features = nn.Sequential(
            # 输入通道数为3，因为图片为彩色，三通道
            # 而输出96、卷积核为11*11，步长为4，是由AlexNet模型决定的，后面的都同理
            nn.Conv2d(in_channels=1,out_channels=96,kernel_size=5,stride=2),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Conv2d(in_channels=96,out_channels=256,kernel_size=3,padding=2),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2,stride=1),
            nn.Conv2d(in_channels=256,out_channels=384,padding=1,kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=384,out_channels=384,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=384,out_channels=256,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        # 全连接层
        self.classifier = nn.Sequential(
            # 全连接的第一层，输入肯定是卷积输出的拉平值，即6*6*256
            # 输出是由AlexNet决定的，为4096
            nn.Linear(in_features=2*2*256,out_features=4096),
            nn.ReLU(),
            # AlexNet采取了DropOut进行正则，防止过拟合
            nn.Dropout(p=0.5),
            nn.Linear(4096,4096),
            nn.ReLU(),
            # 最后一层，输出1000个类别，也是我们所说的softmax层
            nn.Linear(4096,785)
        )

    # 前向算法
    def forward(self,x):
        x = x.reshape(-1, 1, 8, 8)
        x = self.features(x)
        # 不要忘记在卷积--全连接的过程中，需要将数据拉平，之所以从1开始拉平，是因为我们
        # 批量训练，传入的x为[batch（每批的个数）,x(长),x（宽）,x（通道数）]，因此拉平需要从第1（索引，相当于2）开始拉平
        # 变为[batch,x*x*x]
        x = torch.flatten(x,1)
        result = self.classifier(x)
        return result

# if __name__ == "__main__":
# # Test the modified model with random input
#     model = AlexNet()
#     input_tensor = torch.randn(16, 1, 64)  # Example input tensor
#     output = model(input_tensor)
#     print("Output shape:", output.shape)

# if __name__ == '__main__':
#     # 10分类
#     model = AlexNet().to('cuda:0')
#     summary(model, (16, 1, 64))