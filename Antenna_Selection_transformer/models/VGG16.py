# author: baiCai
# 导包
import torch
from torch import nn
# from torchsummary import summary
from torchvision.models import VGG

# 创建模型
class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        # 特征提取层
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=64,kernel_size=3,stride=1,padding=1),
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1),
            # nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            # nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            # nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # 分类层
        self.classifier = nn.Sequential(
            # 全连接的第一层，输入肯定是卷积输出的拉平值，即6*6*256
            # 输出是由AlexNet决定的，为4096
            nn.Linear(in_features=2*2*512,out_features=4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=4096,out_features=4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            # 最后一层，输出1000个类别，也是我们所说的softmax层
            nn.Linear(in_features=4096,out_features=785)  #785?
        )

    def forward(self,x):  # 要求输入bchw
        x = x.reshape(-1, 1, 8, 8)
        x = self.features(x)
        # print("Output shape:", x.shape)
        # 不要忘记在卷积--全连接的过程中，需要将数据拉平，之所以从1开始拉平，是因为我们
        # 批量训练，传入的x为[batch（每批的个数）,x(长),x（宽）,x（通道数）]，因此拉平需要从第1（索引，相当于2）开始拉平
        # 变为[batch,x*x*x]
        x = torch.flatten(x,1)
        result = self.classifier(x)
        return result


# if __name__ == "__main__":
# # Test the modified model with random input
#     model = VGG16()
#     input_tensor = torch.randn(16, 1, 64)  # Example input tensor
#     output = model(input_tensor)
#     print("Output shape:", output.shape)

# if __name__ == '__main__':
#     # 10分类
#     model = VGG16().to('cuda:0')
#     summary(model, (16, 1, 64))