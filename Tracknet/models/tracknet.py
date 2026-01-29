import torch
from torch import nn

from torchsummary import summary


# 将BN的维度以参数的形式传入
class Conv(nn.Module):
    def __init__(self, ic, oc, bc, k=(3,3), p="same", act=True):
        super().__init__()
        self.conv = nn.Conv2d(ic, oc, kernel_size=k, padding=p)
        self.bn = nn.BatchNorm2d(bc)
        self.act = nn.ReLU() if act else nn.Identity()

    def forward(self, x):
        x = self.act(self.conv(x))

        x = x.transpose(1, 3)    #  NCHW ---> NWHC
        x = self.bn(x)
        x = x.transpose(1, 3)    #  NCHW <--- NWHC

        return x


class TrackNet(nn.Module):
    def __init__(self):
        super().__init__()

        # VGG16
        # self.conv2d_1 = Conv(3, 64)   输入3张灰度图
        self.conv2d_1 = Conv(9, 64, 512)     # 输入3张RGB图
        self.conv2d_2 = Conv(64, 64, 512)
        self.max_pooling_1 = nn.MaxPool2d((2,2), stride=(2,2))

        self.conv2d_3 = Conv(64, 128, 256)
        self.conv2d_4 = Conv(128, 128, 256)
        self.max_pooling_2 = nn.MaxPool2d((2,2), stride=(2,2))

        self.conv2d_5 = Conv(128, 256, 128)
        self.conv2d_6 = Conv(256, 256, 128)
        self.conv2d_7 = Conv(256, 256, 128)
        self.max_pooling_3 = nn.MaxPool2d((2,2), stride=(2,2))

        self.conv2d_8 = Conv(256, 512, 64)
        self.conv2d_9 = Conv(512, 512, 64)
        self.conv2d_10 = Conv(512, 512, 64)

        # Deconv / UNet
        self.up_sampling_1 = nn.UpsamplingNearest2d(scale_factor=2)
        # concatenate_1 with conv2d_7, axis = 1

        self.conv2d_11 = Conv(768, 256, 128)
        self.conv2d_12 = Conv(256, 256, 128)
        self.conv2d_13 = Conv(256, 256, 128)

        self.up_sampling_2 = nn.UpsamplingNearest2d(scale_factor=2)
        # concatenate_2 with conv2d_4, axis = 1

        self.conv2d_14 = Conv(384, 128, 256)
        self.conv2d_15 = Conv(128, 128, 256)

        self.up_sampling_3 = nn.UpsamplingNearest2d(scale_factor=2)
        # concatenate_3 with conv2d_2, axis = 1

        self.conv2d_16 = Conv(192, 64, 512)
        self.conv2d_17 = Conv(64, 64, 512)
        self.conv2d_18 = nn.Conv2d(64, 3, kernel_size=(1,1), padding='same')           # 输出3张图
        # self.conv2d_18 = Conv(64, 1, k=(1,1))           输出1张图


    def forward(self, x):
        # VGG16
        x = self.conv2d_1(x)
        x1 = self.conv2d_2(x)
        x = self.max_pooling_1(x1)

        x = self.conv2d_3(x)
        x2 = self.conv2d_4(x)
        x = self.max_pooling_2(x2)

        x = self.conv2d_5(x)
        x = self.conv2d_6(x)
        x3 = self.conv2d_7(x)
        x = self.max_pooling_3(x3)

        x = self.conv2d_8(x)
        x = self.conv2d_9(x)
        x = self.conv2d_10(x)

        # Deconv / UNet
        x = self.up_sampling_1(x)
        x = torch.concat([x, x3], dim=1)

        x = self.conv2d_11(x)
        x = self.conv2d_12(x)
        x = self.conv2d_13(x)

        x = self.up_sampling_2(x)
        x = torch.concat([x, x2], dim=1)

        x = self.conv2d_14(x)
        x = self.conv2d_15(x)

        x = self.up_sampling_3(x)
        x = torch.concat([x, x1], dim=1)

        x = self.conv2d_16(x)
        x = self.conv2d_17(x)
        x = self.conv2d_18(x)

        x = torch.sigmoid(x)

        return x
    

    # torch.load                加载权重
    # model.load_state_dict     将权重加载到模型中

    # model.state_dict()        模型的权重
    # torch.save                保存模型的权重


if __name__ == '__main__':
    model = TrackNet()
    print(summary(model,(9, 288, 512), device="cpu"))
    #print(model)
