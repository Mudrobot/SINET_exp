import torch
from torch import nn 
import math

class CTAttention(torch.nn.Module):
    def __init__(self,
                 in_dim,
                 factor=16,
                 split = 4,
                 t=4):
        super(CTAttention, self).__init__()
        self.split = split
        if in_dim == 256:
            in_hw = 64 * 32
        elif in_dim == 512:
            in_hw = 32 * 16
        else:
            in_hw = 16 * 8
        
        inter_dim = in_dim // factor
        inter_hw = in_hw // factor
        # t*h*w 维度减少
        self.hw_reduction = nn.Sequential(
            nn.Conv2d(in_hw, inter_hw,kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(inter_hw)
        )
        # t*h*w 维度增加
        self.hw_expansion = nn.Sequential(
            nn.Conv2d(inter_hw, in_hw, kernel_size=(1, 1), stride=(1, 1))
        )
        # 通道数量减少
        self.chl_reduction = nn.Sequential(
            nn.Conv3d(in_dim, inter_dim, kernel_size=(1, 1, 1), stride=(1, 1, 1)),
            nn.BatchNorm3d(inter_dim)
        )
        # 通道数量增加
        self.chl_expansion = nn.Sequential(
            nn.Conv3d(inter_dim, in_dim, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        )
        self.attention = nn.MultiheadAttention(embed_dim=32, num_heads=4) #  SelfAttention(32, 1)
        
        # init param
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

        self.zero_init(self.chl_expansion)
        
    def zero_init(self, W):
        nn.init.constant_(W[-1].weight.data, 0.0)
        nn.init.constant_(W[-1].bias.data, 0.0)
        
    def forward(self, x):
        
        x1 = self.chl_reduction(x)
        b, c, t, h, w = x1.size() # 32, 32, 4, 32, 16
        x1 = x1.reshape(b, c, t, -1).transpose(1, 3)    # (b, h*w, t, c)
        x2 = self.hw_reduction(x1) # b, hw, t,c 32 32 4 32
        b1, hw, t1, c1 = x2.size()
        x2 = x2.reshape(b, hw ,-1).transpose(1, 2) # (b1, c1*t1, hw)
        x2, _ = self.attention(x2,x2,x2)
        x2 = x2.transpose(1, 2).reshape(b1, hw, t1, c1)
        x3 = self.hw_expansion(x2)
        x3 = x3 + x1
        x3 = x3.transpose(1, 3).reshape(b, c, t, h, w)
        x4 = self.chl_expansion(x3)
        return x4
        
        
if __name__ == '__main__':
    x = torch.randn(32, 512, 4, 32, 16)
    model = CTAttention(in_dim=512)
    model(x)