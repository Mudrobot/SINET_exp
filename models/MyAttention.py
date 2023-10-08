import torch
from torch import nn 
import math

class MyAttention(torch.nn.Module):
    def __init__(self, in_dim, factor=32, t=4):
        super(MyAttention, self).__init__()
        if in_dim == 256:
            in_hw = 64 * 32
        elif in_dim == 512:
            in_hw = 32 * 16
        else:
            in_hw = 16 * 8

        inter_dim = in_dim // factor
        # 通道数量减少
        self.chl_reduction = nn.Sequential(
            nn.Conv3d(in_dim, inter_dim, kernel_size=(1, 1, 1), stride=(1, 1, 1)),
            nn.BatchNorm3d(inter_dim)
        )
        # 通道数量增加
        self.chl_expansion = nn.Sequential(
            nn.Conv3d(inter_dim, in_dim, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        )
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=in_hw*inter_dim, nhead=16)
        # self.attention = nn.MultiheadAttention(embed_dim=in_hw*inter_dim, num_heads=16)
        
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
        """_summary_

        Args:
            x (tensor): 32 * 1024 * 4 * 16 * 8
        """
        x1 = self.chl_reduction(x)
        b, c, t, h, w = x1.size() 
        x2 = x1.reshape(b, c ,t,-1).transpose(1,2).reshape(b,t,-1) # b,t,chw
        x2 = self.transformer_layer(x2)
        x2 = x2.reshape(b,t,c,-1).transpose(1,2).reshape(b,c,t,h,w)
        x2 = x2 + x1
        x3 = self.chl_expansion(x2)
        return x3
        
if __name__ == "__main__":
    input = torch.randn(32, 1024, 4, 16, 8)
    model = MyAttention(1024)
    output = model(input)