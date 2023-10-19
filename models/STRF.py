import os
import sys
import torch
import torch.nn as nn

class FFM(nn.Module):
    def __init__(self, in_dim, ftype, resolution, factor = 16, pool_type = 'max') -> None:
        super(FFM, self).__init__()
        self.ftype = ftype; self.resolution = resolution
        assert ftype in ['spatial', 'temporal'], 'ftype must be spatial or temporal'
        assert pool_type in ['max', 'avg'], 'pool_type must be max or avg'
        inter_dim = in_dim // factor
        self.chl_reduction = nn.Sequential(
            nn.BatchNorm3d(in_dim),
            nn.Conv3d(in_dim, inter_dim, kernel_size=(1, 1, 1), stride=(1, 1, 1)),
            # nn.BatchNorm3d(inter_dim)
        )
        if pool_type == 'max':
            if ftype == 'temporal': self.pool = nn.MaxPool3d(kernel_size=(resolution, 1, 1), stride=(1, 1, 1))
            if ftype == 'spatial':  self.pool = nn.MaxPool3d(kernel_size=(1, resolution, resolution), stride=(1, 1, 1), padding=(0, (resolution-1)//2, (resolution-1)//2))
        else:
            if ftype == 'temporal': self.pool = nn.AvgPool3d(kernel_size=(resolution, 1, 1), stride=(1, 1, 1))
            if ftype == 'spatial':  self.pool = nn.AvgPool3d(kernel_size=(1, resolution, resolution), stride=(1, 1, 1))
        
        
    def forward(self, x):
        b_all, c_all, t_all, h_all, w_all = x.shape
        if self.ftype == 'temporal' and (self.resolution-1)//2 > 0:
            x_pad1 = x[:,:,:(self.resolution-1)//2,:,:]
            x_pad2 = x[:,:,-(self.resolution-1)//2:,:,:]
            xx = torch.cat((x_pad1, x, x_pad2), dim=2)
        else:
            xx = x
        x1 = self.chl_reduction(xx)
        x1 = self.pool(x1)
        b,c,t,h,w = x1.shape
        x1 = x1.reshape(b,c*t,-1)
        x1_T = x1.transpose(1,2)
        C = x1_T @ x1; C = C * 4
        m = torch.softmax(C, dim=1)
        x = x.reshape(b_all, c_all*t_all, -1)
        f_out = x @ m
        f_out = f_out.reshape(b_all,c_all,-1,h_all,w_all)
        return f_out
        

class STRF(nn.Module):
    def __init__(self, in_dim, reso1, reso2 , factor = 16) -> None:
        assert reso1 > reso2, 'resolution 1 should be larger than resolution 2'
        super(STRF,self).__init__()
        inter_dim = in_dim // factor
        self.ffm_s_c = FFM(in_dim, ftype='spatial', resolution=reso1)
        self.ffm_s_f = FFM(in_dim, ftype='spatial', resolution=reso2)
        self.ffm_t_s = FFM(in_dim, ftype='temporal', resolution=reso1)
        self.ffm_t_d = FFM(in_dim, ftype='temporal', resolution=reso2)
       

    def forward(self,x):
        
        x_t_s = self.ffm_t_s(x)
        x_t_d = self.ffm_t_d(x)
        x_t = x_t_s + x_t_d
        
        x_s_c = self.ffm_s_c(x_t)
        x_s_f = self.ffm_s_f(x_t)
        x_s = x_s_c + x_s_f
        return x_s
        

if __name__ == "__main__":
    input = torch.randn(32,1024,4,32,16)# bcthw
    module = STRF(1024, 3, 1)
    module(input)