import torch
import torch.nn as nn
import torch.nn.functional as F
from opt_einsum import contract
from core.utils.basic_layers import Conv2x_IN, BasicConv, BasicConv_IN, Conv2x_IN_C, Conv2x_IN_G
from timm.models.layers import DropPath
from typing import Optional, Callable
from functools import partial
from core.mb import *



from core.submodule import Get_gradient_nopadding_d,disp2disp_gradient_xy,disp2disp_grad_candidates

class DispHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=1):
        super(DispHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, output_dim, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))
    
    
        
class DispHeadIN(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=1):
        super(DispHeadIN, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, output_dim, 3, padding=1)
        self.in1 = nn.InstanceNorm2d(hidden_dim)
        self.in2 = nn.InstanceNorm2d(output_dim)
        self.relu = nn.ReLU(inplace=True)
        self.relu2 = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.in1(x)
        x = self.in2(self.conv2(x))
        x = self.relu2(x)
        return x

class ChannelAttentionEnhancement(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttentionEnhancement, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttentionExtractor(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttentionExtractor, self).__init__()

        self.samconv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.samconv(x)
        return self.sigmoid(x)

class RaftConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=256, kernel_size=3, dilation=1):
        super(RaftConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=(kernel_size+(kernel_size-1)*(dilation-1))//2, dilation=dilation)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=(kernel_size+(kernel_size-1)*(dilation-1))//2, dilation=dilation)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=(kernel_size+(kernel_size-1)*(dilation-1))//2, dilation=dilation)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))

        h = (1-z) * h + z * q
        return h

class RaftSepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=256, kernel_size=5):
        super(RaftSepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1, kernel_size), padding=(0, kernel_size // 2))
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1, kernel_size), padding=(0, kernel_size // 2))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1, kernel_size), padding=(0, kernel_size // 2))

        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (kernel_size, 1), padding=(kernel_size // 2, 0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (kernel_size, 1), padding=(kernel_size // 2, 0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (kernel_size, 1), padding=(kernel_size // 2, 0))


    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))        
        h = (1-z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))       
        h = (1-z) * h + z * q

        return h

class ConvGRU(nn.Module):
    def __init__(self, hidden_dim, input_dim, kernel_size=3):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size//2)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size//2)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size//2)

    def forward(self, h, cz, cr, cq, *x_list):

        x = torch.cat(x_list, dim=1)
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz(hx) + cz)
        r = torch.sigmoid(self.convr(hx) + cr)
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)) + cq)
        h = (1-z) * h + z * q
        return h

class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))

        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))


    def forward(self, h, *x):
        # horizontal
        x = torch.cat(x, dim=1)
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))        
        h = (1-z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))       
        h = (1-z) * h + z * q

        return h

class SelectiveConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=256, small_kernel_size=1, large_kernel_size=3):
        super(SelectiveConvGRU, self).__init__()
        self.small_gru = RaftConvGRU(hidden_dim, input_dim, small_kernel_size)
        self.large_gru = RaftConvGRU(hidden_dim, input_dim, large_kernel_size)

    def forward(self, att, h, *x):
        x = torch.cat(x, dim=1)
        h = self.small_gru(h, x) * att + self.large_gru(h, x) * (1 - att)

        return h

class BasicMotionEncoder(nn.Module):
    def __init__(self, args):
        super(BasicMotionEncoder, self).__init__()
        self.args = args
        cor_planes = args.corr_levels * (2*args.corr_radius + 1) * (8+1)
        self.convc1 = nn.Conv2d(cor_planes, 64, 1, padding=0)
        self.convc2 = nn.Conv2d(64, 64, 3, padding=1)
        self.convd1 = nn.Conv2d(1, 64, 7, padding=3)
        self.convd2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv = nn.Conv2d(64+64, 128-1, 3, padding=1)

    def forward(self, disp, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        disp_ = F.relu(self.convd1(disp))
        disp_ = F.relu(self.convd2(disp_))

        cor_disp = torch.cat([cor, disp_], dim=1)
        out = F.relu(self.conv(cor_disp))
        return torch.cat([out, disp], dim=1)

def pool2x(x):
    return F.avg_pool2d(x, 3, stride=2, padding=1)

def pool4x(x):
    return F.avg_pool2d(x, 5, stride=4, padding=1)

def interp(x, dest):
    interp_args = {'mode': 'bilinear', 'align_corners': True}
    return F.interpolate(x, dest.shape[2:], **interp_args)

class BasicMultiUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dims=[]):
        super().__init__()
        self.args = args
        self.encoder = BasicMotionEncoder(args)
        encoder_output_dim = 128

        self.gru04 = ConvGRU(hidden_dims[2], encoder_output_dim + hidden_dims[1] * (args.n_gru_layers > 1))
        self.gru08 = ConvGRU(hidden_dims[1], hidden_dims[0] * (args.n_gru_layers == 3) + hidden_dims[2])
        self.gru16 = ConvGRU(hidden_dims[0], hidden_dims[1])
        self.disp_head = DispHead(hidden_dims[2], hidden_dim=256, output_dim=1)

        self.mask_feat_4 = nn.Sequential(
            nn.Conv2d(hidden_dims[2], 32, 3, padding=1),
            nn.ReLU(inplace=True))

    def forward(self, net, inp, corr=None, disp=None, iter04=True, iter08=True, iter16=True, update=True):

        if iter16:
            net[2] = self.gru16(net[2], *(inp[2]), pool2x(net[1]))
        if iter08:
            if self.args.n_gru_layers > 2:
                net[1] = self.gru08(net[1], *(inp[1]), pool2x(net[0]), interp(net[2], net[1]))
            else:
                net[1] = self.gru08(net[1], *(inp[1]), pool2x(net[0]))
        if iter04:
            motion_features = self.encoder(disp, corr)
            if self.args.n_gru_layers > 1:
                net[0] = self.gru04(net[0], *(inp[0]), motion_features, interp(net[1], net[0]))
            else:
                net[0] = self.gru04(net[0], *(inp[0]), motion_features)

        if not update:
            return net

        delta_disp = self.disp_head(net[0])
        mask_feat_4 = self.mask_feat_4(net[0])
        return net, mask_feat_4, delta_disp

class BasicSelectiveMultiUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dims):
        super().__init__()
        self.args = args
        self.encoder = BasicMotionEncoder(args)
        encoder_output_dim = 128

        if args.n_gru_layers == 3:
            self.gru16 = SelectiveConvGRU(hidden_dims[0], hidden_dims[0] + hidden_dims[1])
        if args.n_gru_layers >= 2:
            self.gru08 = SelectiveConvGRU(hidden_dims[1], hidden_dims[0] * (args.n_gru_layers == 3) + hidden_dims[1] + hidden_dims[2])
        self.gru04 = SelectiveConvGRU(hidden_dims[2], encoder_output_dim + hidden_dims[1] * (args.n_gru_layers > 1) + hidden_dims[2])
        self.disp_head = DispHead(hidden_dims[2], 256)

        self.mask_feat_4 = nn.Sequential(
            nn.Conv2d(hidden_dims[2], 32, 3, padding=1),
            nn.ReLU(inplace=True))

    def forward(self, net, inp, corr, disp, att):
        if self.args.n_gru_layers == 3:
            net[2] = self.gru16(att[2], net[2], inp[2], pool2x(net[1]))
        if self.args.n_gru_layers >= 2:
            if self.args.n_gru_layers > 2:
                net[1] = self.gru08(att[1], net[1], inp[1], pool2x(net[0]), interp(net[2], net[1]))
            else:
                net[1] = self.gru08(att[1], net[1], inp[1], pool2x(net[0]))
        
        motion_features = self.encoder(disp, corr)
        
        motion_features = torch.cat([inp[0], motion_features], dim=1)
        if self.args.n_gru_layers > 1:
            net[0] = self.gru04(att[0], net[0], motion_features, interp(net[1], net[0]))

        delta_disp = self.disp_head(net[0])

        # scale mask to balence gradients
        mask_feat_4 = .25 * self.mask_feat_4(net[0])
        return net, mask_feat_4, delta_disp


class BasicMotionEncoder3(nn.Module):
    def __init__(self, args):
        super(BasicMotionEncoder3, self).__init__()
        self.args = args
        cor_planes = args.corr_levels * (2*args.corr_radius + 1) * (8+2)
        self.convc1 = nn.Conv2d(cor_planes, 64, 1, padding=0)
        self.convc2 = nn.Conv2d(64, 64, 3, padding=1)
        self.convd1 = nn.Conv2d(1, 64, 7, padding=3)
        self.convd2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv = nn.Conv2d(64+64, 128-1, 3, padding=1)

    def forward(self, disp, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        disp_ = F.relu(self.convd1(disp))
        disp_ = F.relu(self.convd2(disp_))

        cor_disp = torch.cat([cor, disp_], dim=1)
        out = F.relu(self.conv(cor_disp))
        return torch.cat([out, disp], dim=1)



class BasicSelectiveMultiUpdateBlock3(nn.Module):
    def __init__(self, args, hidden_dims):
        super().__init__()
        self.args = args
        self.encoder = BasicMotionEncoder3(args)
        encoder_output_dim = 128

        if args.n_gru_layers == 3:
            self.gru16 = SelectiveConvGRU(hidden_dims[0], hidden_dims[0] + hidden_dims[1])
        if args.n_gru_layers >= 2:
            self.gru08 = SelectiveConvGRU(hidden_dims[1], hidden_dims[0] * (args.n_gru_layers == 3) + hidden_dims[1] + hidden_dims[2])
        self.gru04 = SelectiveConvGRU(hidden_dims[2], encoder_output_dim + hidden_dims[1] * (args.n_gru_layers > 1) + hidden_dims[2])
        self.disp_head = DispHead(hidden_dims[2], 256)

        self.mask_feat_4 = nn.Sequential(
            nn.Conv2d(hidden_dims[2], 32, 3, padding=1),
            nn.ReLU(inplace=True))

    def forward(self, net, inp, corr, disp, att):
        if self.args.n_gru_layers == 3:
            net[2] = self.gru16(att[2], net[2], inp[2], pool2x(net[1]))
        if self.args.n_gru_layers >= 2:
            if self.args.n_gru_layers > 2:
                net[1] = self.gru08(att[1], net[1], inp[1], pool2x(net[0]), interp(net[2], net[1]))
            else:
                net[1] = self.gru08(att[1], net[1], inp[1], pool2x(net[0]))
        
        motion_features = self.encoder(disp, corr)
        
        motion_features = torch.cat([inp[0], motion_features], dim=1)
        if self.args.n_gru_layers > 1:
            net[0] = self.gru04(att[0], net[0], motion_features, interp(net[1], net[0]))

        delta_disp = self.disp_head(net[0])

        # scale mask to balence gradients
        mask_feat_4 = .25 * self.mask_feat_4(net[0])
        return net, mask_feat_4, delta_disp

 
class SelectiveConvGRUFD(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=256, small_kernel_size=1, large_kernel_size=3):
        super(SelectiveConvGRU, self).__init__()
        self.small_gru = RaftConvGRU(hidden_dim, input_dim, small_kernel_size)
        self.large_gru = RaftConvGRU(hidden_dim, input_dim, large_kernel_size)

    def forward(self, att, h, motion_high, motion_low, *x):
        high_x = torch.cat([motion_high] + x, dim=1)
        low_x = torch.cat([motion_low] + x, dim=1)
        h = self.small_gru(h, high_x) * att + self.large_gru(h, low_x) * (1 - att)

        return h
    
    

class BasicSelectiveMultiUpdateBlockFD(nn.Module):
    def __init__(self, args, hidden_dims):
        super().__init__()
        self.args = args
        self.encoder = BasicMotionEncoder(args)
        encoder_output_dim = 128

        if args.n_gru_layers == 3:
            self.gru16 = SelectiveConvGRUFD(hidden_dims[0], hidden_dims[0] + hidden_dims[1])
        if args.n_gru_layers >= 2:
            self.gru08 = SelectiveConvGRUFD(hidden_dims[1], hidden_dims[0] * (args.n_gru_layers == 3) + hidden_dims[1] + hidden_dims[2])
        self.gru04 = SelectiveConvGRUFD(hidden_dims[2], encoder_output_dim + hidden_dims[1] * (args.n_gru_layers > 1) + hidden_dims[2])
        self.disp_head = DispHead(hidden_dims[2], 256)

        self.mask_feat_4 = nn.Sequential(
            nn.Conv2d(hidden_dims[2], 32, 3, padding=1),
            nn.ReLU(inplace=True))

    def forward(self, net, inp_high, inp_low, corr, disp, att):
        if self.args.n_gru_layers == 3:
            net[2] = self.gru16(att[2], net[2], inp_high[2], inp_low[2], pool2x(net[1]))
        if self.args.n_gru_layers >= 2:
            if self.args.n_gru_layers > 2:
                net[1] = self.gru08(att[1], net[1], inp_high[1], inp_low[1], pool2x(net[0]), interp(net[2], net[1]))
            else:
                net[1] = self.gru08(att[1], net[1], inp_high[1], inp_low[1], pool2x(net[0]))
        
        motion_features = self.encoder(disp, corr)
        
        motion_features_high = torch.cat([inp_high[0], motion_features], dim=1)
        motion_features_low = torch.cat([inp_low[0], motion_features], dim=1)
        if self.args.n_gru_layers > 1:
            net[0] = self.gru04(att[0], net[0], motion_features_high, motion_features_low , interp(net[1], net[0]))

        delta_disp = self.disp_head(net[0])

        # scale mask to balence gradients
        mask_feat_4 = .25 * self.mask_feat_4(net[0])
        return net, mask_feat_4, delta_disp   

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction_radio=16):
        super().__init__()
        self.channels = channels
        self.inter_channels = self.channels  // reduction_radio
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(  # 使用1x1卷积代替线性层，可以不用调整tensor的形状
            nn.Conv2d(self.channels, self.inter_channels,
                    kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.inter_channels),
            nn.LeakyReLU(),
            nn.Conv2d(self.inter_channels, self.channels,
                    kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # (b, c, h, w)
        maxout = self.maxpool(x) # (b, c, 1, 1)
        avgout = self.avgpool(x) # (b, c, 1, 1)

        maxout = self.mlp(maxout) # (b, c, 1, 1)
        avgout = self.mlp(avgout) # (b, c, 1, 1)

        attention = self.sigmoid(maxout + avgout) #(b, c, 1, 1)

        return attention


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
 
        padding = 3 if kernel_size == 7 else 1
 
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class multiscaleblock(nn.Module):
    def __init__(self, n_feats, hidden_dims):
        super(multiscaleblock, self).__init__()
        self.conv = BasicConv_IN(n_feats, hidden_dims, kernel_size=3, padding=1, stride=1)
        self.conv1 = nn.Sequential(nn.Conv2d(hidden_dims, hidden_dims,3,1,1,bias=False),nn.InstanceNorm2d(hidden_dims),nn.LeakyReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(hidden_dims, hidden_dims,3,dilation = 2, padding = 2),nn.InstanceNorm2d(hidden_dims),nn.LeakyReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(hidden_dims, hidden_dims,7, groups=hidden_dims, padding = 3),nn.LeakyReLU(),nn.InstanceNorm2d(hidden_dims),nn.Conv2d(hidden_dims, hidden_dims, kernel_size=1, stride=1, padding=0, bias=False), nn.LeakyReLU())
        self.conv4 =  BasicConv_IN(hidden_dims*3, hidden_dims, kernel_size=3, stride=1, padding=1)
        self.sa = SpatialAttention()
        self.conv5 = BasicConv_IN(hidden_dims, n_feats, kernel_size=3, padding=1, stride=1)
    def forward(self,input):
        x = self.conv(input)
        x = torch.cat([self.conv1(x),self.conv2(x),self.conv3(x)], dim = 1)
        x = self.conv4(x)
        x = x * self.sa(x) + x
        x = self.conv5(x) + input
        
        
        return x
        



class DispRefine(nn.Module):
    def __init__(self, in_channel):
        super(DispRefine, self).__init__()
        self.grad_c = nn.Sequential(
            BasicConv_IN(1, 16, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(16, 16, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(16), nn.LeakyReLU()
            )
        self.fuse_process = nn.Sequential(
            BasicConv_IN(35, 32, kernel_size=3, stride=1, padding=1),
            BasicConv_IN(32, 32, kernel_size=1, stride=1, padding=1)
            )
        
        self.final_head = nn.Conv2d(32, 1, 3, 1, 1)
        self.multi = multiscaleblock(32,32)
        
    def forward(self, disp, grad_feat, img_grad):
        disp_grad = Get_gradient_nopadding_d(disp)
        disp_grad_feat = self.grad_c(disp_grad)
        grad_feat = F.interpolate(grad_feat, scale_factor=4, mode='bilinear', align_corners=True)
        grad_feat = torch.cat([grad_feat,img_grad,disp_grad_feat], dim=1)
        x = self.fuse_process(grad_feat)
        x = self.multi(x)
        
        res = self.final_head(x)  # [B, 1, H, W]

        disp = F.relu(disp + res, inplace=True)  # [B, 1, H, W]
        
        return disp
        
        
        

class GradRefine(nn.Module):
    def __init__(self):
        super(DispRefine, self).__init__()
        self.grad_c = nn.Sequential(
            BasicConv_IN(2, 16, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(16, 16, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(16), nn.LeakyReLU()
            )
        self.fuse_process = nn.Sequential(
            BasicConv_IN(46, 64, kernel_size=3, stride=1, padding=1),
            BasicConv_IN(64, 32, kernel_size=3, stride=1, padding=1),
            BasicConv_IN(32, 32, kernel_size=1, stride=1, padding=1)
            )
        
        self.final_head = nn.Sequential(nn.Conv2d(32, 48, 3, 1, 1),
                                           nn.LeackyReLU(inplace=True),
                                           nn.Conv2d(48, 2, 3, 1, 1))
        self.multi = multiscaleblock(32,32)
        self.conv_out = nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU(inplace=True))
        
    def forward(self, disp, grad_feat, img_grad):
        disp_grad = disp2disp_gradient_xy(disp)
        disp_grad_feat = self.grad_c(disp_grad)
        grad_feat = 4*F.interpolate(grad_feat, scale_factor=4, mode='bilinear', align_corners=True)
        grad_cands = disp2disp_grad_candidates(disp, level=2)
        grad_feat = torch.cat([grad_feat,img_grad,disp_grad_feat,grad_cands], dim=1)
        x = self.fuse_process(grad_feat)
        x = self.multi(x)
        context = self.conv_out(x)
        res = self.final_head(x)  # [B, 1, H, W]
        
        disp_grad = disp_grad + res  # [B, 2, H, W]
        
        return disp, context
    
def coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1) 
    

class GradPropagate(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # propagation kernels
        kernel_prop = torch.zeros((5, 1, 3, 3))
        vus = [(0, 1),(1, 0), (1, 1), (1, 2), (2, 1)]
        for i, vu in enumerate(vus):
            v, u = vu
            kernel_prop[i, :, v, u] = 1
        self.kernel_prop = kernel_prop

        # difference kernels
        kernel_diff = torch.zeros((5, 1, 3, 3))
        vus = [(0, 1),(1, 0), (1, 1), (1, 2), (2, 1)]
        kernel_diff[:, :, 1, 1] = 1
        for i, vu in enumerate(vus):
            v, u = vu
            kernel_diff[i, :, v, u] = kernel_diff[i, :, v, u] - 1
        self.kernel_diff = kernel_diff

        self.context_compress = nn.Sequential(
            nn.Conv2d(128 + 64, 96, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, 1, 1)
        )
        self.disp_f_stem = nn.Sequential(nn.Conv2d(27, 32, 1, 1, 0),
                                         nn.LeackyReLU(inplace=True),
                                         nn.Conv2d(32, 32, 1, 1, 0))
        self.w_head = nn.Sequential(nn.Conv2d(64, 48, 3, 1, 1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(48, 9, 1, 1, 0))

    def propagate_disparity(self, disparity_grad, disparity_map):
        """
        Args:
        - disparity_grad: tensor of shape [N, 2, H, W], disparity gradients along x and y axes
        - disparity_map: tensor of shape [N, 1, H, W], initial disparity map

        Returns:
        - propagated_disparities: tensor of shape [N, 9, H, W], disparity candidates in 8-neighborhood
        """
        N, _, H, W = disparity_grad.size()

        # pad
        disparity_grad = F.pad(disparity_grad, pad=(1, 1, 1, 1))
        disparity_map = F.pad(disparity_map, pad=(1, 1, 1, 1), mode='replicate')
        coords = coords_grid(N, H + 2, W + 2).to(disparity_grad.device)  # n,2,h+2,w+2

        # prop
        cat_prop = torch.cat((disparity_map, disparity_grad), dim=1).reshape(-1, 1, H + 2, W + 2)
        cat_prop = F.conv2d(cat_prop.repeat(1, 5, 1, 1), self.kernel_prop.to(disparity_grad.device), padding=0, groups=9).reshape(N, 3, 9, H, W)  # n,3,9,h,w
        disparity_map_prop, disparity_grad_prop = cat_prop[:, :1], cat_prop[:, 1:]

        # diff
        cat_diff = torch.cat((disparity_grad, coords), dim=1).reshape(-1, 1, H + 2, W + 2)  # n,4,h+2,w+2
        cat_diff = F.conv2d(cat_diff.repeat(1, 5, 1, 1), self.kernel_diff.to(disparity_grad.device), padding=0, groups=9).reshape(N, -1, 9, H, W)  # n,4,9,h,w
        grad_diff, coords_diff = cat_diff[:, :2], cat_diff[:, 2:]

        #  propagate
        propagated_disparities = disparity_map_prop + disparity_grad_prop[:, :1] * coords_diff[:, :1] + disparity_grad_prop[:, 1:] * coords_diff[:, 1:]   # n,1,9,h,w
        matrix = grad_diff.reshape(N, -1, H, W).abs()

        return propagated_disparities.squeeze(1), matrix.detach()
    
    def forward(self, disp_grads, disp, context):
        disp = disp.detach()
        disp_candidates, matrix = self.propagate_disparity(disp_grads, disp)  # N, 9, H, W
        disp_f = self.disp_f_stem(torch.cat((disp_candidates.detach(), matrix), dim=1))
        w = self.w_head(torch.concat(disp_f, context))
        w_max = torch.max(w, dim=1, keepdim=True)[0]
        w = torch.softmax(w - w_max, dim=1)
        refined_disparity = torch.sum(w * disp_candidates, dim=1, keepdim=True)
        return refined_disparity
    
    
    
class Fre_Decouple(nn.Module):
    def __init__(self, in_channel):
        super(Fre_Decouple, self).__init__()
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.encoder = mixblock(in_channel)

        
    def forward(self, feats):
        feats1 = self.encoder(feats)
        low_feats = self.avgpool(feats1)
        low_feats2 = F.interpolate(low_feats, size=feats.size()[-2:], mode='bilinear', align_corners=True)
        high_feats = feats1 - low_feats2

        
        return high_feats, low_feats

    
class Feature_Decouple(nn.Module):
    def __init__(self, in_channels):
        super(Feature_Decouple, self).__init__()
        self.Fredecouple0 = Fre_Decouple(in_channels[0])
        self.Fredecouple1 = Fre_Decouple(in_channels[1])
        self.Fredecouple2 = Fre_Decouple(in_channels[2])
        
        
    def forward(self, features):
        features4_high, features4_low = self.Fredecouple0(features[0])
        features8_high, features8_low = self.Fredecouple1(features[1])
        features16_high, features16_low = self.Fredecouple2(features[2])
        
        return [features4_high, features8_high, features16_high], [features4_low, features8_low, features16_low]


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction_radio=16):
        super().__init__()
        self.channels = channels
        self.inter_channels = self.channels  // reduction_radio
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(  # 使用1x1卷积代替线性层，可以不用调整tensor的形状
            nn.Conv2d(self.channels, self.inter_channels,
                    kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.inter_channels),
            nn.LeakyReLU(),
            nn.Conv2d(self.inter_channels, self.channels,
                    kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # (b, c, h, w)
        maxout = self.maxpool(x) # (b, c, 1, 1)
        avgout = self.avgpool(x) # (b, c, 1, 1)

        maxout = self.mlp(maxout) # (b, c, 1, 1)
        avgout = self.mlp(avgout) # (b, c, 1, 1)

        attention = self.sigmoid(maxout + avgout) #(b, c, 1, 1)

        return attention


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
 
        padding = 3 if kernel_size == 7 else 1
 
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

    
class mixblock(nn.Module):
    def __init__(self, n_feats):
        super(mixblock, self).__init__()
        self.conv1=nn.Sequential(nn.Conv2d(n_feats,n_feats,3,1,1,bias=False),nn.InstanceNorm2d(n_feats),nn.LeakyReLU())
        self.conv2=nn.Sequential(nn.Conv2d(n_feats,n_feats,3,1,1,bias=False),nn.InstanceNorm2d(n_feats),nn.LeakyReLU(),nn.Conv2d(n_feats,n_feats,3,1,1,bias=False),nn.InstanceNorm2d(n_feats),nn.LeakyReLU(),nn.Conv2d(n_feats,n_feats,3,1,1,bias=False),nn.InstanceNorm2d(n_feats),nn.LeakyReLU())

        self.alpha=nn.Parameter(torch.ones(1) + 1e-3)
        self.beta=nn.Parameter(torch.ones(1) + 1e-3)
        
    def forward(self,x):
        x = self.alpha*self.conv1(x)+self.beta*self.conv2(x)
        
        return x


class multiscaleblock(nn.Module):
    def __init__(self, n_feats, hidden_dims):
        super(multiscaleblock, self).__init__()
        self.conv = BasicConv_IN(n_feats, hidden_dims, kernel_size=3, padding=1, stride=1)
        self.conv1 = nn.Sequential(nn.Conv2d(hidden_dims, hidden_dims,3,1,1,bias=False),nn.InstanceNorm2d(hidden_dims),nn.LeakyReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(hidden_dims, hidden_dims,3,dilation = 2, padding = 2),nn.InstanceNorm2d(hidden_dims),nn.LeakyReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(hidden_dims, hidden_dims,7, groups=hidden_dims, padding = 3),nn.LeakyReLU(),nn.InstanceNorm2d(hidden_dims),nn.Conv2d(hidden_dims, hidden_dims, kernel_size=1, stride=1, padding=0, bias=False), nn.LeakyReLU())
        self.conv4 =  BasicConv_IN(hidden_dims*3, hidden_dims, kernel_size=3, stride=1, padding=1)
        self.sa = SpatialAttention()
        self.conv5 = BasicConv_IN(hidden_dims, n_feats, kernel_size=3, padding=1, stride=1)
    def forward(self,input):
        x = self.conv(input)
        x = torch.cat([self.conv1(x),self.conv2(x),self.conv3(x)], dim = 1)
        x = self.conv4(x)
        x = x * self.sa(x) + x
        x = self.conv5(x) + input
        
        
        return x
    

    
    
class SKMamba(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        layer: int = 1,
        **kwargs,
    ):
        super(SKMamba, self).__init__()
        factor = 2.0 
        d_model = int(hidden_dim // factor)
        self.down = nn.Linear(hidden_dim, d_model)
        self.up = nn.Linear(d_model, hidden_dim)
        self.ln_1 = norm_layer(d_model)
        self.self_attention = SS2D(d_model=d_model, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)
        self.layer = layer

        
        
        
    def forward(self, x: torch.Tensor):
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()
        input_x = self.down(x)
        input_x = self.ln_1(input_x)
        input_x = self.self_attention(input_x)
        input_x = input_x + self.drop_path(input_x)
        x = self.up(input_x) + x
        x = x.reshape(b, c, h, w)
        return x
    
   
def data_transform(x):
    return 2 * x - 1.0

def inverse_data_transform(x):
    return torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)

    
class FreMamba(nn.Module):
    def __init__(self, dim, LayerNorm_type='WithBias'):
        super(FreMamba, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.illu = Illumination_Estimator(dim, n_fea_in=dim+1, n_fea_out=dim)
        self.Mamba = SKMamba(dim)
        self.dwt = DWT()
        self.iwt = IWT()

    def forward(self, x: torch.Tensor):
        b, c, h, w = x.shape
        x = self.norm1(x)
        x = data_transform(x)
        input_dwt = self.dwt(x)
        input_LL, input_high = input_dwt[:b,...], input_dwt[b:,...]
        input_LL = self.illu(input_LL)
        inpput_LL = self.Mamba(input_LL)
        output = self.iwt(torch.cat([inpput_LL, input_high], dim=0))
        output = inverse_data_transform(output)
        
        x = x + output
        
        return x   
    


class FrequencyFusion(nn.Module):
    def __init__(self, in_channels):
        super(FrequencyFusion, self).__init__()
        self.skmamba1 = FreMamba(in_channels)

        self.convfusion = BasicConv_IN(in_channels*2, in_channels, kernel_size=3, padding=1, stride=1)
        self.ca = ChannelAttention(in_channels)
        self.mutiblock = multiscaleblock(in_channels, 96)
        self.sa = SpatialAttention()

        
    def forward(self, features_high, features_low, grad_sa):
        

        features_low = self.skmamba1(features_low)
        features_low_c = F.interpolate(features_low, size=features_high.size()[-2:], mode='bilinear', align_corners=True)
        
        features_high = features_high * grad_sa
        features = torch.concat([features_high, features_low_c], dim = 1)
        features = self.convfusion(features)

        channela = self.ca(features)
        features1 = self.mutiblock(features)
        spatiala = self.sa(features1)
        features1 = features1 + spatiala * features1
        features_f = features + channela * features1
        
        return features_f  
    
class FrequencyFusionLow(nn.Module):
    def __init__(self, in_channels):
        super(FrequencyFusionLow, self).__init__()

        self.convfusion = BasicConv_IN(in_channels*2, in_channels, kernel_size=3, padding=1, stride=1)

        
    def forward(self, features_high, features_low, grad_sa):
        

        features_low_c = F.interpolate(features_low, size=features_high.size()[-2:], mode='bilinear', align_corners=True)
        features_high = features_high * grad_sa
        features = torch.concat([features_high, features_low_c], dim = 1)
        features_f = self.convfusion(features)

        
        return features_f


class FrequencyEasy(nn.Module):
    def __init__(self, in_channels):
        super(FrequencyEasy, self).__init__()
        
        self.frefusion4 = FrequencyFusion(in_channels[0])
        self.frefusion8 = FrequencyFusion(in_channels[1])
        self.frefusion16 = FrequencyFusionLow(in_channels[2])
        

    def forward(self, features_high, features_low, features_left, grad_sa):
        
        
        features_4x = self.frefusion4(features_high[0], features_low[0], grad_sa[0]) + features_left[0]
        features_8x = self.frefusion8(features_high[1], features_low[1], grad_sa[1]) + features_left[1]
        features_16x = self.frefusion16(features_high[2], features_low[2], grad_sa[2]) + features_left[2]
        
        return [features_4x, features_8x, features_16x, features_left[3]]
    
    

class FrequencyEnhance(nn.Module):
    def __init__(self, in_channels):
        super(FrequencyEnhance, self).__init__()
        self.skmamba1 = SKMamba(in_channels)
        self.skmamba2 = SKMamba(in_channels)
        self.skblock = SKBlock(in_channels, in_channels)
        

    def forward(self, features_low):
        
        
        features_4x = self.skmamba1(features_low[0]) + features_low[0]
        features_8x = self.skmamba2(features_low[1]) + features_low[1]
        features_16x = self.skblock(features_low[2]) + features_low[2]
        
        return [features_4x, features_8x, features_16x]






class DispMonoBin(nn.Module):
    def __init__(self, feats_channels):
        super(DispMonoBin, self).__init__()
        self.enc_feat_chns = (48, 64, 128, 192)
        self.dec_feat_chns = (128, 96, 64, 48) 
        self.enc = nn.ModuleList([
            nn.Sequential(
                BasicConv_IN(3, self.enc_feat_chns[0], is_3d=False, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                BasicConv_IN(self.enc_feat_chns[0], self.enc_feat_chns[0], is_3d=False, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1)),
            nn.Sequential(
                BasicConv_IN(self.enc_feat_chns[0], self.enc_feat_chns[1], is_3d=False, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                BasicConv_IN(self.enc_feat_chns[1], self.enc_feat_chns[1], is_3d=False, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1)),
            nn.Sequential(
                BasicConv_IN(self.enc_feat_chns[1], self.enc_feat_chns[2], is_3d=False, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                BasicConv_IN(self.enc_feat_chns[2], self.enc_feat_chns[2], is_3d=False, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1)),
            nn.Sequential(
                BasicConv_IN(self.enc_feat_chns[2], self.enc_feat_chns[3], is_3d=False, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                BasicConv_IN(self.enc_feat_chns[3], self.enc_feat_chns[3], is_3d=False, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))
        ])
        
        self.skmamba1 = SKMamba(self.enc_feat_chns[3])
        #self.skmamba2 = SKMamba(self.enc_feat_chns[3])
        

        self.dec = nn.ModuleList([
            Conv2x_IN_C(self.enc_feat_chns[3], feats_channels[2], self.dec_feat_chns[0], deconv=True, concat=True),
            Conv2x_IN_C(self.dec_feat_chns[0]+self.enc_feat_chns[2], feats_channels[1], self.dec_feat_chns[1],deconv=True, concat=True),
            Conv2x_IN_C(self.dec_feat_chns[1]+self.enc_feat_chns[1], feats_channels[0], self.dec_feat_chns[2], deconv=True, concat=True),
            BasicConv_IN(self.dec_feat_chns[2]+self.enc_feat_chns[0], self.dec_feat_chns[3], is_3d=False, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1)
            
        
            
        ])

        self.disphead = DispHeadIN(input_dim = 48, output_dim = 48)
        
        
        
    def forward(self, feats, img):
        cv_feats = []
        x = img
        for layer in self.enc:
            x = layer(x)
            cv_feats.append(x)

        cv_feats[-1] = self.skmamba1(cv_feats[-1])
        #cv_feats[-1] = self.skmamba2(cv_feats[-1])
        for i, layer in enumerate(self.dec):
            if i == 0:
                x = cv_feats[-1]
            else:
                x = torch.cat([cv_feats[3-i], x], dim=1)
            if i == 3:
                x = layer(x)
            else:
                x = layer(x, feats[2-i])
            
            
           
        disp_monobin = self.disphead(x) 
        
        
        
        return disp_monobin
    
    
    

class CrossInter(nn.Module):
    # input channel1 input channel2
    def __init__(self, channels1,channels2):
        super(CrossInter, self).__init__()
        
                                    
        self.conv1 = nn.Sequential(
            BasicConv_IN(channels1, channels1, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(channels1, channels1, bias=False, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(channels1)
            )
        
        self.conv2 = nn.Sequential(
            BasicConv_IN(channels2, channels2, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(channels2, channels2, bias=False, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(channels2)
            )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(channels2, channels1, bias=False, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
            )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(channels1, channels2, bias=False, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
            )
        

        

    def forward(self, x1, x2):
        f1 = self.conv1(x1)
        f2 = self.conv2(x2)
        att1 = self.conv3(f2)
        x1out = f1 * att1
        x1out = x1 + x1out
        att2 = self.conv4(x1out)
        x2out = att2 * f2 + x2
        return x2out
    
    
class CrossInterMamba(nn.Module):
    # input channel1 input channel2
    def __init__(self, channels1,channels2):
        super(CrossInterMamba, self).__init__()
        
                                    
        self.conv1 = nn.Sequential(
            BasicConv_IN(channels1, channels1, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(channels1, channels1, bias=False, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(channels1)
            )
        
        self.conv2 = nn.Sequential(
            BasicConv_IN(channels2, channels2, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(channels2, channels2, bias=False, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(channels2)
            )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(channels2*2, channels1, bias=False, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(channels2, channels2,kernel_size=7,stride=1, padding=3,groups=channels2, bias=False),
            nn.InstanceNorm2d(channels1),
            
            nn.Conv2d(channels2, channels2, kernel_size=1,stride=1,padding=0, bias=False),
            nn.InstanceNorm2d(channels2),
            nn.Sigmoid()
            )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(channels2*2, channels1, bias=False, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(channels2, channels2,kernel_size=7,stride=1,padding=3,groups=channels2, bias=False),
            nn.InstanceNorm2d(channels1),
            
            nn.Conv2d(channels2, channels2,kernel_size=1,stride=1,padding=0, bias=False),
            nn.InstanceNorm2d(channels2),
            nn.Sigmoid()
            )
        self.skmamba1 = SKMamba(channels2)
        self.skmamba2 = SKMamba(channels2)

    
    def forward(self, x1, x2):
        f1 = self.conv1(x1)
        f2 = self.conv2(x2)
        f2 = self.skmamba1(f2)
        att1 = self.conv3(torch.cat([f1,f2],dim=1))
        x1out = f1 * att1 + x1
        x1out = self.skmamba2(x1out)
        att2 = self.conv4(torch.cat([x1out, f2],dim=1))
        x2out = att2 * f2 + x2
        return x2out
    
class DispMonoBinF(nn.Module):
    def __init__(self, feats_channels):
        super(DispMonoBinF, self).__init__()
        self.dec_feat_chns = (48,64,96,128) 
        
        

        self.dec = nn.ModuleList([
            
            Conv2x_IN_G(feats_channels[3], feats_channels[3]+feats_channels[2], self.dec_feat_chns[3], deconv=True, concat=True),
            Conv2x_IN_G(self.dec_feat_chns[3], self.dec_feat_chns[3]+feats_channels[1], self.dec_feat_chns[2],deconv=True, concat=True),
            Conv2x_IN_G( self.dec_feat_chns[2], self.dec_feat_chns[2]+feats_channels[0], self.dec_feat_chns[1], deconv=True, concat=True),
            BasicConv_IN(self.dec_feat_chns[1], self.dec_feat_chns[0], is_3d=False, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1)
            
            
        ])

        self.disphead = DispHeadIN(input_dim = 48, output_dim = 48)
        
        
        
    def forward(self, feats):
        x = feats[3]
        for i, layer in enumerate(self.dec):
            
            if i == 3:
                x = layer(x)
            else:
                x = layer(x, feats[2-i])
            
            
            
           
        disp_monobin = self.disphead(x) 
        
        
        
        return disp_monobin
    
    
    
    
def trapped_inter(x):
    B, C, H, W = x.shape
    mask1 = torch.round(torch.abs(torch.sin(x)))
    mask2 = torch.round(torch.abs(torch.cos(x)))
    mask3 = torch.round(torch.abs(2*torch.sin(x)*torch.cos(x)))
    mask4 = torch.round(torch.sin(x)**2)

    x1 = mask1 * x
    x2 = mask2 * x
    x3 = mask3 * x
    x4 = mask4 * x
    x = torch.cat([x1, x3, x2, x4], dim=1)
    x = x.view(B, 2, 2*C, H, W)
    x = x.permute(0, 2, 3, 1, 4).flatten(2).contiguous()
    x = x.view(B, 2*C, H * 2, W)
    x = x.view(B, 2, C, H * 2, W)
    x = x.permute(0, 2, 3, 4, 1).flatten(-1).contiguous()
    x = x.view(B, C, H * 2, W * 2)
    return x



class TA(nn.Module):
    def __init__(
            self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
            ls_init_value=1., trap=True, drop_path=0.
    ):
        super(TA, self).__init__()


        self.dw_conv = nn.Conv2d(in_features, in_features, kernel_size=7, padding=3, groups=in_features)
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=True)
        self.norm1 = nn.InstanceNorm2d(in_features) 
        self.norm2 = nn.InstanceNorm2d(in_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=True)
        self.fc3 = nn.Conv2d(in_features, out_features, kernel_size=1, bias=True)
        self.drop = nn.Dropout(drop)
        self.drop_path = nn.Dropout(drop_path)
        self.gamma2 = nn.Parameter(ls_init_value * torch.ones(out_features)) if ls_init_value > 0 else None

        self.trap = trap
        if self.trap:
            # self.attn_conv = nn.Conv2d(in_features, in_features, kernel_size=3, padding=1, groups=in_features)
            # self.attn_conv1 = nn.Conv2d(in_features, in_features, kernel_size=3, stride=2, padding=1, groups=in_features)
            self.downsample = nn.PixelUnshuffle(2)
            self.attn_conv = nn.Conv2d(in_features*4, in_features, kernel_size=3, padding=1, groups=in_features)
            # self.attn_conv = nn.Conv2d(in_features, in_features, kernel_size=7, padding=3, groups=in_features)
            # self.pool = nn.AvgPool2d(2, 2)
            self.norm1 = nn.InstanceNorm2d(in_features*4)
            self.gamma1 = nn.Parameter(ls_init_value * torch.ones(in_features)) if ls_init_value > 0 else None

    def forward(self, x):
        x = self.dw_conv(x) + x
        if self.trap:
            shortcut1 = x
            x = trapped_inter(self.downsample(x))
            x = self.norm1(x)
            x = self.attn_conv(x)
            x = x.mul(self.gamma1.reshape(1, -1, 1, 1))
            x = self.drop_path(x) + shortcut1

        shortcut2 = x
        x = self.norm2(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop_path(x) + self.fc3(shortcut2)
    
class DispMonoBin(nn.Module):
    def __init__(self, feats_channels):
        super(DispMonoBin, self).__init__()
        self.dec_feat_chns = (48,64,96,128) 
        
        
        self.dec1 = Conv2x_IN_G(feats_channels[3], feats_channels[3]+feats_channels[2], self.dec_feat_chns[3], deconv=True, concat=True),
        self.dec2 = Conv2x_IN_G(self.dec_feat_chns[3], self.dec_feat_chns[3]+feats_channels[1], self.dec_feat_chns[2],deconv=True, concat=True),
        self.dec3 = Conv2x_IN_G( self.dec_feat_chns[2], self.dec_feat_chns[2]+feats_channels[0], self.dec_feat_chns[1], deconv=True, concat=True),
        self.dec4 = BasicConv_IN(self.dec_feat_chns[1], self.dec_feat_chns[0], is_3d=False, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1)


        self.disphead = DispHead(input_dim = 48, output_dim = 48)
        
        
        
    def forward(self, feats):
        x32 = feats[3]
        
        
        x16 = self.dec2(x32, feats[2])
        
        x8 = self.dec3(x16, feats[1])
        x4 = self.dec3(x8, feats[1])
        x4 = self.dec4(x4, feats[1])

           
        disp_monobin = self.disphead(x4) 
        
        
        
        return disp_monobin
    
    
class DispMonoBinTA(nn.Module):
    def __init__(self, feats_channels):
        super(DispMonoBinTA, self).__init__()
        # 96,64,192,160
        self.dec_feat_chns = (48,64,96,128) 
        
        
        self.dec1 = BasicConv_IN(feats_channels[3], feats_channels[2], is_3d=False, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1)
        self.dec2 = TA(in_features=feats_channels[2],hidden_features=self.dec_feat_chns[2], out_features=feats_channels[1])
        self.dec3 = TA(in_features=feats_channels[1],hidden_features=self.dec_feat_chns[1], out_features=feats_channels[0])
        self.dec4 = TA(in_features=feats_channels[0],hidden_features=self.dec_feat_chns[0], out_features=feats_channels[0])
        
        


        self.disphead = DispHeadIN(input_dim = 448, output_dim = 48)
        
        
        
    def forward(self, feats):
        x32 = feats[3]
        
        
        x32 = self.dec1(x32)
        x16 = trapped_inter(x32) + feats[2]
        
        
        x16 = self.dec2(x16)
        x8 = trapped_inter(x16) + feats[1]
        
        x8 = self.dec3(x8)
        x4 = trapped_inter(x8) + feats[0]
        
        x4 = self.dec4(x4)
        
        
        fusion = torch.cat([F.interpolate(x32, scale_factor=8, mode='bilinear', align_corners=False),
                     F.interpolate(x16, scale_factor=4, mode='bilinear', align_corners=False),
                     F.interpolate(x8, scale_factor=2, mode='bilinear', align_corners=False),
                     x4],dim = 1)

           
        disp_monobin = self.disphead(fusion) 
        
        
        
        return disp_monobin
    
    
    
class MonoBinTA(nn.Module):
    def __init__(self, feats_channels):
        super(MonoBinTA, self).__init__()
        # 96,64,192,160
        self.dec_feat_chns = (48,64,96,128) 
        
        
        self.dec1 = BasicConv_IN(feats_channels[3], feats_channels[2], is_3d=False, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1)
        self.dec2 = TA(in_features=feats_channels[2],hidden_features=self.dec_feat_chns[2], out_features=feats_channels[1])
        self.dec3 = TA(in_features=feats_channels[1],hidden_features=self.dec_feat_chns[1], out_features=feats_channels[0])
        self.dec4 = TA(in_features=feats_channels[0],hidden_features=self.dec_feat_chns[0], out_features=feats_channels[0])
        
        
        #self.disphead = DispHeadIN(input_dim = 496, output_dim = 48)
        #self.disphead = DispHeadIN(input_dim = 448, output_dim = 48)
        self.disphead = DispHeadIN(input_dim = feats_channels[0]*4,output_dim = 48)
        
        
        
    def forward(self, feats):
        x32 = feats[3]
        
        
        x32 = self.dec1(x32)
        x16 = trapped_inter(x32) + feats[2]
        
        x16 = self.dec2(x16)
        x8 = trapped_inter(x16) + feats[1]
        
        x8 = self.dec3(x8)
        x4 = trapped_inter(x8) + feats[0]
        
        x4 = self.dec4(x4)
        
        
        fusion = torch.cat([F.interpolate(x32, scale_factor=8, mode='bilinear', align_corners=False),
                     F.interpolate(x16, scale_factor=4, mode='bilinear', align_corners=False),
                     F.interpolate(x8, scale_factor=2, mode='bilinear', align_corners=False),
                     x4],dim = 1)

           
        disp_monobin = self.disphead(fusion) 
        
        
        return disp_monobin