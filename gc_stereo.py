import torch
import torch.nn as nn
import torch.nn.functional as F
from core.update import BasicSelectiveMultiUpdateBlock, SpatialAttentionExtractor, ChannelAttentionEnhancement
from core.extractor import MultiBasicEncoder, Feature
from core.geometry import Combined_Geo_Encoding_Volume
from core.submodule import *


try:
    autocast = torch.cuda.amp.autocast
except:
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass
        

class hourglass(nn.Module):
    def __init__(self, in_channels):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(BasicConv(in_channels*2, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))
                                    
        self.conv2 = nn.Sequential(BasicConv(in_channels*2, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))                             

        self.conv3 = nn.Sequential(BasicConv(in_channels*4, in_channels*6, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*6, in_channels*6, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1)) 


        self.conv3_up = BasicConv(in_channels*6, in_channels*4, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv2_up = BasicConv(in_channels*4, in_channels*2, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv1_up = BasicConv(in_channels*2, 8, deconv=True, is_3d=True, bn=False,
                                  relu=False, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.agg_0 = nn.Sequential(BasicConv(in_channels*8, in_channels*4, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, kernel_size=3, padding=1, stride=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, kernel_size=3, padding=1, stride=1),)

        self.agg_1 = nn.Sequential(BasicConv(in_channels*4, in_channels*2, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1))



        self.feature_att_8 = FeatureAtt(in_channels*2, 64)
        self.feature_att_16 = FeatureAtt(in_channels*4, 192)
        self.feature_att_32 = FeatureAtt(in_channels*6, 160)
        self.feature_att_up_16 = FeatureAtt(in_channels*4, 192)
        self.feature_att_up_8 = FeatureAtt(in_channels*2, 64)

    def forward(self, x, features, sa):
        conv1 = self.conv1(x)
        conv1 = self.feature_att_8(conv1, features[1])
        conv1 = conv1*sa[1] + conv1
        conv2 = self.conv2(conv1)
        conv2 = self.feature_att_16(conv2, features[2])
        conv2 = conv2*sa[2] + conv2

        conv3 = self.conv3(conv2)
        conv3 = self.feature_att_32(conv3, features[3])

        conv3_up = self.conv3_up(conv3)
        conv2 = torch.cat((conv3_up, conv2), dim=1)
        conv2 = self.agg_0(conv2)
        conv2 = self.feature_att_up_16(conv2, features[2])
        conv2 = conv2*sa[2] + conv2
        conv2_up = self.conv2_up(conv2)
        conv1 = torch.cat((conv2_up, conv1), dim=1)
        conv1 = self.agg_1(conv1)
        conv1 = self.feature_att_up_8(conv1, features[1])
        conv1 = conv1*sa[1] + conv1
        conv = self.conv1_up(conv1)

        return conv


class GEM(nn.Module):
    def __init__(self, feats_num):
        super(GEM, self).__init__()

        # def default_conv(in_channels, out_channels, kernel_size, bias=True):
        # nn.Conv2d(
        # in_channels, out_channels, kernel_size,
        # padding=(kernel_size//2), bias=bias)
        self.grad_rgb = Get_gradient_nopadding_rgb()
        self.conv_grad2 = BasicConv_IN(3, feats_num, deconv=False, is_3d=False, IN=True,
                                  relu=True, kernel_size=3, padding=1, stride=1)
        self.conv_grad2to4 = BasicConv_IN(feats_num, feats_num, deconv=False, is_3d=False, IN=True,
                                  relu=True, kernel_size=3, padding=1, stride=2)
        
        
       
        self.conv_grad4to8 = BasicConv_IN(feats_num, feats_num*2, deconv=False, is_3d=False, IN=True,
                                  relu=True, kernel_size=3, padding=1, stride=2)
        
        self.conv_grad8to16 = BasicConv_IN(feats_num*2, feats_num*3, deconv=False, is_3d=False, IN=True,
                                  relu=True, kernel_size=3, padding=1, stride=2)
                                    
                                    
        self.conv16to8 = BasicConv_IN(feats_num*2, feats_num*2, deconv=True, is_3d=False, IN=True,
                                  relu=True, kernel_size=4, padding=1, stride=2)
        
        self.conv8to4 = BasicConv_IN(feats_num*2, feats_num, deconv=True, is_3d=False, IN=True,
                                  relu=True, kernel_size=4, padding=1, stride=2)
        
        self.convfuse_grad16 = BasicConv_IN(feats_num*3, feats_num*2, deconv=False, is_3d=False, IN=True,
                                  relu=True, kernel_size=3, padding=1, stride=1)
        
        self.convfuse_grad8 = BasicConv_IN(feats_num*4, feats_num*2, deconv=False, is_3d=False, IN=True,
                                  relu=True, kernel_size=3, padding=1, stride=1)
        
        self.convfuse_grad4 = BasicConv_IN(feats_num*2, feats_num, deconv=False, is_3d=False, IN=True,
                                  relu=True, kernel_size=3, padding=1, stride=1)        
        
        self.conv_grad4 = nn.Sequential(BasicConv_IN(feats_num+96, 64, deconv=False, is_3d=False, IN=True,
                                  relu=True, kernel_size=3, padding=1, stride=1),
                                  nn.Conv2d(64, 64, kernel_size=1, padding=0, stride=1)      
                                        )
        
        self.samconv16 = nn.Sequential(nn.Conv2d(2, 1, 7, padding=7//2, bias=False),
                                    nn.Sigmoid())
        
        self.samconv8 = nn.Sequential(nn.Conv2d(2, 1, 7, padding=7//2, bias=False),
                                    nn.Sigmoid())
        
        self.samconv4 = nn.Sequential(nn.Conv2d(2, 1, 7, padding=7//2, bias=False),
                                    nn.Sigmoid())
        self.desc = nn.Conv2d(96, 96, kernel_size=1, padding=0, stride=1)
    def forward(self, image, features):
        
        #梯度特征 空间注意力 加强features
        
        grad_img = self.grad_rgb(image)
        grad_scale2 = F.avg_pool2d(grad_img, kernel_size=2, stride=2)
        
        
        feature_scale2 = self.conv_grad2(grad_scale2)
        feature_scale4 = self.conv_grad2to4(feature_scale2)
        
        feature_scale8 = self.conv_grad4to8(feature_scale4)
        
        feature_scale16 = self.conv_grad8to16(feature_scale8)
        
        spatial_scale16 = self.convfuse_grad16(feature_scale16)
        
        spatial_scale16to8 = self.conv16to8(spatial_scale16)
        spatial_scale8 = self.convfuse_grad8(torch.cat([spatial_scale16to8, feature_scale8], dim = 1))

        spatial_scale8to4 = self.conv8to4(spatial_scale8)
        spatial_scale4 = self.convfuse_grad4(torch.cat([spatial_scale8to4, feature_scale4], dim = 1))
        features = self.desc(features)
        spatial_scale4 = self.conv_grad4(torch.cat([spatial_scale4, features], dim = 1))
    
        
        avg_out = torch.mean(spatial_scale16, dim=1, keepdim=True)
        max_out, _ = torch.max(spatial_scale16, dim=1, keepdim=True)
        spatial_scale16x = torch.cat([avg_out, max_out], dim=1)
        sa_scale16 = self.samconv16(spatial_scale16x)
        sa_scale16 = sa_scale16.unsqueeze(2)
        avg_out = torch.mean(spatial_scale8, dim=1, keepdim=True)
        max_out, _ = torch.max(spatial_scale8, dim=1, keepdim=True)
        spatial_scale8x = torch.cat([avg_out, max_out], dim=1)
        sa_scale8 = self.samconv8(spatial_scale8x)
        sa_scale8 = sa_scale8.unsqueeze(2)
        avg_out = torch.mean(spatial_scale4, dim=1, keepdim=True)
        max_out, _ = torch.max(spatial_scale4, dim=1, keepdim=True)
        spatial_scale4x = torch.cat([avg_out, max_out], dim=1)
        sa_scale4 = self.samconv4(spatial_scale4x)
        sa_scale4 = sa_scale4.unsqueeze(2)
        return spatial_scale4, [sa_scale4, sa_scale8, sa_scale16]



        


class GCStereo(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        context_dims = args.hidden_dims

        self.cnet = MultiBasicEncoder(output_dim=[args.hidden_dims, context_dims], norm_fn="batch", downsample=args.n_downsample)
        self.update_block = BasicSelectiveMultiUpdateBlock(self.args, hidden_dims=args.hidden_dims)
        self.sam = SpatialAttentionExtractor()
        self.cam = ChannelAttentionEnhancement(128)

        self.feature = Feature()

        self.stem_2 = nn.Sequential(
            BasicConv_IN(3, 32, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(32), nn.ReLU()
            )
        self.stem_4 = nn.Sequential(
            BasicConv_IN(32, 48, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(48, 48, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(48), nn.ReLU()
            )

        self.spx = nn.Sequential(nn.ConvTranspose2d(2*32, 9, kernel_size=4, stride=2, padding=1),)
        self.spx_2 = Conv2x_IN(24, 32, True)
        self.spx_4 = nn.Sequential(
            BasicConv_IN(96, 24, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(24, 24, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(24), nn.ReLU()
            )

        self.spx_2_gru = Conv2x(32, 32, True)
        self.spx_gru = nn.Sequential(nn.ConvTranspose2d(2*32, 9, kernel_size=4, stride=2, padding=1),)

        self.conv = BasicConv_IN(96, 96, kernel_size=3, padding=1, stride=1)
        self.desc = nn.Conv2d(96, 96, kernel_size=1, padding=0, stride=1)

        self.corr_stem = BasicConv(8, 8, is_3d=True, kernel_size=3, stride=1, padding=1)
        self.corr_feature_att = FeatureAtt(8, 96)
        self.cost_agg = hourglass(8)
        self.classifier = nn.Conv3d(8, 1, 3, 1, 1, bias=False)
        self.graenhance = GEM(24)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def upsample_disp(self, disp, mask_feat_4, stem_2x):

        with autocast(enabled=self.args.mixed_precision):
            xspx = self.spx_2_gru(mask_feat_4, stem_2x)
            spx_pred = self.spx_gru(xspx)
            spx_pred = F.softmax(spx_pred, 1)
            up_disp = context_upsample(disp*4., spx_pred).unsqueeze(1)

        return up_disp


    def forward(self, image1, image2, iters=12, flow_init=None, test_mode=False):
        """ Estimate disparity between pair of frames """

        image1 = (2 * (image1 / 255.0) - 1.0).contiguous()
        image2 = (2 * (image2 / 255.0) - 1.0).contiguous()
        with autocast(enabled=self.args.mixed_precision):
            features_left = self.feature(image1)
            features_right = self.feature(image2)
            stem_2x = self.stem_2(image1)
            stem_4x = self.stem_4(stem_2x)
            stem_2y = self.stem_2(image2)
            stem_4y = self.stem_4(stem_2y)
            features_left[0] = torch.cat((features_left[0], stem_4x), 1)
            features_right[0] = torch.cat((features_right[0], stem_4y), 1)

            match_left = self.desc(self.conv(features_left[0]))
            match_right = self.desc(self.conv(features_right[0]))
            gwc_volume = build_gwc_volume(match_left, match_right, self.args.max_disp//4, 8)
            gwc_volume = self.corr_stem(gwc_volume)
            
            
            #features_left = self.graenhance(image1, features_left)
            grad_left,sa = self.graenhance(image1,features_left[0])
            grad_right,_ = self.graenhance(image2,features_right[0])
            #？？？？
            gwc_volume = self.corr_feature_att(gwc_volume, match_left)
            gwc_volume = sa[0]*gwc_volume + gwc_volume
            
            grad_volume = build_gwc_volume(grad_left, grad_right, self.args.max_disp//4, 8)
            
            gwc_volume = torch.cat([gwc_volume, grad_volume], dim=1)
            
            
            
            geo_encoding_volume = self.cost_agg(gwc_volume, features_left, sa)



            # Init disp from geometry encoding volume
            scores = self.classifier(geo_encoding_volume).squeeze(1)

            # 显式稳定化：减去最大值（PyTorch 内部虽已实现，但显式操作可增强鲁棒性）
            scores = scores - scores.max(dim=1, keepdim=True).values
            
            # prob = F.softmax(self.classifier(geo_encoding_volume).squeeze(1), dim=1)
            prob = F.softmax(scores, dim=1)
            
            init_disp = disparity_regression(prob, self.args.max_disp//4)
            
            del prob, gwc_volume

            if not test_mode:
                xspx = self.spx_4(features_left[0])
                xspx = self.spx_2(xspx, stem_2x)
                spx_pred = self.spx(xspx)
                spx_pred = F.softmax(spx_pred, 1)

            cnet_list = self.cnet(image1, num_layers=self.args.n_gru_layers)
            net_list = [torch.tanh(x[0]) for x in cnet_list]
            inp_list = [torch.relu(x[1]) for x in cnet_list]
            inp_list = [self.cam(x) * x for x in inp_list]
            att = [self.sam(x) for x in inp_list]

        geo_block = Combined_Geo_Encoding_Volume
        geo_fn = geo_block(match_left.float(), match_right.float(), geo_encoding_volume.float(), radius=self.args.corr_radius, num_levels=self.args.corr_levels)
        b, c, h, w = match_left.shape
        coords = torch.arange(w).float().to(match_left.device).reshape(1,1,w,1).repeat(b, h, 1, 1)
        disp = init_disp
        disp_preds = []

        # GRUs iterations to update disparity
        for itr in range(iters):
            disp = disp.detach()
            geo_feat = geo_fn(disp, coords)
            with autocast(enabled=self.args.mixed_precision):
                net_list, mask_feat_4, delta_disp = self.update_block(net_list, inp_list, geo_feat, disp, att)
            disp = disp + delta_disp
            if test_mode and itr < iters-1:
                continue

            # upsample predictions
            disp_up = self.upsample_disp(disp, mask_feat_4, stem_2x)
            disp_preds.append(disp_up)

        if test_mode:
            return disp_up

        init_disp = context_upsample(init_disp*4., spx_pred.float()).unsqueeze(1)
        return init_disp, disp_preds
