import torch
import torch.nn as nn
import torch.nn.functional as F

from .Basic import *
from .mpvit import *
from .hr_decoder import *

class Derain_transformer(nn.Module):

    def __init__(self):
        super(Derain_transformer, self).__init__()
        self.encoder = mpvit_small()
        self.encoder.num_ch_enc = [64,128,216,288,288]

        self.depth_decoder = DepthDecoder(self.encoder.num_ch_enc, scales=range(4))

        # depth attention
        self.depthattblock1 = DepthAttentionBlock(64)
        self.depthattblock2 = DepthAttentionBlock(128)
        self.depthattblock3 = DepthAttentionBlock(216)

        # Derain decoder (U-Net)
        self.up1 = nn.ConvTranspose2d(216, 128, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 3, kernel_size=3, padding=1)

        # depth attention fusion
        self.fusion = EltwiseAdv(operation='PROD')


    def forward(self, x):
        features = self.encoder(x) # [1/2, 1/4, 1/8, 1/16, 1/32]
        
        depth = self.depth_decoder(features) #[1, 1/2, 1/4, 1/8]

        # depth attention
        depth_att1 = self.depthattblock1(depth[("disp", 1)]) # H/2, W/2
        depth_att2 = self.depthattblock2(depth[("disp", 2)]) # H/4, W/4
        depth_att3 = self.depthattblock3(depth[("disp", 3)]) # H/8, W/8

        # depth attention fusion
        att_feat1 =  self.fusion(features[0], depth_att1) # 64, H/2, W/2
        att_feat2 =  self.fusion(features[1], depth_att2) # 128, H/4, W/4
        att_feat3 =  self.fusion(features[2], depth_att3) # 216, H/8, W/8

        # Derain decoder
        up1 = self.conv1(self.up1(att_feat3)) # H/4, W/4
        up2 = self.conv2(self.up2(torch.cat([up1, att_feat2], 1))) # H/2, W/2
        up3 = self.conv3(self.up3(torch.cat([up2, att_feat1], 1))) # H, W

        residual = self.conv4(up3) # 3, H, W
        out = x + residual
        # out = self.conv4(up3)

        return residual, out, depth
    

    def load_pretrained_model(self, model_path):
        self.encoder.load_state_dict(torch.load(model_path))

        print('Pretrained model loaded from {}'.format(model_path))