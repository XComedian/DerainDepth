import torch
import torch.nn as nn
import torch.nn.functional as F

from .Basic import *


class DAF_LSTM_Net(nn.Module):
    def __init__(self, recurrent_iter=6):
        super(DAF_LSTM_Net, self).__init__()
        self.iter = recurrent_iter
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.pool5a = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

        self.conv1_dsn6 = nn.Conv2d(512, 512, 3, padding=1)

        # depth
        self.conv2_dsn6_d = nn.Conv2d(512, 512, 3, padding=1)
        self.up_conv2_dsn6_d = InterpAdvance()

        self.conv2_dsn5_d = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.up_conv2_dsn5_d = InterpAdvance()

        self.conv2_dsn4_d = nn.Conv2d(1024, 384, kernel_size=3, padding=1)
        self.up_conv2_dsn4_d = InterpAdvance()

        self.conv2_dsn3_d = nn.Conv2d(640, 256, kernel_size=3, padding=1)
        self.depth_predict = nn.Conv2d(256, 1, kernel_size=1)
        self.sig_depth_predict = nn.Sigmoid()

        self.depth_down = InterpAdvance()
        self.loss_depth = EuclideanLoss()

        self.atten1_c1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.atten1_c2 = nn.Conv2d(32, 32, kernel_size=3, padding=2, dilation=2)
        self.up_atten1_c2 = InterpAdvance()
        
        self.atten2_c1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.atten2_c2 = nn.Conv2d(64, 64, kernel_size=3, padding=2, dilation=2)
        self.up_atten2_c2 = InterpAdvance()
        
        self.atten3_c1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.atten3_c2 = nn.Conv2d(128, 128, kernel_size=3, padding=2, dilation=2)
        self.atten_weight = nn.Conv2d(128, 64, kernel_size=1)
        self.softmax1 = nn.Softmax(dim=1)

        # LSTM
        self.conv_i = nn.Conv2d(512, 512, 3, padding=1)
        self.conv_f = nn.Conv2d(512, 512, 3, padding=1)
        self.conv_g = nn.Conv2d(512, 512, 3, padding=1)
        self.conv_o = nn.Conv2d(512, 512, 3, padding=1)

        # att features
        self.conv2_dsn6 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.up_conv2_dsn6 = InterpAdvance()
        
        self.conv2_dsn5 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.up_conv2_dsn5 = InterpAdvance()
        
        self.conv2_dsn4 = nn.Conv2d(1024, 384, kernel_size=3, padding=1)
        self.up_conv2_dsn4 = InterpAdvance()
        
        self.conv2_dsn3 = nn.Conv2d(640, 256, kernel_size=3, padding=1)
        self.up_conv2_dsn3 = InterpAdvance()
        
        self.conv2_dsn2 = nn.Conv2d(384, 160, kernel_size=3, padding=1)
        self.up_conv2_dsn2 = InterpAdvance()
        
        self.conv2_dsn1 = nn.Conv2d(224, 128, kernel_size=3, padding=1)
        
        self.conv1_group_1x1 = nn.Conv2d(128, 256, kernel_size=1)
        self.conv1_group_3x3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv2_group_1x1 = nn.Conv2d(256, 256, kernel_size=1)
        
        self.attentional_conv1 = nn.Conv2d(256, 192, kernel_size=1, groups=64)
        self.attentional_conv2 = nn.Conv2d(192, 192, kernel_size=3, padding=1, groups=64)
        self.attentional_conv3 = nn.Conv2d(192, 3, kernel_size=1)
        
        self.norm_residual = nn.Tanh()
        
        self.loss1 = EuclideanLoss()

    def forward(self, input, depth=None, label=None):
        batch_size, _, height, width = input.size()

        h = torch.zeros(batch_size, 512, height//32, width//32).to(input.device)
        c = torch.zeros(batch_size, 512, height//32, width//32).to(input.device)

        total_depth_loss = 0
        total_ssim_loss = 0
        for i in range(self.iter):

            x11 = F.relu(self.conv1_1(input)) # 64 256 512
            x12 = F.relu(self.conv1_2(x11)) # 64 256 512
            x = self.pool1(x12) # 64 128 256

            x21 = F.relu(self.conv2_1(x)) # 128 128 256
            x22 = F.relu(self.conv2_2(x21)) # 128 128 256
            x = self.pool2(x22) # 128 64 128
            
            x = F.relu(self.conv3_1(x)) # 256 64 128
            x = F.relu(self.conv3_2(x)) # 256 64 128
            x33 = F.relu(self.conv3_3(x)) # 256 64 128
            x = self.pool3(x33) # 256 32 64
            
            x = F.relu(self.conv4_1(x)) # 512 32 64
            x = F.relu(self.conv4_2(x)) # 512 32 64
            x43 = F.relu(self.conv4_3(x)) # 512 32 64
            x = self.pool4(x43) # 512 16 32
            
            x = F.relu(self.conv5_1(x)) # 512 16 32
            x = F.relu(self.conv5_2(x)) # 512 16 32
            x53 = F.relu(self.conv5_3(x)) # 512 16 32
            pool5 = self.pool5(x53)  # 512 8 16
            
            pool5a = self.pool5a(pool5) # 512 8 16
            conv1_dsn6 = F.relu(self.conv1_dsn6(pool5a)) # 512 8 16

            if i == 0:
                # depth updates only in the first iteration
                conv2_dsn6_d = F.relu(self.conv2_dsn6_d(conv1_dsn6)) # 512 8 16
                up_conv2_dsn6_d = self.up_conv2_dsn6_d(conv2_dsn6_d, x53) # 512 16 32

                sum5_d = torch.cat([x53, up_conv2_dsn6_d], dim=1) # 1024 16 32
                conv2_dsn5_d = F.relu(self.conv2_dsn5_d(sum5_d)) # 512 16 32
                up_conv2_dsn5_d = self.up_conv2_dsn5_d(conv2_dsn5_d, x43) # 512 32 64

                sum4_d = torch.cat([x43, up_conv2_dsn5_d], dim=1) # 1024 32 64
                conv2_dsn4_d = F.relu(self.conv2_dsn4_d(sum4_d)) # 384 32 64
                up_conv2_dsn4_d = self.up_conv2_dsn4_d(conv2_dsn4_d, x33) # 384 64 128

                sum3_d = torch.cat([x33, up_conv2_dsn4_d], dim=1) # 640 64 128
                conv2_dsn3_d = F.relu(self.conv2_dsn3_d(sum3_d)) # 256 64 128
                depth_predict = self.depth_predict(conv2_dsn3_d) # 1 64 128
                sig_depth_predict = self.sig_depth_predict(depth_predict) # 1 64 128
                
                atten1_c1 = F.relu(self.atten1_c1(sig_depth_predict)) # 32 64 128
                atten1_c2 = F.relu(self.atten1_c2(atten1_c1)) # 32 64 128
                up_atten1_c2 = self.up_atten1_c2(atten1_c2, x21) # 32 128 256
                
                atten2_c1 = F.relu(self.atten2_c1(up_atten1_c2)) # 64 128 256
                atten2_c2 = F.relu(self.atten2_c2(atten2_c1)) # 64 128 256
                up_atten2_c2 = self.up_atten2_c2(atten2_c2, x11) # 64 256 512

                atten3_c1 = F.relu(self.atten3_c1(up_atten2_c2)) # 128 256 512
                atten3_c2 = F.relu(self.atten3_c2(atten3_c1)) # 128 256 512
                atten_weight = self.softmax1(self.atten_weight(atten3_c2)) # 64 256 512

                if self.training and depth is not None:
                    depth_down = self.depth_down(depth, sig_depth_predict)
                    loss_depth = self.loss_depth(sig_depth_predict, depth_down)
                    total_depth_loss += loss_depth

            # LSTM
            i = self.conv_i(conv1_dsn6)
            f = self.conv_f(conv1_dsn6)
            g = self.conv_g(conv1_dsn6)
            o = self.conv_o(conv1_dsn6)
            c = f * c + i * g
            h = o * F.tanh(c)

            # att features
            conv2_dsn6 = F.relu(self.conv2_dsn6(h)) # 512 8 16
            up_conv2_dsn6 = self.up_conv2_dsn6(conv2_dsn6, x53)  # 512 16 32
            
            sum5 = torch.cat([x53, up_conv2_dsn6], dim=1) 
            conv2_dsn5 = F.relu(self.conv2_dsn5(sum5))
            up_conv2_dsn5 = self.up_conv2_dsn5(conv2_dsn5, x43) # 512 32 64
            
            sum4 = torch.cat([x43, up_conv2_dsn5], dim=1) # 1024 32 64
            conv2_dsn4 = F.relu(self.conv2_dsn4(sum4))
            up_conv2_dsn4 = self.up_conv2_dsn4(conv2_dsn4, x33) # 384 64 128
            
            sum3 = torch.cat([x33, up_conv2_dsn4], dim=1) # 640 64 128
            conv2_dsn3 = F.relu(self.conv2_dsn3(sum3))
            up_conv2_dsn3 = self.up_conv2_dsn3(conv2_dsn3, x22) # 256 128 256
            
            sum2 = torch.cat([x22, up_conv2_dsn3], dim=1) # 384 128 256
            conv2_dsn2 = F.relu(self.conv2_dsn2(sum2)) # 160 128 256    
            up_conv2_dsn2 = self.up_conv2_dsn2(conv2_dsn2, x12) # 160 256 512

            sum1 = torch.cat([x12, up_conv2_dsn2], dim=1) # 224 256 512
            conv2_dsn1 = F.relu(self.conv2_dsn1(sum1)) # 128 256 512
            
            conv1_group_1x1 = F.relu(self.conv1_group_1x1(conv2_dsn1)) # 256 256 512
            conv1_group_3x3 = F.relu(self.conv1_group_3x3(conv1_group_1x1)) # 256 256 512
            conv2_group_1x1 = self.conv2_group_1x1(conv1_group_3x3) # 256 256 512
            
            attentional_feature = EltwiseAdv(operation='PROD')(conv2_group_1x1, atten_weight) # 256 256 512
            
            attentional_conv1 = F.relu(self.attentional_conv1(attentional_feature)) # 192 256 512
            attentional_conv2 = F.relu(self.attentional_conv2(attentional_conv1)) # 192 256 512
            attentional_conv3 = self.attentional_conv3(attentional_conv2) # 3 256 512
            
            norm_residual = self.norm_residual(attentional_conv3)
            output1 = input + norm_residual

            if self.training and label is not None:
                loss1 = self.loss1(output1, label)
                total_ssim_loss += loss1
                
        if self.training and depth is not None and label is not None:
            return output1, sig_depth_predict, norm_residual, total_depth_loss, total_ssim_loss
        
        return output1, sig_depth_predict, norm_residual
        