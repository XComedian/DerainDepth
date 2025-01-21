import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

from .vgg_depth import *

class Derain(nn.Module):
    def __init__(self, ngpu, use_pretrained_depth_weights=True):
        super().__init__()
        self.ngpu = ngpu
        self.ndf = 16
        self.ngf = 256
        self.nz = 150
        self.use_pretrained_depth_weights = use_pretrained_depth_weights

        # depth prediction encoder
        self.conv1 = make_encoder_layers(cfg[0], 3)
        self.conv2 = make_encoder_layers(cfg[1], 64)
        self.conv3 = make_encoder_layers(cfg[2], 128)
        self.conv4 = make_encoder_layers(cfg[3], 256)
        self.conv5 = make_encoder_layers(cfg[4], 512)

        # Derain AE model
        ## Encoder
        self.e1 = nn.Conv2d(in_channels = 3, out_channels = self.ndf, kernel_size = 4, stride = 2, padding = 1, bias = False)
        self.dw1 = nn.Conv2d(in_channels=self.ndf, out_channels=self.ndf, kernel_size=1, groups=self.ndf, bias=False)
        self.relu = nn.ReLU(inplace = True)

        self.e2 = nn.Conv2d(in_channels = self.ndf + 64, out_channels = self.ndf*2 + 64, kernel_size = 4, stride = 2, padding = 1, bias = False)
        self.dw2 = nn.Conv2d(in_channels=self.ndf*2 + 64, out_channels=self.ndf*2 + 64, kernel_size=1, groups=self.ndf*2 + 64, bias=False)
        self.norm1 = nn.BatchNorm2d(self.ndf*2 + 64)

        self.e3 = nn.Conv2d(in_channels = self.ndf*2 + 192, out_channels = self.ndf*4 + 192, kernel_size = 4, stride = 2, padding = 1, bias = False)
        self.dw3 = nn.Conv2d(in_channels=self.ndf*4 + 192, out_channels=self.ndf*4 + 192, kernel_size=1, groups=self.ndf*4 + 192, bias=False)
        self.norm2 = nn.BatchNorm2d(self.ndf*4 + 192)

        self.e4 = nn.Conv2d(in_channels = self.ndf*4 + 448, out_channels = self.ndf*8 + 448, kernel_size = 4, stride = 2, padding = 1, bias = False)
        self.dw4 = nn.Conv2d(in_channels=self.ndf*8 + 448, out_channels=self.ndf*8 + 448, kernel_size=1, groups=self.ndf*8 + 448, bias=False)
        self.norm3 = nn.BatchNorm2d(self.ndf*8 + 448)

        self.e5 = nn.Conv2d(in_channels = self.ndf*8 + 960, out_channels = self.ndf*16 + 960, kernel_size = 4, stride = 2, padding = 1, bias = False)
        self.dw5 = nn.Conv2d(in_channels=self.ndf*16 + 960, out_channels=self.ndf*16 + 960, kernel_size=1, groups=self.ndf*16 + 960, bias=False)
        self.norm4 = nn.BatchNorm2d(self.ndf*16 + 960)

        self.e6 = nn.Conv2d(in_channels = self.ndf*16 + 1472, out_channels = self.ndf*32 + 1472, kernel_size = 4, stride = 2, padding = 1, bias = False)
        self.dw6 = nn.Conv2d(in_channels=self.ndf*32 + 1472, out_channels=self.ndf*32 + 1472, kernel_size=1, groups=self.ndf*32 + 1472, bias=False)
        self.norm5 = nn.BatchNorm2d(self.ndf*32 + 1472)

        self.e7 = nn.Conv2d(in_channels = self.ndf*32 + 1472, out_channels = self.nz, kernel_size = 4, stride = 1, padding = 0, bias = False)


        self.sig = nn.Sigmoid()

        ## Decoder
        self.upconv1 = nn.ConvTranspose2d(in_channels = self.nz, out_channels = self.ngf*32, kernel_size = 4, stride = 1, padding = 0, bias = False)
        self.upnorm1 = nn.BatchNorm2d(self.ngf*32)
        self.upconv12 = nn.ConvTranspose2d(in_channels = 10176, out_channels = 8192, kernel_size = 1, stride = 1, padding = 0, bias = False)

        self.upconv2 = nn.ConvTranspose2d(in_channels = 8192, out_channels = self.ngf*16, kernel_size = 4, stride = 2, padding = 1, bias = False)
        self.upnorm2 = nn.BatchNorm2d(self.ngf*16)
        self.upconv22 = nn.ConvTranspose2d(in_channels = 5312, out_channels = 4096, kernel_size = 1, stride = 1, padding = 0, bias = False)

        self.upconv3 = nn.ConvTranspose2d(in_channels = self.ngf*16, out_channels = self.ngf*8, kernel_size = 4, stride = 2, padding = 1, bias = False)
        self.upnorm3 = nn.BatchNorm2d(self.ngf*8)
        self.upconv32 = nn.ConvTranspose2d(in_channels = 2624, out_channels = 2048, kernel_size = 1, stride = 1, padding = 0, bias = False)

        self.upconv4 = nn.ConvTranspose2d(in_channels = self.ngf*8, out_channels = self.ngf*4, kernel_size = 4, stride = 2, padding = 1, bias = False)
        self.upnorm4 = nn.BatchNorm2d(self.ngf*4)
        self.upconv42 = nn.ConvTranspose2d(in_channels = 1280, out_channels = 1024, kernel_size = 1, stride = 1, padding = 0, bias = False)

        self.upconv5 = nn.ConvTranspose2d(in_channels = self.ngf*4, out_channels = self.ngf*2, kernel_size = 4, stride = 2, padding = 1, bias = False)
        self.upnorm5 = nn.BatchNorm2d(self.ngf*2)
        self.upconv52 = nn.ConvTranspose2d(in_channels = 608, out_channels = 512, kernel_size = 1, stride = 1, padding = 0, bias = False)

        self.upconv6 = nn.ConvTranspose2d(in_channels = self.ngf*2, out_channels = self.ngf, kernel_size = 4, stride = 2, padding = 1, bias = False)
        self.upnorm6 = nn.BatchNorm2d(self.ngf)
        self.upconv62 = nn.ConvTranspose2d(in_channels = 272, out_channels = 256, kernel_size = 1, stride = 1, padding = 0, bias = False)

        self.upconv7 = nn.ConvTranspose2d(in_channels = self.ngf, out_channels = 3, kernel_size = 4, stride = 2, padding = 1, bias = False)
        self.outconv = nn.Tanh()

        self.init_weights(use_pretrained_depth_weights)

    
    def forward(self, input):
        x = input
        # depth encoder
        conv1 = self.conv1(x) # 64 x h/2 x w/2
        conv2 = self.conv2(conv1) # 128 x h/4 x w/4
        conv3 = self.conv3(conv2) # 256 x h/8 x w/8
        conv4 = self.conv4(conv3) # 512 x h/16 x w/16
        conv5 = self.conv5(conv4) # 512 x h/32 x w/32

        if self.use_pretrained_depth_weights:
            conv1 = conv1.detach()
            conv2 = conv2.detach()
            conv3 = conv3.detach()
            conv4 = conv4.detach()
            conv5 = conv5.detach()
        
        # derain encoder
        x1 = self.e1(x) 
        # x1 = self.relu(self.dw1(x1))
        xd1 = torch.cat((x1, conv1), 1) # (ndf+64) x h/2 x w/2
        x2 = self.norm1(self.relu(self.dw2(self.e2(xd1)))) # (ndf*2+64) x h/4 x w/4
        # x2 = self.norm1(self.relu(self.e2(xd1)))
        xd2 = torch.cat((x2, conv2), 1) # (ndf*2+192) x h/4 x w/4
        x3 = self.norm2(self.relu(self.dw3(self.e3(xd2)))) # (ndf*4+192) x h/8 x w/8
        # x3 = self.norm2(self.relu(self.e3(xd2)))
        xd3 = torch.cat((x3, conv3), 1) # (ndf*4+448) x h/8 x w/8
        x4 = self.norm3(self.relu(self.dw4(self.e4(xd3)))) # (ndf*8+448) x h/16 x w/16
        # x4 = self.norm3(self.relu(self.e4(xd3)))
        xd4 = torch.cat((x4, conv4), 1) # (ndf*8+960) x h/16 x w/16
        x5 = self.norm4(self.relu(self.dw5(self.e5(xd4)))) # (ndf*16+960) x h/32 x w/32
        # x5 = self.norm4(self.relu(self.e5(xd4)))
        xd5 = torch.cat((x5, conv5), 1) # (ndf*16+1472) x h/32 x w/32
        x6 = self.norm5(self.relu(self.dw6(self.e6(xd5)))) # (ndf*32+1472) x h/64 x w/64
        # x6 = self.norm5(self.relu(self.e6(xd5)))
        x7 = self.sig(self.e7(x6))

        # derain decoder
        xu1 = self.upconv1(x7) 
        xc1 = torch.cat((xu1, x6), 1) # 10176 x h/32 x w/32
        xu12 = self.upconv12(xc1) # 8192 x h/32 x w/32
        xx1 = self.relu(self.upnorm1(xu12)) # 8192 x h/32 x w/32
        
        xu2 = self.upconv2(xx1) # 4096 x h/32 x w/32
        xc3 = torch.cat((xu2, x5), 1) # 5312 x h/32 x w/32
        xu22 = self.upconv22(xc3)
        xx2 = self.relu(self.upnorm2(xu22))
        
        xu3 = self.upconv3(xx2) # 2048 x h/16 x w/16
        xc4 = torch.cat((xu3, x4), 1) # 2624 x h/16 x w/16
        xu32 = self.upconv32(xc4)
        xx3 = self.relu(self.upnorm3(xu32)) # 2048 x h/16 x w/16
        
        xu4 = self.upconv4(xx3) # 1024 x h/8 x w/8
        xc5 = torch.cat((xu4, x3), 1)
        xu42 = self.upconv42(xc5) # 1280x h/8 x w/8
        xx4 = self.relu(self.upnorm4(xu42)) # 1024 x h/8 x w/8

        xu5 = self.upconv5(xx4) # 512 x h/4 x w/4
        xc6 = torch.cat((xu5, x2), 1) # 608 x h/4 x w/4
        xu52 = self.upconv52(xc6)
        xx5 = self.relu(self.upnorm5(xu52)) # 512 x h/4 x w/4
        
        xu6 = self.upconv6(xx5) # 256 x h/2 x w/2
        xc7 = torch.cat((xu6, x1), 1) # 272 x h/2 x w/2
        xu62 = self.upconv62(xc7)
        xx6 = self.relu(self.upnorm6(xu62)) 

        xx7 = self.relu(self.upconv7(xx6))
        
        return conv5, x7, self.outconv(xx7) # 3 x h x w
    

    def init_weights(self, use_pretrained_weights=False):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    torch.nn.init.constant_(m.weight, 1)
                    torch.nn.init.constant_(m.bias, 0)
        if use_pretrained_weights:
            print("loading pretrained weights downloaded from pytorch.org")
            self.load_vgg_params(model_zoo.load_url('https://download.pytorch.org/models/vgg16-397923af.pth'))
        else:
            print("do not load pretrained weights for the monocular model")

    def load_vgg_params(self, params):
        transfer_cfg = {
            "conv1": {0: 0, 2: 2},
            "conv2": {0: 5, 2: 7},
            "conv3": {0: 10, 2: 12, 4: 14},
            "conv4": {0: 17, 2: 19, 4: 21},
            "conv5": {0: 24, 2: 26, 4: 28}
        }

        def load_with_cfg(module, cfg):
            state_dict = {}
            for to_id, from_id in cfg.items():
                state_dict["{}.weight".format(to_id)] = params["features.{}.weight".format(from_id)]
                state_dict["{}.bias".format(to_id)] = params["features.{}.bias".format(from_id)]
            module.load_state_dict(state_dict)

        load_with_cfg(self.conv1, transfer_cfg["conv1"])
        load_with_cfg(self.conv2, transfer_cfg["conv2"])
        load_with_cfg(self.conv3, transfer_cfg["conv3"])
        load_with_cfg(self.conv4, transfer_cfg["conv4"])
        load_with_cfg(self.conv5, transfer_cfg["conv5"])


if __name__ == '__main__':
    model = Derain(1, use_pretrained_depth_weights=True)
    input = torch.rand(1, 3, 256, 256)
    _, latent, out = model(input)
    print(latent.size(), out.size())
  