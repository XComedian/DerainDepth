import torch
import torch.nn as nn
import torch.nn.parallel


nc = 3 # Number of channels in the training images. For color images this is 3

nz = 150 # Size of latent vector (i.e. size of generator input)

ngf = 256 # Size of feature maps in generator

ndf = 16 # Size of feature maps in Encoder


class DerainAE(nn.Module):
    def __init__(self, ngpu):
        super().__init__()
        self.ngpu = ngpu
        
        # Encoder
        # In the encoder, convolutional layers with the Conv2d function are used to extract features from the input image. 
        # Each block in the encoder consists of two convolutional layers followed by a max-pooling layer, with the exception of the last block which does not include a max-pooling layer.
        # -------
        
        self.e1 = nn.Conv2d(in_channels = nc, out_channels = ndf, kernel_size = 4, stride = 2, padding = 1, bias = False)
        self.relu = nn.ReLU(inplace = True)
        # State Size: '(ndf) x 128 x 128'
        
        self.e2 = nn.Conv2d(in_channels = ndf, out_channels = ndf*2, kernel_size = 4, stride = 2, padding = 1, bias = False)
        self.norm1 = nn.BatchNorm2d(ndf*2)
        # State Size: '(ndf*2) x 64 x 64'
        
        self.e3 = nn.Conv2d(in_channels = ndf*2, out_channels = ndf*4, kernel_size = 4, stride = 2, padding = 1, bias = False)
        self.norm2 = nn.BatchNorm2d(ndf*4)
        # State Size: '(ndf*4) x 32 x 32'

        self.e4 = nn.Conv2d(in_channels = ndf*4, out_channels = ndf*8, kernel_size = 4, stride = 2, padding = 1, bias = False)
        self.norm3 = nn.BatchNorm2d(ndf*8)
        # State Size: '(ndf*8) x 16 x 16'

        self.e5 = nn.Conv2d(in_channels = ndf*8, out_channels = ndf*16, kernel_size = 4, stride = 2, padding = 1, bias = False)
        self.norm4 = nn.BatchNorm2d(ndf*16)
        # State Size: '(ndf*16) x 8 x 8'

        self.e6 = nn.Conv2d(in_channels = ndf*16, out_channels = ndf*32, kernel_size = 4, stride = 2, padding = 1, bias = False)
        self.norm5 = nn.BatchNorm2d(ndf*32)
        # State Size: '(ndf*32) x 4 x 4'

        self.e7 = nn.Conv2d(in_channels = ndf*32, out_channels = nz, kernel_size = 4, stride = 1, padding = 0, bias = False)
        self.sig = nn.Sigmoid()
        # State Size: '100'


        # Decoder
        self.upconv1 = nn.ConvTranspose2d(in_channels = nz, out_channels = ngf*32, kernel_size = 4, stride = 1, padding = 0, bias = False)
        self.upnorm1 = nn.BatchNorm2d(ngf*32)
        self.upconv12 = nn.ConvTranspose2d(in_channels = 8704, out_channels = 8192, kernel_size = 1, stride = 1, padding = 0, bias = False)
        # State Size: '(ngf*32) x 4 x 4'

        self.upconv2 = nn.ConvTranspose2d(in_channels = ngf*32, out_channels = ngf*16, kernel_size = 4, stride = 2, padding = 1, bias = False)
        self.upnorm2 = nn.BatchNorm2d(ngf*16)
        self.upconv22 = nn.ConvTranspose2d(in_channels = 4352, out_channels = 4096, kernel_size = 1, stride = 1, padding = 0, bias = False)
        # State Size: '(ngf*16) x 8 x 8'

        self.upconv3 = nn.ConvTranspose2d(in_channels = ngf*16, out_channels = ngf*8, kernel_size = 4, stride = 2, padding = 1, bias = False)
        self.upnorm3 = nn.BatchNorm2d(ngf*8)
        self.upconv32 = nn.ConvTranspose2d(in_channels = 2176, out_channels = 2048, kernel_size = 1, stride = 1, padding = 0, bias = False)
        # State Size: '(ngf*8) x 16 x 16'

        self.upconv4 = nn.ConvTranspose2d(in_channels = ngf*8, out_channels = ngf*4, kernel_size = 4, stride = 2, padding = 1, bias = False)
        self.upnorm4 = nn.BatchNorm2d(ngf*4)
        self.upconv42 = nn.ConvTranspose2d(in_channels = 1088, out_channels = 1024, kernel_size = 1, stride = 1, padding = 0, bias = False)
        # State Size: '(ngf*4) x 32 x 32'
                                          
        self.upconv5 = nn.ConvTranspose2d(in_channels = ngf*4, out_channels = ngf*2, kernel_size = 4, stride = 2, padding = 1, bias = False)
        self.upnorm5 = nn.BatchNorm2d(ngf*2)
        self.upconv52 = nn.ConvTranspose2d(in_channels = 544, out_channels = 512, kernel_size = 1, stride = 1, padding = 0, bias = False)
        # State Size: '(ngf*2) x 64 x 64'
        
        self.upconv6 = nn.ConvTranspose2d(in_channels = ngf*2, out_channels = ngf, kernel_size = 4, stride = 2, padding = 1, bias = False)
        self.upnorm6 = nn.BatchNorm2d(ngf)
        self.upconv62 = nn.ConvTranspose2d(in_channels = 272, out_channels = 256, kernel_size = 1, stride = 1, padding = 0, bias = False)
        # State Size: '(ngf) x 128 x 128'                                          
        
        self.upconv7 = nn.ConvTranspose2d(in_channels = ngf, out_channels = nc, kernel_size = 4, stride = 2, padding = 1, bias = False)
        self.outconv = nn.Tanh() # nn.Tanh() is used to return the input data range to [-1,1]
        # State Size: '(nc) x 256 x 256' <-- This is the original image size with RGB channels
        
    def forward(self, x):
        ### Encoder ###
        xe1 = self.relu(self.e1(x))
        print(f'xe1: {xe1.size()}')

        xe2 = self.e2(xe1)
        #print(f'xe2: {xe2.size()}')
        xnorm1 = self.relu(self.norm1(xe2))
        print(f'xnorm1: {xnorm1.size()}')
        
        xe3 = self.e3(xnorm1)
        #print(f'xe3: {xe3.size()}')
        xnorm2 = self.relu(self.norm2(xe3))
        print(f'xnorm2: {xnorm2.size()}')
        
        xe4 = self.e4(xnorm2)
        #print(f'xe4: {xe4.size()}')
        xnorm3 = self.relu(self.norm3(xe4))
        print(f'xnorm3: {xnorm3.size()}')
        
        xe5 = self.e5(xnorm3)
        #print(f'xe5: {xe5.size()}')
        xnorm4 = self.relu(self.norm4(xe5))
        print(f'xnorm4: {xnorm4.size()}')

        xe6 = self.e6(xnorm4)
        # print(f'xe6: {xe6.size()}')
        xnorm5 = self.relu(self.norm5(xe6))
        print(f'xnorm5: {xnorm5.size()}')
        
        xe7 = self.e7(xnorm5)
        # print(f'xe7: {xe7.size()}')
        xsig = self.sig(xe7)
        # print(f'xsig: {xsig.size()}')
        
        
        ### Decoder ###
        xu1 = self.upconv1(xsig)
        print(f'xu1: {xu1.size()}')
        xc1 = torch.cat([xu1, xnorm5], dim=1) # Adds the 1st Dimension of xu1 and xnorm together. 8192+512=8704.
        #print(f'xc1: {xc1.size()}')
        xu12 = self.upconv12(xc1)
        print(f'xu12: {xu12.size()}')
        xd1 = self.relu(self.upnorm1(xu12))
        #print(f'xd1: {xd1.size()}')
        
        xu2 = self.upconv2(xd1)
        print(f'xu2: {xu2.size()}')
        xc2 = torch.cat([xu2, xnorm4], dim=1)
        #print(f'xc2: {xc2.size()}')
        xu22 = self.upconv22(xc2)
        print(f'xu22: {xu22.size()}')
        xd2 = self.relu(self.upnorm2(xu22))
        #print(f'xd2: {xd2.size()}\n')

        xu3 = self.upconv3(xd2)
        xc3 = torch.cat([xu3, xnorm3], dim=1)
        xu32 = self.upconv32(xc3)
        xd3 = self.relu(self.upnorm3(xu32))
        
        xu4 = self.upconv4(xd3)
        xc4 = torch.cat([xu4, xnorm2], dim=1)
        xu42 = self.upconv42(xc4)
        xd4 = self.relu(self.upnorm4(xu42))        
        
        xu5 = self.upconv5(xd4)
        #print(f'xu5: {xu5.size()}')
        xc5 = torch.cat([xu5, xnorm1], dim=1)
        #print(f'xc5: {xc5.size()}')
        xu52 = self.upconv52(xc5)
        #print(f'xu52: {xu52.size()}')
        xd5 = self.relu(self.upnorm5(xu52))
        #print(f'xd5: {xd5.size()}\n')

        xu6 = self.upconv6(xd5)
        #print(f'xu6: {xu6.size()}')
        xc6 = torch.cat([xu6, xe1], dim=1)
        #print(f'xc6: {xc6.size()}')
        xu62 = self.upconv62(xc6)
        #print(f'xu62: {xu62.size()}')
        xd6 = self.relu(self.upnorm6(xu62))
        #print(f'xd6: {xd6.size()}')
        
        xd7 = self.relu(self.upconv7(xd6))      
        
        # Output layer
        out = self.outconv(xd7)

        return out
    

if __name__ == '__main__':
    model = DerainAE(1)
    x = torch.randn(1, 3, 256, 256)
    output = model(x)