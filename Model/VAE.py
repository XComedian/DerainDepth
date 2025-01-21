import torch
import torch.nn as nn
from diffusers.models import AutoencoderKL


class SupVae(nn.Module):
    def __init__(self):
        super(SupVae, self).__init__()
        self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
        self.project_layer = nn.Linear(4*32*64, 150)

    def forward(self, x):
        latent = self.vae.encode(x).latent_dist.sample() # 1 x 4 x 32 x 64
        latent = latent.reshape(x.size(0), -1)
        x = self.project_layer(latent)
        x = x.reshape(x.size(0), 150, 1, 1)
        return x
    
    def freeze_layers(self):
        # freeze VAE
        for param in self.vae.parameters():
            param.requires_grad = False
        # unfreeze project layer
        for param in self.project_layer.parameters():
            param.requires_grad = True
    

if __name__ == "__main__":
    model = SupVae()
    input = torch.rand(4, 3, 256, 512)
    output = model(input)
    print(output.shape)