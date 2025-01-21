import torch
import torch.nn as nn
import torch.nn.functional as F

from abc import abstractmethod
from typing import List, Callable, Union, Any, TypeVar, Tuple
from torch import tensor as Tensor

 
class InterpAdvance(nn.Module):
    def __init__(self, scale_factor=None, size=None, mode='bilinear'):
        super(InterpAdvance, self).__init__()
        self.scale_factor = scale_factor
        self.size = size
        self.mode = mode
    
    def forward(self, x, reference):
        if self.size is None and self.scale_factor is not None:
            return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=True)
        elif self.size is not None:
            return F.interpolate(x, size=self.size, mode=self.mode, align_corners=True)
        else:
            return F.interpolate(x, size=reference.shape[2:], mode=self.mode, align_corners=True)

class EuclideanLoss(nn.Module):
    def __init__(self):
        super(EuclideanLoss, self).__init__()
    
    def forward(self, input, target):
        return F.mse_loss(input, target)

class EltwiseAdv(nn.Module):
    def __init__(self, operation='PROD'):
        super(EltwiseAdv, self).__init__()
        self.operation = operation

    def forward(self, x, y):
        if self.operation == 'PROD':
            if x.size(1) % y.size(1) != 0:
                raise ValueError(f"Channels of x ({x.size(1)}) must be a multiple of channels of y ({y.size(1)})")

            factor = x.size(1) // y.size(1)

            y_expanded = y.unsqueeze(2).expand(-1, -1, factor, -1, -1).reshape(x.size())

            return x * y_expanded
        else:
            raise ValueError(f"Unsupported operation: {self.operation}")
        

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss


class DepthAttentionBlock(nn.Module):
    def __init__(self, output_channel=64):
        super(DepthAttentionBlock, self).__init__()
        
        self.atten1_c1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.atten1_c2 = nn.Conv2d(32, 32, kernel_size=3, padding=2, dilation=2)

        self.atten2_c1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.atten2_c2 = nn.Conv2d(64, 64, kernel_size=3, padding=2, dilation=2)

        self.atten3_c1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.atten3_c2 = nn.Conv2d(128, 128, kernel_size=3, padding=2, dilation=2)
        self.atten_weight = nn.Conv2d(128, output_channel, kernel_size=1)
        
        self.softmax1 = nn.Softmax2d()  # Use Softmax over spatial dimensions

    def forward(self, sig_depth_predict):
        # First Attention Block
        atten1_c1 = F.relu(self.atten1_c1(sig_depth_predict))  # shape: (batch, 32, H, W)
        atten1_c2 = F.relu(self.atten1_c2(atten1_c1))  # shape: (batch, 32, H, W)

        # Second Attention Block
        atten2_c1 = F.relu(self.atten2_c1(atten1_c2))  # shape: (batch, 64, H, W)
        atten2_c2 = F.relu(self.atten2_c2(atten2_c1))  # shape: (batch, 64, H, W)

        # Third Attention Block
        atten3_c1 = F.relu(self.atten3_c1(atten2_c2))  # shape: (batch, 128, H, W)
        atten3_c2 = F.relu(self.atten3_c2(atten3_c1))  # shape: (batch, 128, H, W)
        atten_weight = self.softmax1(self.atten_weight(atten3_c2))  # shape: (batch, 64, H, W)

        return atten_weight
    

class BaseVAE(nn.Module):
    
    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size:int, current_device: int, **kwargs) -> Tensor:
        raise NotImplementedError

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass


class CosineSimilarityLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(CosineSimilarityLoss, self).__init__()
        self.reduction = reduction
    
    def forward(self, x1, x2):
        cosine_sim = F.cosine_similarity(x1, x2, dim=-1)
        loss = 1 - cosine_sim
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def compute_ssim(x1, x2):
    x1 = x1.squeeze(0).permute(1, 2, 0).cpu().numpy()
    x2 = x2.squeeze(0).permute(1, 2, 0).cpu().numpy()
    ssim_val = ssim(x1, x2, data_range=255, multichannel=True, win_size=3, channel_axis=2)
    return ssim_val

def calculate_psnr(original, reconstructed):
    # Ensure the images have the same shape
    assert original.shape == reconstructed.shape, "Original and reconstructed images must have the same shape."
    
    # Remove the batch dimension if it exists
    original = original.squeeze(0)
    reconstructed = reconstructed.squeeze(0)
    
    # Calculate Mean Squared Error (MSE)
    mse = torch.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')  # PSNR is infinity if MSE is zero
    
    # Calculate the maximum pixel value of the image
    max_pixel_value = 255.0  # Assuming 8-bit images (0-255)
    
    # Calculate PSNR
    psnr = 20 * torch.log10(max_pixel_value / torch.sqrt(mse))
    return psnr.item()