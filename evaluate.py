import torch
import numpy as np
import cv2
import os
import time

from torchvision import transforms
from Model.Basic import *
from Model.DepthDerain import Derain
from Model.vgg_depth import *
from Dataloader.dataloader import ImageFolder
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm


import triple_transforms


args = {
    'img_size_w': 1920,
    'img_size_h': 1080,
	'crop_size': 256,
    'save_img': True
}
transform = transforms.Compose([
    transforms.ToTensor()
])
triple_transform = triple_transforms.Compose([
    triple_transforms.Resize((args['img_size_h'], args['img_size_w'])),
    #triple_transforms.RandomCrop(args['crop_size']),
    triple_transforms.RandomHorizontallyFlip()
])

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


def compute_ssim(x1, x2):
    x1 = x1.squeeze(0).permute(1, 2, 0).cpu().numpy()
    x2 = x2.squeeze(0).permute(1, 2, 0).cpu().numpy()
    ssim_val = ssim(x1, x2, data_range=255, multichannel=True, win_size=3, channel_axis=2)
    return ssim_val


def resize_image(image, target_size):
    # Resize the image to the target size
    image = F.interpolate(image, size=target_size, mode='bilinear', align_corners=False)
    return image


# load model and calculate psnr
def calculate_psnr_for_model(model_path, model_name):
    # Load the model
    print('----------Loading model and calculate psnr----------')
    model_dict = torch.load(model_path, weights_only=True)
    
    test_set = ImageFolder('/home/disk/ning/DerainDepth/data', is_train=False, transform=transform, target_transform=transform, triple_transform=triple_transform)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=8)
    # Set the model to evaluation
    model =  Derain(1, use_pretrained_depth_weights=True).cuda().eval()
    model.load_state_dict(model_dict['model_state_dict'])

    # Calculate PSNR for each image in the test set
    psnr_values = []
    ssim_values = []
    time_lst = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader, desc="Processing images")):
            inputs, targets, depths = data
            inputs= inputs.cuda()
            targets = targets.cuda()

            # Forward pass
            start_time = time.time()
            print(inputs.shape)
            rain_depth_latent, derain_latent, derain_pred = model(inputs)
            end_time = time.time()
            print(f"Image {i}, Time: {end_time - start_time}")
            time_lst.append(end_time - start_time)
            # # resize
            targets = resize_image(targets, (1024, 2048))
            derain_pred = resize_image(derain_pred, (1024, 2048))
            if args['save_img']:
                # save gt
                targets_img = targets.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255
                targets_img = targets_img.astype(np.uint8)
                targets_img = cv2.cvtColor(targets_img, cv2.COLOR_RGB2BGR)

                save_gt_path = f"./test_gt/"
                if not os.path.exists(save_gt_path):
                    os.makedirs(save_gt_path)
                save_gt_name = save_gt_path + f"{i}.png"
                cv2.imwrite(save_gt_name, targets_img)
                # save derain image
                derain = derain_pred.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255
                derain = derain.astype(np.uint8)
                derain = cv2.cvtColor(derain, cv2.COLOR_RGB2BGR)
                save_path = f"./evaluate/{model_name}/"
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                save_name = save_path + f"{i}.png"
                cv2.imwrite(save_name, derain)
                # print(np.max(derain), np.min(derain))
            # Calculate PSNR
            targets = targets * 255
            derain_pred = derain_pred * 255
            psnr = calculate_psnr(targets, derain_pred)
            psnr_values.append(psnr)

            # Calculate SSIM
            ssim_val = compute_ssim(targets, derain_pred)
            ssim_values.append(ssim_val)

    # Calculate the average PSNR
    average_psnr = np.mean(psnr_values)
    average_ssim = np.mean(ssim_values)
    average_time = np.mean(time_lst)
    return average_psnr, average_ssim, psnr_values, ssim_values, average_time
    

if __name__ == '__main__':
    model_name = 'RainCityscapes'
    model_path = f"./ckpt/{model_name}/model_iter_2999.pth"
    average_psnr, average_ssim, psnr_lst, ssim_lst, average_time = calculate_psnr_for_model(model_path, model_name)
    print(f"Average PSNR: {average_psnr}", f"Average SSIM: {average_ssim}, Max PSNR: {max(psnr_lst)}, Min PSNR: {min(psnr_lst)}, Max SSIM: {max(ssim_lst)}, Min SSIM: {min(ssim_lst)}", f"Average Time: {average_time}")
