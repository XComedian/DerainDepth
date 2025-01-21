import torch
import numpy as np
import cv2
import os


from PIL import Image
from Model.Basic import *
from Model.DepthDerain import Derain
from Model.vgg_depth import *

from torchvision import transforms

args = {
    'img_size_w': 512,
    'img_size_h': 256,
	'crop_size': 256,
    'save_img': True
}

transform = transforms.Compose([
    transforms.Resize((256, 512)),
    transforms.ToTensor()])

model_name = 'RainCityscapes2'
model_path = f"./ckpt/{model_name}/model_iter_1999.pth"

model_dict = torch.load(model_path, weights_only=True)

image_path = f'./data/real8.jpg'
img = Image.open(image_path).convert('RGB')
original_size = img.size
input_img = transform(img).unsqueeze(0).cuda()

model = Derain(1, use_pretrained_depth_weights=True).cuda().eval()
model.load_state_dict(model_dict['model_state_dict'])

with torch.no_grad():
    rain_depth_latent, derain_latent, derain_pred = model(input_img)

derain = derain_pred.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255
derain = derain.astype(np.uint8)
derain = cv2.cvtColor(derain, cv2.COLOR_RGB2BGR)

derain_resized = cv2.resize(derain, original_size)

save_path = f'./data/'
save_name = save_path + 'predict8.png'
cv2.imwrite(save_name, derain_resized)

print(f"Saved predicted image at {save_name}")