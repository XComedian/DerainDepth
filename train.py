import datetime
import os

import torch.optim as optim
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import vgg16

from Model.perceptual import LossNetwork
from Model.DepthDerain import Derain
from Model.vgg_depth import *
from Dataloader.dataloader import ImageFolder
from Model.Basic import *
from Model.VAE import SupVae
import triple_transforms

ckpt_path = './ckpt'
exp_name = 'RainCityscapes2'
args = {
    'iter_num': 3000,
    'train_batch_size': 4,
    'last_iter': 0,
    'lr': 5e-4,
    'lr_decay': 0.9,
    'weight_decay': 0,
    'momentum': 0.9,
    'val_freq': 50000000,
    'img_size_h': 256,
	'img_size_w': 512,
	'crop_size': 256,
    'load_model': False
}

transform = transforms.Compose([
    transforms.ToTensor()
])
triple_transform = triple_transforms.Compose([
    triple_transforms.Resize((args['img_size_h'], args['img_size_w'])),
    #triple_transforms.RandomCrop(args['crop_size']),
    triple_transforms.RandomHorizontallyFlip()
])

train_set = ImageFolder('/home/disk/ning/DAFNet/data', is_train=True, transform=transform, target_transform=transform, triple_transform=triple_transform)
train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], shuffle=True, num_workers=8)
test_set = ImageFolder('/home/disk/ning/DAFNet/data', is_train=False, transform=transform, target_transform=transform, triple_transform=triple_transform)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=8)
CosineSimilarityLoss = CosineSimilarityLoss().cuda()
# --- Define the perceptual loss network --- #
vgg_model = vgg16(pretrained=True).features[:16]
vgg_model = vgg_model.cuda()
for param in vgg_model.parameters():
    param.requires_grad = False

loss_network = LossNetwork(vgg_model).cuda()
loss_network.eval()

writer = SummaryWriter(os.path.join(ckpt_path, exp_name, 'logs'))
log_path = os.path.join(ckpt_path, exp_name, str(datetime.datetime.now()) + '.txt')

def main():
    net = Derain(ngpu=1, use_pretrained_depth_weights=True).cuda().train()
    depth_net = Disp_vgg(use_pretrained_weights=True).cuda().train()
    supvae = SupVae().cuda().train()
    supvae.freeze_layers()

    params_to_train = list(net.parameters()) + list(depth_net.parameters()) + list(filter(lambda p: p.requires_grad, supvae.parameters()))

    optimizer = optim.Adam(params_to_train, lr=args['lr'], weight_decay=args['weight_decay'])

    if args['load_model']:
        checkpoint = torch.load(args['model_path'])
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        args['last_iter'] = checkpoint['iteration']
    
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    if not os.path.exists(os.path.join(ckpt_path, exp_name)):
        os.makedirs(os.path.join(ckpt_path, exp_name))

    open(log_path, 'w').write(str(args) + '\n\n')
    train(net, depth_net, supvae, optimizer)


def train(net, depth_net, supvae, optimizer):
    curr_iter = args['last_iter']
    while curr_iter < args['iter_num']:
        for i, data in enumerate(train_loader):
            inputs, gts, depths = data
            inputs, gts, depths = inputs.cuda(), gts.cuda(), depths.cuda()
            optimizer.zero_grad()

            rain_depth_latent, rain_latent, derain_pred = net(inputs)
            clear_depth_latent, _, _ = net(gts)
            clear_latent = supvae(gts)
            rain_depth_pred = depth_net(inputs)[0]
            clear_depth_pred = depth_net(gts)[0]

            depth_losses = F.mse_loss(rain_depth_pred, depths) + F.mse_loss(clear_depth_pred, depths)
            depth_latent_losses = CosineSimilarityLoss(rain_depth_latent, clear_depth_latent)

            # print(rain_latent.shape, clear_latent.shape)
            derain_latent_losses = CosineSimilarityLoss(rain_latent, clear_latent)
            derain_losses = F.l1_loss(derain_pred, gts)

            perceptual_loss = loss_network(derain_pred, gts)

            loss = 10 * derain_losses + 2 * depth_losses +  0.5*perceptual_loss  + 0.1 * depth_latent_losses + 0.1 * depth_latent_losses
            

            loss.backward()
            optimizer.step()

            # Log scalar data and images every 100 iterations
            if curr_iter % 50 == 0:
                writer.add_scalar('Train/Loss', loss.item(), curr_iter)
                writer.add_scalar('Train/Depth_Loss', depth_losses.item(), curr_iter)
                writer.add_scalar('Train/Depth_latent_Loss', depth_latent_losses.item(), curr_iter)
                writer.add_scalar('Train/Derain_Loss', derain_losses.item(), curr_iter)
                writer.add_scalar('Train/Derain_latent_Loss', derain_latent_losses.item(), curr_iter)
                writer.add_scalar('Train/Perceptual_Loss', perceptual_loss.item(), curr_iter)
                writer.add_image('Train/Inputs', make_grid(inputs[:4].detach().cpu()), curr_iter)
                writer.add_image('Train/Derained_Images', make_grid(derain_pred[:4].detach().cpu()), curr_iter)
                writer.add_image('Train/Rain_Depth_Images', make_grid(rain_depth_pred[:4].detach().cpu()), curr_iter)
                writer.add_image('Train/Clear_Depth_Images', make_grid(clear_depth_pred[:4].detach().cpu()), curr_iter)
            
            if (curr_iter+1) % 1000 == 0:
                save_path = os.path.join(ckpt_path, exp_name, f'model_iter_{curr_iter}.pth')
                torch.save({
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'iteration': curr_iter
                }, save_path)
            
            curr_iter += 1

            # print all loss items
            print(f'Iter: {curr_iter}, Loss: {loss.item()}, Depth_Loss: {depth_losses.item()}, Depth_latent_Loss: {depth_latent_losses.item()}, Derain_Loss: {derain_losses.item()}, Derain_latent_Loss: {derain_latent_losses.item()}', flush=True)

            if curr_iter >= args['iter_num']:
                break


def validate(net, curr_iter, optimizer):
    net.eval()
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, gts, depths = data
            inputs, gts, depths = inputs.cuda(), gts.cuda(), depths.cuda()

            derain_pred, depth_pred, residual, test_depth_loss, test_net_loss = net(inputs, gts, depths)
            loss = test_depth_loss + test_net_loss

            writer.add_scalar('Test/Loss', loss.item(), curr_iter)
            writer.add_scalar('Test/Net_Loss', test_net_loss.item(), curr_iter)
            writer.add_scalar('Test/Depth_Loss', test_depth_loss.item(), curr_iter)
            writer.add_image('Test/Derained_Images', make_grid(derain_pred.detach().cpu()), curr_iter)
            writer.add_image('Test/Depth_Images', make_grid(depth_pred.detach().cpu()), curr_iter)
            writer.add_image('Test/Residual_Images', make_grid(residual.detach().cpu()), curr_iter)

    net.train()

if __name__ == '__main__':
    main()
