"""
Inverse an image to the latent space of StyleGAN3
"""
import os
from PIL import Image
import os.path as osp
import numpy as np
import pickle as pkl
from tqdm import tqdm
from dnnlib.util import Logger

import torch
import torch.nn as nn
import torch.nn.functional as F
import click
import pyspng

from crtiterions import LPIPS

class Inversor():
    def __init__(self, stylegan3_path, logpath):
        self.device = torch.device('cuda')
        self.load_G(stylegan3_path)
        self.set_criterions()
        self.logger = Logger()


    def process_image(self,img, img2tensor=True):
        """
        @img2tensor
            - False: inverse tensor to array, which can be saved directly by PIL
            - True: convert array to tensor
        """
        if img2tensor:
            with open(img, 'rb') as f:
                if os.path.splitext(img)[1].lower() == '.png':
                    image = pyspng.load(f.read())
                else:
                    image = np.array(Image.open(f))
            image = image/255*2 -1
            image = torch.from_numpy(image).permute(2,0,1).unsqueeze(0).float().to(self.device)
            return image
        else:
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            image = Image.fromarray(img[0].cpu().numpy(), 'RGB')
            return image

    def set_criterions(self,):
        self.MSE_C = nn.MSELoss().to(self.device)
        self.LPIPS_C = LPIPS(net_type='alex').to(self.device).eval()


    def load_G(self, stylegan3_path):
        with open(stylegan3_path, 'rb') as f:
            self.G = pkl.load(f)['G_ema']
            self.G.to(self.device)

    def resize_image(self, image, pixel=256):
        # resized_image = F.adaptive_avg_pool2d(image, (pixel, pixel))
        # resized_image = F.interpolate(image, (pixel, pixel), mode='area')
        resized_image = F.interpolate(image, (pixel, pixel), mode='bicubic')
        return resized_image


    def inverse(self, imgname, total_step, lr, mse_w, lpip_w, truncation_psi, noise_mode, space='z'):
        gt_image = self.process_image(imgname)
        label = torch.zeros([1, self.G.c_dim]).to(self.device)
        assert space in ['z', 'wp']
        code = None
        if space == 'z':
            code = torch.from_numpy(np.random.randn(1, self.G.z_dim)).to(self.device)
        elif space == 'wp':
            code = self.G.mapping.w_avg.clone().detach().unsqueeze(0).unsqueeze(0).repeat([1, self.G.num_ws, 1])
        code.requires_grad = True
        optimizer = torch.optim.Adam([code], lr=lr)
        for step_idx in tqdm(range(total_step), total=total_step):
            if space == 'z':
                inv_image = self.G(code, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
                print(self.G.synthesis.extra_repr())
            elif space == 'wp':
                inv_image = self.G.synthesis(code, noise_mode=noise_mode)
            
            # print(inv_image.shape, gt_image.shape)
            # losses
            mseloss = self.MSE_C(inv_image, gt_image)*mse_w
            lpipsloss = self.LPIPS_C(self.resize_image(inv_image), self.resize_image(gt_image))*lpip_w   # the input to LPIPS should be 256x256 pixels
            loss = mseloss + lpipsloss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step_idx %50 == 0:
                loss_info = 'Step:{} \t MSE:{:0.3f} \t LPIPS:{:0.3f} \n'.format(step_idx, mseloss.item(), lpipsloss.item())
                self.logger.write(loss_info)
        self.logger.close()
        if space == 'z':
            inv_image = self.G(code, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
        elif space == 'wp':
            inv_image = self.G.synthesis(code, noise_mode=noise_mode)
        inv_image = self.process_image(inv_image, img2tensor=False)
        return inv_image

@click.command()
@click.option('--imgname', type=str, help='image name')
@click.option('--out', 'out_path', type=str, default='out', help='save path of inversed image')
@click.option('--stylegan3', 'stylegan3_path', type=str, default='stylegan3-r-afhqv2-512x512.pkl', help='path of stylegan3 pkl file')
@click.option('--steps', 'total_step', type=int, default=1000, help='total optimization step')
@click.option('--lr', type=float, default=1e-1, help='learning rate')
@click.option('--mse_w', type=float, default=1, help='weight of mse loss')
@click.option('--lpip_w', type=float, default=10, help='weight of lpips')
@click.option('--trunc', 'truncation_psi', type=float, default=0.7, help='Truncation psi')
@click.option('--noise_mode', type=click.Choice(['const', 'random', 'none']), default='none',help='noise mode')
@click.option('--space', type=click.Choice(['z', 'wp']), default='wp',help='noise mode') # w space is to be implemented.
def main(imgname, out_path, stylegan3_path, total_step, lr, mse_w, lpip_w, truncation_psi, noise_mode, space):
    logpath = osp.join(out_path, 'log.txt')
    inversor = Inversor(stylegan3_path, logpath)
    inv_image = inversor.inverse(imgname, total_step, lr,  mse_w, lpip_w, truncation_psi, noise_mode, space)
    inv_image.save(osp.join(out_path, imgname.split('/')[-1]))


if __name__ == '__main__':
    main()
