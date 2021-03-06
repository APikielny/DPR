from skimage import filters
import torch

def L1(N, I_t, I_tp, L_s, L_sp):

    # Fix norms
    img_norm = torch.norm(I_t - I_tp)
   #grad_norm = torch.norm(filters.gaussian(I_t) - filters.gaussian(I_tp))
    image_loss = img_norm #+ grad_norm

    light_loss = (L_s - L_sp) ** 2

    loss = (1/N) * image_loss + light_loss
    return loss
