import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from utils.data_loading import BasicDataset
from unet.unet_model import UNet
from utils.utils import plot_img_and_mask

device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')

directory = r'/home/data/spleen_blood/data/test/imgs'

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()

def get_output_filenames(in_files):
    return f'{os.path.splitext(in_files)[0]}_OUT.png'


def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    model = './epoch_26_acc_0.90_best_val_acc.pth'

    for fn in os.listdir(directory):
        in_files = os.path.join(directory, fn)
        out_files = get_output_filenames(in_files)
        #print(fn, in_files, out_files)

        net = UNet(n_channels=1, n_classes=5, bilinear=True)
        
        logging.info(f'Loading model {model}')
        logging.info(f'Using device {device}')

        net.to(device=device)
        state_dict = torch.load(model, map_location=device)
        mask_values = state_dict.pop('mask_values', [0, 1])
        net.load_state_dict(state_dict)

        logging.info('Model loaded!')
        
        filename = in_files
        logging.info(f'Predicting image {filename} ...')
        img = Image.open(filename)

        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=0.5,
                           out_threshold=0.5,
                           device=device)
                           
        out_filename = out_files
        result = mask_to_image(mask, mask_values)
        result.save(out_filename)
        logging.info(f'Mask saved to {out_filename}')

        #logging.info(f'Visualizing results for image {filename}, close to continue...')
        #plot_img_and_mask(img, mask)



























