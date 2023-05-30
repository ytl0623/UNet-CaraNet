import argparse
import logging
import os
import random
import sys
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from evaluate import evaluate
from unet.unet_model import UNet
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss

device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')

PRED_MODEL = './epoch_26_acc_0.90_best_val_acc.pth'

dir_img = Path('./data/test/imgs/')
dir_mask = Path('./data/test/masks/')
#dir_checkpoint = Path('./out_checkpoints/')

def test_model(
        model, device, 
        epochs: int = 1,
        batch_size: int = 1,
        learning_rate: float=0.001,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
    ):
    
    data_transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    try:
        dataset = CarvanaDataset(dir_img, dir_mask, img_scale)
    except (AssertionError, RuntimeError, IndexError):
        dataset = BasicDataset(dir_img, dir_mask, img_scale, data_transform)

    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    test_loader = DataLoader(dataset, shuffle=True, **loader_args)
    optimizer = optim.Adam(model.parameters(),lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)

    test_score = evaluate(model, test_loader, device, amp)
    scheduler.step(test_score)

    logging.info('Test Dice score: {}'.format(test_score))


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--model', '-m', default= PRED_MODEL, metavar='FILE',help="Specify the file in which the model is stored")
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', default = dir_img)
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    #device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    """
    Change here to adapt to your data
    n_channels=3 for RGB images
    n_classes is the number of probabilities you want to get per pixel
    """
    model = UNet(n_channels=1, n_classes=5, bilinear=True)

    #Load pre-trained model
    model.load_state_dict(torch.load(PRED_MODEL, map_location=device))

    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    # if args.load:
    #     state_dict = torch.load(args.load, map_location=device)
    #     del state_dict['mask_values']
    #     model.load_state_dict(state_dict)
    #     logging.info(f'Model loaded from {args.load}')

    model.to(device=device)

    test_model(
        model=model,
        device=device,
        img_scale=args.scale,
        amp=args.amp
    )
