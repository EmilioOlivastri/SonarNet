import argparse
import logging
import os

import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from utils.data_loading import SonarDataset
from model import UNet

def predict_img(net,
                full_img,
                device,
                scale_factor=1):
    net.eval()
    img = SonarDataset.preprocess(full_img, scale_factor)
    img = img.unsqueeze(0)
    #print(f"Printed Shape In Predict = {img.shape}")
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(output, (full_img.shape[1], full_img.shape[2]), mode='bilinear')
        
    return output.long().squeeze().numpy()


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=1, help='Number of classes')
    
    return parser.parse_args()


def plot_img_and_mask(img, mask):
    fig, ax = plt.subplots(1, 2)

    #print(img.shape)
    print(mask.dtype)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    ax[1].set_title('Prediction image')
    ax[1].imshow(mask)

    plt.xticks([]), plt.yticks([])
    plt.show()


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    in_files = args.input
    net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)

    logging.info('Model loaded!')
    
    transform = transforms.Compose([
    transforms.PILToTensor()])

    for i, filename in enumerate(in_files):
        logging.info(f'Predicting image {filename} ...')
        img = Image.open(filename)
        img_t = transform(img)
        print(img_t.shape)
        mask = predict_img(net=net,
                           full_img=img_t,
                           scale_factor=args.scale,
                           device=device)
        if args.viz:
            logging.info(f'Visualizing results for image {filename}, close to continue...')
            plot_img_and_mask(img, mask)
            
            
