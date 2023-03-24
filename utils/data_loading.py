import numpy as np
import torch
from torchvision.io import read_image
import torchvision.transforms as T
from PIL import Image
from os import listdir
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm
  
class SonarDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        
        self.imgs = listdir(self.images_dir)
        self.labels = listdir(self.mask_dir)
        self.imgs.sort()
        self.labels.sort()
        
        if not self.imgs:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')


    def __len__(self):
        return len(self.imgs)

    @staticmethod
    def preprocess(tensor_img, scale):
        
        transform = T.ToPILImage()
        pil_img = transform(tensor_img)
        w, h = pil_img.size
        #print(f"W = {w} and H = {h}")
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        
        pil_img = pil_img.resize((newW, newH), resample=Image.BICUBIC)
        img = np.asarray(pil_img)

        if ( len(img.shape) == 3 ) :
          img = img / 255.0
          img = img.transpose((2, 0, 1))
          
        return torch.from_numpy(img.astype(float))

    def __getitem__(self, idx):
    
        img_name = self.imgs[idx]
        label_name = self.labels[idx]
        
        assert len(img_name) != 1, f'Either no image or multiple images found for the ID {img_name}: {img_name}'
        assert len(label_name) != 1, f'Either no mask or multiple masks found for the ID {img_name}: {img_name}'
        
        label = read_image(str(self.mask_dir) + '/' + label_name)
        img   = read_image(str(self.images_dir) + '/' + img_name)

        img = self.preprocess(img, self.scale)
        mask = self.preprocess(label, self.scale)

        return {
            'image': img.float().contiguous(),
            'mask' : mask.float().contiguous()
        }
