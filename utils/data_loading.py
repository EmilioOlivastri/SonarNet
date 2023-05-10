import numpy as np
import torch
from torchvision.io import read_image
import torchvision.transforms as T
from PIL import Image
from os import listdir
from pathlib import Path
from torch.utils.data import Dataset
import math
from utils.cfar.CFAR import CFAR

class SonarDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, angle_dir: str, scale: float = 1.0):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        self.angle_dir = Path(angle_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
    
        self.imgs = listdir(self.images_dir)
        self.labels_heat = listdir(self.mask_dir)
        self.labels_yaw = listdir(self.angle_dir)
        self.imgs.sort()
        self.labels_heat.sort()
        self.labels_yaw.sort()
        Ntc = 20
        Ngc = 4
        Pfa = 5e-2
        rank = Ntc // 2
        self.cfar_detector = CFAR(Ntc, Ngc, Pfa, rank)
        
        if not self.imgs:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')


    def __len__(self):
        return len(self.imgs)

    @staticmethod
    def preprocessImg(tensor_img, scale):
        
        transform = T.ToPILImage()
        pil_img = transform(tensor_img)
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        
        pil_img = pil_img.resize((newW, newH), resample=Image.BICUBIC)
        img = np.asarray(pil_img)

        img = img / 255.0
          
        return torch.from_numpy(img.astype(float))
    
    @staticmethod
    def preprocessImgCFAR(cfar_detector, tensor_img, scale):
        
        transform = T.ToPILImage()
        pil_img = transform(tensor_img)
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        
        pil_img = pil_img.resize((newW, newH), resample=Image.BICUBIC)
        img = np.asarray(pil_img, dtype=np.float32)

        #print(f'Img type = {img.dtype}')

        peaks = cfar_detector.detect(img, 'GOCA')
        peaks &= img > 10
        peaks = np.expand_dims(peaks, 0).astype(np.float32)

        peaks = peaks / 255.0
          
        return torch.from_numpy(peaks)
    

    @staticmethod
    def preprocessYaw(angle_txt, discretized_size):
        
        var = np.pi / discretized_size
        const_factor = 1 / ( np.sqrt(2 * np.pi * var))

        with open(angle_txt) as f:
            yaw = float(f.readlines()[0].replace('\n', '')) 
        
        step_cell = ( math.pi * 2 )/ float(discretized_size)

        np_yaw = np.zeros(discretized_size)

        yaw = yaw if yaw >= 0 else math.pi * 2 + yaw

        arr_idx = int( yaw / step_cell ) % discretized_size
        np_yaw[arr_idx] = 1.0
    
        # Angle follows gaussian distribution
        '''
        for idx in range(1, 5) :
            right_idx = (arr_idx + idx) % discretized_size
            delta = arr_idx - idx
            f_delta = step_cell * idx
            left_idx = delta if delta > 0 else (discretized_size + delta) % discretized_size
            np_yaw[right_idx] = np_yaw[left_idx] = const_factor * np.exp( - (f_delta)**2 / (2 * var) ) * 0.3
        '''

        #print(np_yaw)

        return torch.from_numpy(np_yaw.astype(float))

    def __getitem__(self, idx):
    
        img_name = self.imgs[idx]
        label_heat_name = self.labels_heat[idx]
        label_yaw_name = self.labels_yaw[idx]
        
        assert len(img_name) != 1, f'Either no image or multiple images found for the ID {img_name}: {img_name}'
        assert len(label_heat_name) != 1, f'Either no mask or multiple masks found for the ID {img_name}: {img_name}'
        
        heat  = read_image(str(self.mask_dir) + '/' + label_heat_name)
        img   = read_image(str(self.images_dir) + '/' + img_name)
        
        img        = self.preprocessImgCFAR(self.cfar_detector, img, self.scale)
        mask_heat  = self.preprocessImg(heat, self.scale)
        yaw_label  = self.preprocessYaw(str(self.angle_dir) + '/' + label_yaw_name, 36) 


        return {
            'image': img.float().contiguous(),
            'mask_heat' : mask_heat.float().contiguous(),
            'yaw_label' : yaw_label.float().contiguous()
        }
