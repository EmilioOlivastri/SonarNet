import numpy as np
import matplotlib.pyplot as plt
from torchvision import utils
import logging
import torch
from model import SonarNet

def visTensor(tensor, ch=0, allkernels=False, nrow=8, padding=1): 
    n,c,w,h = tensor.shape

    if allkernels: tensor = tensor.view(n*c, -1, w, h)
    elif c != 3: tensor = tensor[:,ch,:,:].unsqueeze(dim=1)

    rows = np.min((tensor.shape[0] // nrow + 1, 64))    
    grid = utils.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)
    plt.figure( figsize=(nrow,rows) )
    plt.imshow(grid.numpy().transpose((1, 2, 0)))


if __name__ == "__main__":

    net = SonarNet(n_channels=3, n_classes=1, n_angles=18)

    model = "/home/slam-emix/Workspace/Underwater/Saipem/underwater_sim/src/nps_uw_multibeam_sonar/scripts/sonar_net/checkpoints/checkpoint_epoch_best8.pth"
    logging.info(f'Loading model {model}')

    state_dict = torch.load(model)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)

    layer = 1
    filter = net.down2.maxpool_conv[layer].double_conv[3].weight.data.clone()
    visTensor(filter, ch=3, allkernels=False)

    plt.axis('off')
    plt.ioff()
    plt.show()