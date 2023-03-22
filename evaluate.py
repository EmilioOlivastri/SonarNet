import torch
from torch import nn
from tqdm import tqdm


@torch.inference_mode()
def evaluate(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
    score = 0

    loss = torch.nn.MSELoss()

    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, mask_true = batch['image'], batch['mask']

        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
        mask_true = mask_true.to(device=device, dtype=torch.float32)

        # predict the mask
        mask_true = torch.unsqueeze(mask_true, dim=1)
        mask_pred = net(image)
        
        # Add error metrix for this
        score += loss(mask_pred, mask_true)
        
    net.train()
    return score / max(num_val_batches, 1)
