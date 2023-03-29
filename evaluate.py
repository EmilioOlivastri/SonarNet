import torch
from tqdm import tqdm


@torch.inference_mode()
def evaluate(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
    score = 0

    criterion_heat = torch.nn.MSELoss()
    criterion_yaw  = torch.nn.CrossEntropyLoss()

    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, true_masks, true_yaws = batch['image'], batch['mask_heat'], batch['yaw_label'] 

        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
        true_masks = true_masks.to(device=device, dtype=torch.float32)
        true_yaws = true_yaws.to(device=device, dtype=torch.float32)

        # Predict mask and angle
        heat_pred, yaw_pred = net(image)

        # Computing the losses
        true_masks = torch.unsqueeze(true_masks, dim=1)
        loss_heat = criterion_heat(heat_pred, true_masks)
        loss_yaw = criterion_yaw(yaw_pred, true_yaws)

        # Add error metrix for this
        score += loss_heat + loss_yaw
        
    net.train()
    return score / max(num_val_batches, 1)
