""" Full assembly of the parts to form the complete network """

from .unet_parts import *


class SonarNet(nn.Module):
    def __init__(self, n_channels, n_classes, n_angles):
        super(SonarNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_angles = n_angles
        self.bilinear = False

        self.inc = (DoubleConv(n_channels, 16))
        self.down1 = (Down(16, 32))
        self.down2 = (Down(32, 64))
        self.down3 = (Down(64, 128))
        factor = 2 if self.bilinear else 1
        self.down4 = (Down(128, 256 // factor))

        # Head that works for the angle estimation
        self.flatten = nn.Flatten(start_dim=1)
        
        self.yaw_estim = nn.Sequential(
        
        # 2 Linear Layer
        nn.Linear(12 * 16 * 256, 16 * 256),
        nn.ReLU(True),

        # 2 Linear Layer
        nn.Linear(16 * 256, 4 * 128),
        nn.ReLU(True),

        # 3 Linear Layer
        nn.Linear(4 * 128, self.n_angles)) 

        # Head that works for pose estimation using the heatmap
        self.up1 = (Up(256, 128 // factor, self.bilinear))
        self.up2 = (Up(128, 64 // factor, self.bilinear))
        self.up3 = (Up(64, 32 // factor, self.bilinear))
        self.up4 = (Up(32, 16, self.bilinear))
        self.outc = (OutConv(16, n_classes))

    def forward(self, x):

        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # FC Net
        y = self.yaw_estim(x5)
        
        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits, y

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)
