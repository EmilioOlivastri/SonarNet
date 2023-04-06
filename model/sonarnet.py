""" Full assembly of the parts to form the complete network """

from .unet_parts import *
from sklearn.random_projection import GaussianRandomProjection
import numpy as np

class SonarNet(nn.Module):
    def __init__(self, n_channels, n_classes, n_angles):
        super(SonarNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_angles = n_angles
        self.bilinear = False

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if self.bilinear else 1
        self.down4 = (Down(512, 1024 // factor))

        # Head that works for the angle estimation
        dummy = np.empty((1, 512*25*32))
        transformer = GaussianRandomProjection(n_components=300, random_state = 42)
        transformer.fit(dummy)
        self.proj_mat = torch.from_numpy(transformer.components_.astype(np.float32)).cuda()
        self.proj_mat.requires_grad = False

        self.flatten = nn.Flatten(start_dim=1)

        self.yaw_estim = nn.Sequential(
        
        # 1 Linear Layer
        nn.Linear(300, 150),
        nn.ReLU(True),

        # 2 Linear Layer
        nn.Linear(150, 75),
        nn.ReLU(True),

        # 3 Linear Layer
        nn.Linear(75, self.n_angles)) 

        # Head that works for pose estimation using the heatmap
        self.up1 = (Up(1024, 512 // factor, self.bilinear))
        self.up2 = (Up(512, 256 // factor, self.bilinear))
        self.up3 = (Up(256, 128 // factor, self.bilinear))
        self.up4 = (Up(128, 64, self.bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):

        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # FC Net
        y0 = self.flatten(x4)
        y1 = y0.permute(1,0) # featxb
        y2 = torch.matmul(self.proj_mat, y1)
        y3 = y2.permute(1, 0)
        y = self.yaw_estim(y3)
        
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
