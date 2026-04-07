import torch
from monai.networks.nets import UNet
from monai.losses import DiceFocalLoss

device = torch.device("cpu") # Use CPU for simplicity

model = UNet(
    spatial_dims=3,
    in_channels=2,
    out_channels=1,
    channels=(32, 64, 128, 256),
    strides=(2, 2, 2),
    num_res_units=2,
    dropout=0.2
).to(device)

loss_func = DiceFocalLoss(
    sigmoid=True,
    gamma=2.0,
    include_background=False,
    lambda_dice=1.5,
    lambda_focal=1.0
)

print("Check completed successfully")
