import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from PIL import Image
import numpy as np
import rasterio
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
import torch.nn as nn

# ---------------------------------------------------------------------
# 1) Utility: Replace all BatchNorm with GroupNorm
# ---------------------------------------------------------------------
def replace_bn_with_groupnorm(module: nn.Module, num_groups=32):
    """
    Recursively replace all nn.BatchNorm2d with nn.GroupNorm.
    GroupNorm does not require large batch sizes or large spatial dims.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            num_features = child.num_features
            # You can tune num_groups, but 32 is a common default
            new_child = nn.GroupNorm(num_groups=num_groups, num_channels=num_features)
            setattr(module, name, new_child)
        else:
            replace_bn_with_groupnorm(child, num_groups)


# ---------------------------------------------------------------------
# 2) Dataset Definition
# ---------------------------------------------------------------------
class RooftopDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None, mask_transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Read the satellite image (RGB) using rasterio
        with rasterio.open(self.image_paths[idx]) as src:
            image = src.read([1, 2, 3])  # select RGB bands
            # (C, H, W) -> (H, W, C) for PIL
            image = np.moveaxis(image, 0, -1)

        image = Image.fromarray(image.astype('uint8'))

        # Apply transforms on the image
        if self.transform:
            image = self.transform(image)

        # Read the mask
        with rasterio.open(self.mask_paths[idx]) as src:
            mask = src.read(1)

        mask = Image.fromarray(mask.astype('uint8'))

        # Apply mask transforms (e.g., resizing)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        # Convert mask to long tensor
        mask = torch.tensor(np.array(mask), dtype=torch.long)

        return image, mask


# ---------------------------------------------------------------------
# 3) Transforms
# ---------------------------------------------------------------------
image_transform = transforms.Compose([
    # 768x768 is usually large enough to avoid 1x1 in the final layer
    transforms.Resize((768, 768)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

mask_transform = transforms.Compose([
    transforms.Resize((768, 768)),
])

# ---------------------------------------------------------------------
# 4) Dataset & DataLoader
# ---------------------------------------------------------------------
# Change these to **valid** .tif paths on your system
image_paths = [
    r"D:/Rishikesh/BCT/VI/Minor/rooftop-detection/data/satellite_image_rgb.tif"
]
mask_paths = [
    r"D:/Rishikesh/BCT/VI/Minor/rooftop-detection/data/rooftop_mask.tif"
]

dataset = RooftopDataset(
    image_paths=image_paths,
    mask_paths=mask_paths,
    transform=image_transform,
    mask_transform=mask_transform
)

# Batch size = 1 to reproduce the error with BatchNorm,
# but GroupNorm won't fail on 1×1.
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# ---------------------------------------------------------------------
# 5) Properly Setting Output Stride = 8 in DeepLab
# ---------------------------------------------------------------------
def make_deeplab_v3_resnet101_os8(num_classes=2):
    """
    Returns a DeepLabV3-ResNet101 model with output stride 8
    and the classifier replaced to have 'num_classes' outputs.
    """
    model = models.segmentation.deeplabv3_resnet101(weights="DEFAULT")

    # Override the classifier
    model.classifier = DeepLabHead(2048, num_classes)

    # The official approach for output stride 8 modifies BOTH layer3 and layer4.
    # Otherwise, you can still get a very small feature map.
    # Stride in layer3 -> 1 (was 2)
    # Stride in layer4 -> 1 (was 2)
    # Dilation is doubled accordingly
    for n, m in model.backbone.layer3.named_modules():
        if 'conv2' in n:
            m.dilation = 2
            m.padding = 2
        elif 'downsample' in n:
            m.stride = (1, 1)

    for n, m in model.backbone.layer4.named_modules():
        if 'conv2' in n:
            m.dilation = 4
            m.padding = 4
        elif 'downsample' in n:
            m.stride = (1, 1)

    return model

# ---------------------------------------------------------------------
# 6) Modified DeepLab with GroupNorm
# ---------------------------------------------------------------------
class ModifiedDeepLabV3(nn.Module):
    def __init__(self, num_classes=2, output_stride=8):
        super().__init__()
        if output_stride == 8:
            self.model = make_deeplab_v3_resnet101_os8(num_classes)
        else:
            # If you need output_stride=16, just load the default
            self.model = models.segmentation.deeplabv3_resnet101(
                weights="DEFAULT"
            )
            self.model.classifier = DeepLabHead(2048, num_classes)

        # Replace all BatchNorm with GroupNorm
        replace_bn_with_groupnorm(self.model)

    def forward(self, x):
        return self.model(x)

# ---------------------------------------------------------------------
# 7) Initialize Model
# ---------------------------------------------------------------------
model = ModifiedDeepLabV3(num_classes=2, output_stride=8)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ---------------------------------------------------------------------
# 8) Training Setup
# ---------------------------------------------------------------------
optimizer = Adam(model.parameters(), lr=0.001)
criterion = CrossEntropyLoss()

# ---------------------------------------------------------------------
# 9) Training Loop
# ---------------------------------------------------------------------
os.makedirs("../models", exist_ok=True)
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)["out"]  # "out" is the segmentation logit
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

# ---------------------------------------------------------------------
# 10) Save Model
# ---------------------------------------------------------------------
torch.save(model.state_dict(), "D:/Rishikesh/BCT/VI/Minor/rooftop-detection/models/fine_tuned_rooftop_model.pth")
print("Model saved successfully!")
