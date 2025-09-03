import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from PIL import Image
import numpy as np
import cv2


###########################################################
# 1) Utility: Replace BatchNorm with GroupNorm (same as training)
###########################################################
def replace_bn_with_groupnorm(module: nn.Module, num_groups=32):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            num_features = child.num_features
            new_child = nn.GroupNorm(num_groups=num_groups, num_channels=num_features)
            setattr(module, name, new_child)
        else:
            replace_bn_with_groupnorm(child, num_groups)


###########################################################
# 2) Helper: Make DeepLabV3-ResNet101 with output_stride=8
###########################################################
def make_deeplab_v3_resnet101_os8(num_classes=2):
    """
    Returns a DeepLabV3-ResNet101 model with output stride 8
    and the classifier replaced to have 'num_classes' outputs.
    """
    model = models.segmentation.deeplabv3_resnet101(weights="DEFAULT")

    # Override the classifier
    model.classifier = DeepLabHead(2048, num_classes)

    # Adjust layer3 & layer4 for OS=8
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


###########################################################
# 3) Our ModifiedDeepLabV3 Class
###########################################################
class ModifiedDeepLabV3(nn.Module):
    def __init__(self, num_classes=2, output_stride=8):
        super().__init__()
        if output_stride == 8:
            self.model = make_deeplab_v3_resnet101_os8(num_classes)
        else:
            self.model = models.segmentation.deeplabv3_resnet101(weights="DEFAULT")
            self.model.classifier = DeepLabHead(2048, num_classes)

        # Replace all BatchNorm with GroupNorm
        replace_bn_with_groupnorm(self.model)

    def forward(self, x):
        return self.model(x)


###########################################################
# 4) Load the Saved Weights
###########################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Re-initialize the same model architecture used in training
model = ModifiedDeepLabV3(num_classes=2, output_stride=8).to(device)
# Load the state dict (weights)
state_dict = torch.load("../models/fine_tuned_rooftop_model.pth", map_location=device)
model.load_state_dict(state_dict)

# Switch to evaluation mode (disables dropout, etc.)
model.eval()

print("Model loaded successfully!")


###########################################################
# 5) Define Preprocessing (same as training transforms)
###########################################################
inference_transform = transforms.Compose([
    transforms.Resize((768, 768)),  # same size used in training
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


###########################################################
# 6) Create a function to run inference on a single image
###########################################################
def predict_single_image(image_path):
    # 1) Read & preprocess
    image_pil = Image.open(image_path).convert("RGB")
    input_tensor = inference_transform(image_pil).unsqueeze(0).to(device)

    # 2) Forward pass
    with torch.no_grad():
        output = model(input_tensor)
        # For DeepLab, output is a dict: output["out"] -> [1, num_classes, H, W]
        logits = output["out"][0]  # shape [num_classes, H, W]

    # 3) Argmax over channels
    predicted_mask = torch.argmax(logits, dim=0).cpu().numpy()
    
    return image_pil, predicted_mask


###########################################################
# 7) Visualize or Save the Result
###########################################################
def overlay_mask(pil_img, predicted_mask):
    """
    Overlays the predicted_mask (H×W) on top of the original PIL image.
    Class=1 pixels are painted in green. Adjust as you see fit.
    """
    # Resize predicted_mask to original PIL size
    mask_resized = cv2.resize(
        predicted_mask.astype(np.uint8),
        (pil_img.width, pil_img.height),
        interpolation=cv2.INTER_NEAREST
    )

    # Convert PIL -> CV2
    original_cv2 = np.array(pil_img)

    # Create a color overlay
    color_mask = np.zeros_like(original_cv2, dtype=np.uint8)
    color_mask[mask_resized == 1] = [0, 255, 0]  # (B, G, R) = green

    # Blend
    blended = cv2.addWeighted(original_cv2, 0.7, color_mask, 0.3, 0)
    return blended


if __name__ == "__main__":
    #######################################################
    # Example usage
    #######################################################
    test_image_path = "D:/Rishikesh/BCT/VI/Minor/rooftop-detection/data/satellite_image_rgb.tif"  # change to your test image
    pil_img, mask = predict_single_image(test_image_path)

    # Overlay
    result = overlay_mask(pil_img, mask)

    # Display
    cv2.imshow("Rooftop Segmentation", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Optionally save
    os.makedirs("outputs", exist_ok=True)
    out_path = "outputs/inference_result.png"
    cv2.imwrite(out_path, result)
    print(f"Inference result saved to {out_path}")
