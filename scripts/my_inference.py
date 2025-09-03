import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from PIL import Image
import cv2
import numpy as np


###############################################
# 1) Replace BatchNorm with GroupNorm (same as training)
###############################################
def replace_bn_with_groupnorm(module: nn.Module, num_groups=32):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            new_child = nn.GroupNorm(num_groups=num_groups, num_channels=child.num_features)
            setattr(module, name, new_child)
        else:
            replace_bn_with_groupnorm(child, num_groups)


###############################################
# 2) Create Deeplabv3-ResNet101 (output_stride=8)
###############################################
def make_deeplab_v3_resnet101_os8(num_classes=2):
    model = models.segmentation.deeplabv3_resnet101(weights="DEFAULT")
    model.classifier = DeepLabHead(2048, num_classes)

    # Set layer3 & layer4 to stride=1 for OS=8
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


###############################################
# 3) ModifiedDeepLabV3 class
###############################################
class ModifiedDeepLabV3(nn.Module):
    def __init__(self, num_classes=2, output_stride=8):
        super().__init__()
        if output_stride == 8:
            self.model = make_deeplab_v3_resnet101_os8(num_classes)
        else:
            self.model = models.segmentation.deeplabv3_resnet101(weights="DEFAULT")
            self.model.classifier = DeepLabHead(2048, num_classes)
        replace_bn_with_groupnorm(self.model)

    def forward(self, x):
        return self.model(x)


###############################################
# 4) Load the model + weights you saved
###############################################
def load_trained_model(weights_path="../models/fine_tuned_rooftop_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModifiedDeepLabV3(num_classes=2, output_stride=8).to(device)

    # Load the weights (state_dict)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, device


###############################################
# 5) Preprocessing (same as training)
###############################################
inference_transform = transforms.Compose([
    transforms.Resize((768, 768)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


###############################################
# 6) Predict function
###############################################
def predict_mask(model, device, image_path):
    """
    Reads a PNG image, preprocesses it, and returns the predicted mask.
    """
    # Load the image (PNG is fine!)
    pil_image = Image.open(image_path).convert("RGB")

    # Transform
    input_tensor = inference_transform(pil_image).unsqueeze(0).to(device)

    # Forward pass
    with torch.no_grad():
        output = model(input_tensor)["out"][0]  # shape [num_classes, H, W]

    # Argmax => each pixel is assigned a class ID
    predicted_mask = torch.argmax(output, dim=0).cpu().numpy()
    return pil_image, predicted_mask


###############################################
# 7) Overlay function (optional for visualization)
###############################################
def overlay_mask(pil_image, predicted_mask):
    """
    Overlays the segmentation mask on the original PIL image.
    Class=1 -> green overlay.
    """
    # Resize mask to original image size (if transforms changed dimensions)
    mask_resized = cv2.resize(
        predicted_mask.astype(np.uint8),
        (pil_image.width, pil_image.height),
        interpolation=cv2.INTER_NEAREST
    )

    original_cv2 = np.array(pil_image)
    color_mask = np.zeros_like(original_cv2, dtype=np.uint8)
    color_mask[mask_resized == 1] = [0, 255, 0]  # BGR = green
    blended = cv2.addWeighted(original_cv2, 0.7, color_mask, 0.3, 0)
    return blended


################################################
# Example usage on a PNG file
################################################
if __name__ == "__main__":
    # 1) Load the trained model
    model, device = load_trained_model("../models/fine_tuned_rooftop_model.pth")

    # 2) Provide your PNG image path
    #    (change "my_input.png" to your actual filename)
    test_image_path = "image.png"

    # 3) Inference
    pil_img, mask = predict_mask(model, device, test_image_path)

    # 4) Overlay
    result_overlay = overlay_mask(pil_img, mask)

    # 5) Show or save
    cv2.imshow("PNG Inference Result", result_overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Optionally save the overlay
    os.makedirs("outputs", exist_ok=True)
    out_path = "outputs/png_inference_result.png"
    cv2.imwrite(out_path, result_overlay)
    print(f"Saved overlay to {out_path}")
