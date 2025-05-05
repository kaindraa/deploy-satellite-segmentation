# utility functions for model loading and mask visualization
import segmentation_models_pytorch as smp
import torch
import numpy as np
from PIL import Image
def get_unet_vgg16(num_classes=4):
    model = smp.Unet(
        encoder_name="vgg16", 
        encoder_weights="imagenet", 
        classes=num_classes
    )
    return model

def load_model_checkpoint(model_path,num_classes,device):
    model=get_unet_vgg16(num_classes)
    model.load_state_dict(torch.load(model_path,map_location=device))
    model.to(device)
    model.eval()
    return model

def apply_color_palette(mask,palette):
    h,w=mask.shape
    rgb=np.zeros((h,w,3),dtype=np.uint8)
    for cls,color in palette.items():
        rgb[mask==cls]=color
    return rgb

# Overlay mask with original image and optional label rendering
def visualize_prediction_overlay(image, pred_mask, palette, alpha=0.5):
    """
    Overlay predicted segmentation mask on the original image with transparency.
    
    image: PIL.Image (original image)
    pred_mask: np.array (mask in shape [H, W] with class indices)
    palette: dict mapping class index to RGB color
    alpha: float between 0 (only image) and 1 (only mask)
    
    Returns: PIL.Image with overlay
    """
    mask_rgb = apply_color_palette(pred_mask, palette)
    mask_pil = Image.fromarray(mask_rgb).resize(image.size, Image.NEAREST)
    overlay = Image.blend(image, mask_pil, alpha)
    return overlay

