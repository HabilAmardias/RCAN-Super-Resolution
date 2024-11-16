import cv2
import numpy as np

def display_upscaled(original,upscaled):
    upscaled_height,upscaled_width = upscaled.shape[:2]
    original_height,original_width = original.shape[:2]
    pad_height, pad_width = (upscaled_height-original_height)//2, (upscaled_width-original_width)//2
    padded_original = cv2.copyMakeBorder(original,pad_height,pad_height,pad_width,pad_width,cv2.BORDER_CONSTANT)
    concatenated = cv2.hconcat([padded_original,upscaled])
    return concatenated