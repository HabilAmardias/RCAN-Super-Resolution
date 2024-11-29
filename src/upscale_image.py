import torch
import numpy as np
from model import RCAN
from torchvision.transforms import v2
from PIL import Image

transforms = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32,scale=True),
    v2.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
])

def load_image(upload):
    image = Image.open(upload).convert('RGB')
    return image

def exceed_resolution_threshold(image:Image,threshold=350*350):
    height, width = image.size
    return height * width > threshold

def upscale_image(image:Image,
                  model:RCAN):
    
    tensor = transforms(image).unsqueeze(0) #[0,255] -> [-1,1]
    pred = model.predict(tensor) #generate upscaled image
    pred = pred*127.5+127.5 #[-1,1] -> [0,255]
    pred = pred.permute(0,2,3,1).squeeze(0).cpu().numpy().astype(np.uint8)
    return Image.fromarray(pred)