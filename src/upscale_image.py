import torch
import numpy as np
from model import RCAN
from torchvision.transforms import v2
import cv2

transforms = v2.Compose([
    v2.ToDtype(torch.float32,scale=True),
    v2.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
])

def upscale_image(upload,
                  model:RCAN):
    
    file = np.asarray(bytearray(upload.read()),dtype=np.uint8) #convert to bytes
    image = cv2.imdecode(file,cv2.IMREAD_COLOR)
    image = np.moveaxis(image,-1,0) #(H,W,C) -> (C,H,W)
    image = np.expand_dims(image, axis=0) #(C,H,W) -> (1,C,H,W)
    tensor = torch.tensor(image)
    tensor = transforms(tensor) #[0,255] -> [-1,1]
    pred = model.predict(tensor).cpu().numpy().squeeze(axis=0) #generate upscaled image and convert it to numpy
    pred = np.moveaxis(pred,0,-1) #(C,H,W) -> (H,W,C)
    pred = pred*127.5+127.5 #[-1,1] -> [0,255]
    pred = np.clip(pred,0,255).astype(np.uint8) #convert to unsigned 8-bit integer
    _, encoded = cv2.imencode('.png',pred)

    return encoded.tobytes()