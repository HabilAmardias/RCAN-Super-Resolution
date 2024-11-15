from torch import nn
import torch
from torch import GradScaler

class SqueezeExcitation(nn.Module):
    def __init__(self,C,r):
        super().__init__()
        model = [
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(C,C//r,1),
            nn.ReLU(),
            nn.Conv2d(C//r,C,1),
            nn.Sigmoid()
        ]
        self.model = nn.Sequential(*model)
    def forward(self,x):
        return x * self.model(x)
    
class RCAB(nn.Module):
    def __init__(self,C,r):
        super().__init__()
        conv=[
            nn.Conv2d(C,C,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(C,C,kernel_size=3,padding=1),
        ]
        self.conv = nn.Sequential(*conv)
        self.ca = SqueezeExcitation(C,r)
    def forward(self,x):
        h = self.conv(x)
        h = self.ca(h)
        return h+x

class RG(nn.Module):
    def __init__(self,C,r,num_RCAB=10):
        super().__init__()
        model = [RCAB(C,r) for i in range(num_RCAB)]
        model.append(nn.Conv2d(C,C,kernel_size=3,padding=1))
        self.model = nn.Sequential(*model)
    def forward(self,x):
        return x + self.model(x)

class RiR(nn.Module):
    def __init__(self,C,r,num_RG=5):
        super().__init__()
        model = [RG(C,r) for i in range(num_RG)]
        model.append(nn.Conv2d(C,C,kernel_size=3,padding=1))
        self.model = nn.Sequential(*model)
    def forward(self,x):
        return x + self.model(x)

class RCAN(nn.Module):
    def __init__(self, C=64, r=16):
        super().__init__()
        self.device = 'cpu'
        model = [
            nn.Conv2d(3,C,kernel_size=3,padding=1),
            RiR(C=C,r=r),
            nn.PixelShuffle(upscale_factor=4),
            nn.Conv2d(4,3,kernel_size=3,padding=1),
            nn.Tanh()
        ]
        self.model = nn.DataParallel(nn.Sequential(*model)).to(self.device)
        self.loss_fn = nn.L1Loss()
        self.opt = torch.optim.AdamW(self.parameters(),lr=2e-4)
        self.scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.opt,
                                                                      patience=5,
                                                                      min_lr=1e-6,
                                                                      factor=0.2)
        self.scaler = GradScaler(self.device)
        self.iters_to_accumulate = 2
        
    def forward(self,x):
        h = self.model(x)
        return h

    def prepare_input(self,x,y):
        self.low_res = x.to(self.device)
        self.high_res = y.to(self.device)

    def optimize(self,x,y,iter,num_batch):
        self.prepare_input(x,y)
        self.train()
        with torch.autocast(device_type=self.device,dtype=torch.float16):
            pred = self.forward(self.low_res)
            loss = self.loss_fn(pred,self.high_res)
            loss = loss / self.iters_to_accumulate
        self.scaler.scale(loss).backward()
        if (iter % self.iters_to_accumulate == 0) or (iter == num_batch):
            self.scaler.step(self.opt)
            self.scaler.update()
            self.opt.zero_grad()
        return loss.item() * self.iters_to_accumulate

    def evaluate(self,x,y):
        self.prepare_input(x,y)
        self.eval()
        with torch.no_grad():
            with torch.autocast(device_type=self.device,dtype=torch.float16):
                pred = self.forward(self.low_res)
                loss = self.loss_fn(pred,self.high_res)
        return loss.item()

    def predict(self,x:torch.Tensor)->torch.Tensor:
        self.low_res = x.to(self.device)
        self.eval()
        with torch.no_grad():
            with torch.autocast(device_type=self.device,dtype=torch.float16):
                pred:torch.Tensor = self.forward(self.low_res)
        return pred