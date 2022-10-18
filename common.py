import torch
import torchvision
from torchvision import transforms
import torch.nn as nn

import matplotlib.pyplot as plt
import numpy as np

import torchvision.transforms.functional as TF
import random

def get_dataloaders(img_size=256, batch_size=16):
    
    image_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    target_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Lambda(lambda x: np.round(x)), # Round the labels to get integer like labels
        transforms.Lambda(lambda x: x - 1.9), # labels will go from (1, 2, 3) to (-0.9, 0.1, 1.1)
        transforms.Lambda(lambda x: np.minimum(x, 0)), # labels will be (-0.9, 0)
        transforms.Lambda(lambda x: 1 - np.round(x + 0.9)), # labels will be (0, 1) where 1 is pet, 0 is everything else
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(img_size, img_size).long()),

    ])
    
    dataset = torchvision.datasets.OxfordIIITPet(
        root='./pets',
        split='trainval',
        target_types='segmentation',
        download=True,
        transform=image_transforms,
        target_transform=target_transforms
    )
    
    images = len(dataset)
    train_idx = int(0.8 * images)

    train_snip, val_snip = torch.utils.data.random_split(dataset, [train_idx, images - train_idx])

    train_dataloader = torch.utils.data.DataLoader(train_snip, batch_size)
    val_dataloader = torch.utils.data.DataLoader(val_snip, batch_size)
    
    return train_dataloader, val_dataloader


# From: https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch/notebook#Dice-Loss
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        #flatten label and prediction tensors
        inputs = inputs.reshape(-1) # if we select a slice we need to use reshape as memory needs to be contiguous
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice
    


def train_loop(dataloader, model, loss_fn, optimizer, device, step=0, wb=None, augment=True):
    size = len(dataloader.dataset)
    
    for batch, (X, y) in enumerate(dataloader):
        # Apply some random augmentaion
        if augment:
            t = random.random()
            if t < 0.5:
                # Left Right Flip the whole batch
                X = TF.hflip(X)
                y = TF.hflip(y)
            
            # Rotate the batch a random amount
            r = random.randint(-15, 15)
            X = TF.rotate(X, r)
            y = TF.rotate(y, r)
            
            # Adjust the brightness randomly from 0.9 to 1.1
            b = random.random() * 0.2 + 0.9
            X = TF.adjust_brightness(X, b)
        
        prediction = model(X.to(device))
        loss = loss_fn(prediction, y.to(device))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        step = step + 1
        if wb is not None:
            wb.log({"loss": loss.item()}, step=step)
        
        if batch % 5 == 0:
            loss, current = loss.item(), batch * len(X)
            
            print(f"loss: {loss:>4f} [{current:>5d}/{size:>5d}]")
    
    return step

            

def test_loop(dataloader, model, loss_fn, device, step=0, wb=None):
    with torch.no_grad():
        size = len(dataloader.dataset)
        
        losses = []
        dices = []
        hard_dices = []
        
        dl = DiceLoss()
        
        for batch, (X, y) in enumerate(dataloader):
            prediction = model(X.to(device))
            loss = loss_fn(prediction, y.to(device))
            
            # Note this just does the active class
            scaled = nn.functional.softmax(prediction, dim=1)[:, 1, :, :]
            dice_soft = 1 - dl(scaled, y.to(device))
            dice_hard = 1 - dl(torch.round(scaled), y.to(device))
            
            losses += [loss.item()]
            dices += [dice_soft.item()]
            hard_dices += [dice_hard.item()]
            
            if batch == 0:
                pred = nn.functional.softmax(prediction[0], dim=0).cpu().detach()[1]
                img = X[0].permute(1, 2, 0) # 3, 128, 128

                plt.imshow(img)
                plt.contour(pred, levels=[0.0, 0.5, 1.0])
                plt.clim(0.0, 1.0)
                plt.colorbar()
                plt.show()
                
                if wb is not None:
                    wb.log({
                        "image_in": wb.Image(X[0], caption="input"),
                        "prediction": wb.Image(pred, caption="prediction"),
                        "hard-prediction": wb.Image(torch.round(pred), caption="prediction hard")
                    }, step=step)
        
        if wb is not None:
            wb.log({"eval-loss": np.mean(losses), "eval-dice": np.mean(dices), "eval-dice-hard": np.mean(hard_dices)}, step=step)
            
        print(f"Mean batch loss on validation set: {np.mean(losses):>4f} dice: {np.mean(dices):>4f}")