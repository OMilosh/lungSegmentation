import torch
import pickle
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torchmetrics import JaccardIndex

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        inputs = torch.sigmoid(inputs)       
        
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice


def measure_metric_on_loader(model, dataloader, device):
    pred_masks, gt_masks = [], []
    for batch in dataloader:
        with torch.no_grad():
            img, mask = batch
            img, mask = torch.Tensor(img), torch.Tensor(mask)
            gt_masks.extend(mask.unsqueeze(1).tolist())
            output = model(img.to(device))
            pred_masks.extend(output.cpu().tolist())
    JI = JaccardIndex(task = "binary")
    return JI(torch.Tensor(pred_masks), torch.Tensor(gt_masks))

def show_pic_with_mask(model, dataloader, device = 'cpu'):
    with open('data_norm.pickle', 'rb') as f:
        mean_set, std_set = pickle.load(f)
    for idx, batch in enumerate(dataloader):
        with torch.no_grad():
            img, mask = batch
            output = model(img.to(device))
            for i in range(4):
                plt.axis('off')
                plt.imshow(img[i].permute(1,2,0)*std_set + mean_set , cmap = "Greys_r")
                plt.imshow(np.ma.masked_where(mask[i].cpu().reshape(640,640).detach().numpy() == False, 
                                              mask[i].cpu().reshape(640,640).detach().numpy()), alpha=0.9, cmap='coolwarm')
                plt.imshow(np.ma.masked_where(output[i].cpu().reshape(640,640).detach().numpy() == False,
                                              output[i].cpu().reshape(640,640).detach().numpy()), alpha=0.3, cmap='Reds')
            
                patches = [ mpatches.Patch(color="blue", label="GT_mask"), mpatches.Patch(color="red", label="Unet_mask")]
                plt.legend(handles=patches, loc=2, borderaxespad=0. )
                plt.savefig(f'seg_lung/{idx*4 + i}.png', bbox_inches='tight')
                torch.cuda.empty_cache()


