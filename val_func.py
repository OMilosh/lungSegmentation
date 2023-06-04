from torch import Tensor, no_grad, sigmoid
from torch.nn import Module
from torchmetrics import JaccardIndex

class DiceLoss(Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        inputs = sigmoid(inputs)       
        
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice


def measure_metric_on_loader(model, dataloader, device):
    pred_masks, gt_masks = [], []
    for batch in dataloader:
        with no_grad():
            img, mask = batch
            img, mask = Tensor(img), Tensor(mask)
            gt_masks.extend(mask.unsqueeze(1).tolist())
            output = model(img.to(device))
            pred_masks.extend(output.cpu().tolist())
    JI = JaccardIndex(task = "binary")
    return JI(Tensor(pred_masks), Tensor(gt_masks))