from preprocessing import create_dataloader
from val_func import measure_metric_on_loader, show_pic_with_mask
import matplotlib.pyplot as plt
import torchmetrics
from segmentation_models_pytorch import Unet
import torch

if __name__ == "__main__":
    unet = unet = Unet('efficientnet-b4', activation = torch.nn.LeakyReLU, in_channels = 1)
    unet.load_state_dict(torch.load("unet.pt"))
    unet.eval()
    _, val_loader, test_loader = create_dataloader()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet.to(device)
    
    # print(f"Jaccard index on test: {measure_metric_on_loader(unet, test_loader, device):.2f}")
    torch.cuda.empty_cache()

    show_pic_with_mask(unet, test_loader,device)
    