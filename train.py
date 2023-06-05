import json
from tqdm import tqdm
import torch
from preprocessing import create_dataloader
from segmentation_models_pytorch import Unet
from val_func import DiceLoss


def train(model, criterion, optimizer, dict_plot, device = 'cpu', epoch=None):
    ep_loss = 0
    model.train()
    for img_batch, masks_batch in train_loader:
        optimizer.zero_grad()
        output = model(img_batch.float().to(device))
        loss = criterion(output, masks_batch.unsqueeze(1).to(device))
        loss.backward()
        optimizer.step()
        ep_loss += loss.item()

    val_loss = 0
    for i, batch in enumerate(val_loader):
        with torch.no_grad():
            img_batch, masks_batch = batch
            output = model(img_batch.float().to(device))
            loss = criterion(output, masks_batch.unsqueeze(1).to(device))
            val_loss += loss.item()

    # dict_plot['train_ji'].append(measure_metric_on_loader(model, train_loader, device))
    # dict_plot['val_ji'].append(measure_metric_on_loader(model, val_loader, device))
    dict_plot['train_loss'].append(ep_loss / len(train_loader))
    dict_plot['val_loss'].append(val_loss / len(val_loader))

    print(
        f"Epoch {epoch} Train loss {dict_plot['train_loss'][-1]:.2f} Val loss {dict_plot['val_loss'][-1]:.2f}")

if __name__ == '__main__':
    torch.cuda.empty_cache()
    train_loader, val_loader, test_loader = create_dataloader()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device = ", device)
    unet = Unet('efficientnet-b4', activation = torch.nn.LeakyReLU, in_channels = 1)
    
    criterion = DiceLoss()
    unet.to(device)
    optimizer = torch.optim.AdamW(unet.parameters(), lr=0.01)
    dict_plot = {name: [] for name in ['train_loss', 'val_loss']}


    for epoch in tqdm(range(40)):
        train(unet, criterion, optimizer, dict_plot, device, epoch)
        if epoch == 15:
            optimizer = torch.optim.AdamW(unet.parameters(), lr=0.001)

    dict_loss_json = json.dumps(dict_plot)

    with open("loss.json", "w") as f:
        f.write(dict_loss_json)

    torch.save(unet.state_dict(), "unet.pt")

    
