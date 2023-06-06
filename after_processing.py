from dataloader_func import LungsDataset
from torch import nn
from torchvision.utils import save_image
import torch
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import pickle
import os
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import numpy as np
from segmentation_models_pytorch import Unet


def crop_custom(img, x,y,w,h):
  img = img[round(y):round(y+h)+1, round(x):round(x+w)+1]
  return img


def get_segmented_lungs(dataloader, path_to_save = 'masks', height=640, width=640):
   
   kernel_for_dilate = np.ones((3, 3), 'uint8')
   clear_img = np.zeros((height, width), 'uint8')

   os.mkdir(path_to_save)

   torch.cuda.empty_cache()
   for idx, batch in tqdm(enumerate(dataloader)):
      with torch.no_grad():
         mask = unet(batch.to(device))
         mask = np.clip(mask.cpu().detach().numpy(), 0, 1).astype("uint8")
         for i,m in enumerate(mask):
            m = m[0]
            mask_dilate = cv2.dilate(m, kernel_for_dilate)
            contours, _ = cv2.findContours(mask_dilate, cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)
            cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)[:2]
            img_cont = cv2.drawContours(clear_img.copy(), cntsSorted, -1, (255,255,255),-1)
            segm_img = torch.Tensor(np.where(img_cont > 0, batch[i][0], batch[i].min()))
            try:
               all_cont = np.concatenate((cntsSorted[0], cntsSorted[1]))
            except:
               all_cont = cntsSorted[0]
            print(segm_img)
            x,y,w,h = cv2.boundingRect(all_cont)
            segm_img = crop_custom(segm_img, x,y,w,h)
            torch.save(segm_img, f"{path_to_save}/{df_test.iloc[idx * 4 + i, 0].split('.')[0]}.pt")


if __name__ == "__main__":
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   df_test = pd.read_csv("index_split/val_index.csv")
   
   transform_val = A.Compose(
      [A.Resize(height = 640, width = 640),
       ToTensorV2()])
   
   with open('data_norm.pickle', 'rb') as f:
    mean_set, std_set = pickle.load(f)

    transform_filter = A.Compose(
        [A.Equalize (mode='cv', always_apply=True),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), always_apply=True),
        A.Normalize(mean_set, std_set)])

   segmentation_dataset = LungsDataset(df_test, transform=transform_val, transform_filter = transform_filter, ismask=False)
   loader = DataLoader(segmentation_dataset, batch_size=4, shuffle=False, num_workers=4)
   weights_unet = torch.load("unet.pt")
   unet = Unet('efficientnet-b4', activation = nn.LeakyReLU, in_channels = 1)
   unet.load_state_dict(weights_unet)
   unet.to(device)
   unet.eval()
   
   get_segmented_lungs(loader)  

   lungs_final = torch.load(f"masks/{df_test.iloc[29][0].split('.')[0]}.pt") * std_set + mean_set
   save_image(lungs_final, "lungs_final.png")
         



