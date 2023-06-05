import os
import cv2
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


def get_df(dir_name = "archive/Lung Segmentation"):
    lst_masks = os.listdir(f"{dir_name}/masks")

    lst_name_masks = [name.split(".png")[0].split("_mask")[0] for name in lst_masks]
    df_segmentation = pd.DataFrame()
    df_segmentation["image"], df_segmentation["mask"] = lst_name_masks, lst_masks

    df_segmentation["image"] = [f'{img}.png' for img in df_segmentation["image"].values]
    df_segmentation["mask"] = [f'{mask}' for mask in df_segmentation["mask"].values]
    return df_segmentation


class LungsDataset(Dataset):
    
    def __init__(self, df, transform, transform_filter, ismask = True):
        
        self.df = df
        self.ismask = ismask
        self.img_path = df["image"].values
        if self.ismask: self.mask_path = df["mask"].values
        self.transform = transform
        self.transform_filter =  transform_filter
        

    def __getitem__(self, idx):

        img = cv2.imread("archive/Lung Segmentation/CXR_png/" + self.img_path[idx], cv2.IMREAD_GRAYSCALE)
        
        if self.ismask is True:

            img = self.transform_filter(image=img)["image"]
            mask = cv2.imread("archive/Lung Segmentation/masks/" + self.mask_path[idx], cv2.IMREAD_GRAYSCALE)
            mask = np.clip(mask, 0, 1).astype("float32")

            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
            return img, mask
    
        else:
          img = self.transform_filter(image=img)["image"]
          augmented = self.transform(image=img)
          img = augmented['image']
          return img
        
    def __len__(self):
        return len(self.df)


def count_mean_std_set(set, x_shape=640, y_shape=640):
    sum_tmp, sum_sq_tmp = 0.0, 0.0
    for img, _ in tqdm(set):
        sum_tmp += img.sum()
        sum_sq_tmp += (img**2).sum()
    mean_set = sum_tmp / (len(set) * x_shape * y_shape)
    std_set = ((sum_sq_tmp / (len(set) * x_shape * y_shape)) - mean_set ** 2) ** 0.5

    data_norm = (mean_set.item(), std_set.item())
    with open('data_norm.pickle', 'wb') as f:
        pickle.dump(data_norm, f)
    return 0


def train_val_test_df(df_segmentation):
    
    train, val_test = train_test_split(df_segmentation,
                              test_size=0.2,
                              random_state=42)

    val, test = train_test_split(val_test,
                              test_size=0.5,
                              random_state=42)
    os.mkdir("index_split")
    train.to_csv("index_split/train_index.csv", index = False)
    val.to_csv("index_split/val_index.csv", index = False)
    test.to_csv("index_split/test_index.csv", index = False)
    return 0


def create_dataset(df_train, df_val, df_test, transform_train, transform_filter, transform_val):

    train_dataset = LungsDataset(df_train, transform=transform_train, transform_filter = transform_filter)
    val_dataset = LungsDataset(df_val, transform=transform_val, transform_filter = transform_filter)
    test_dataset = LungsDataset(df_test, transform=transform_val, transform_filter = transform_filter)

    return train_dataset, val_dataset, test_dataset

