import os
import pickle
import pandas as pd
import dataloader_func as dl    
import albumentations as A
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2

def create_dataloader(height=640, width=640, batch_size=4):

    if 'index_split' not in os.listdir():
        df_segmentation = dl.get_df()
        dl.train_val_test_df(df_segmentation)

    df_train, df_val, df_test = pd.read_csv("index_split/train_index.csv"), pd.read_csv("index_split/val_index.csv"), pd.read_csv("index_split/test_index.csv")

    transform_train = A.Compose(
        [ A.Resize(height = height, width = width),  A.HorizontalFlip(0.5) , ToTensorV2()])

    transform_val = A.Compose(
        [A.Resize(height = height, width = width),
        ToTensorV2()])

    if "data_norm.pickle" not in os.listdir():

        transform_filter_before = A.Compose(
            [A.Equalize (mode='cv', always_apply=True),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), always_apply=True),
            A.Normalize(0, 1)])
        
        dataset_train_before_norm, _, _ = dl.create_dataset(df_train, df_val, df_test, transform_train, transform_filter_before, transform_val)
        dl.count_mean_std_set(dataset_train_before_norm)

    with open('data_norm.pickle', 'rb') as f:
        mean_set, std_set = pickle.load(f)

    transform_filter = A.Compose(
        [A.Equalize (mode='cv', always_apply=True),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), always_apply=True),
        A.Normalize(mean_set, std_set)])

    train_ds, val_ds, tset_ds = dl.create_dataset(df_train, df_val, df_test, transform_train, transform_filter, transform_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(tset_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader

