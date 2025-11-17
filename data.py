import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import glob
from torch.utils.data import Dataset
import argparse


class PlateDataset(Dataset):
    def __init__(self, plate_ids, strict_discard=True, load_imgs=True, load_masks=False, 
                 load_chrometrics=False, load_dino=False):
        self.imgs, self.info, self.masks, self.chrometrics, self.dino_zs = self._load_data(plate_ids, strict_discard, load_imgs, 
                                                                                           load_masks, load_chrometrics, load_dino)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        return self.imgs[idx], self.info.iloc[idx]

    def _load_data(self, plates, strict_discard, load_imgs, load_masks, load_chrometrics, load_dino,
                   data_dir=f'/ewsc/hschluet/pbmc5/bundled_data/'):
      info = []
      imgs = []
      masks = []
      chrometrics = []
      dino_zs = []

      for plate in tqdm(plates):
        df = pd.read_csv(f'{data_dir}plate_{plate}_info.csv')
        info.append(df)
        if load_imgs:
          imgs.append(torch.load(f'{data_dir}plate_{plate}_imgs.pt'))
        if load_masks:
          masks.append(torch.load(f'{data_dir}plate_{plate}_masks.pt'))
        if load_chrometrics:
          chrometrics.append(pd.read_csv(f'{data_dir}plate_{plate}_chrometric.csv'))
        if load_dino:
          dino_zs.append(torch.load(f'{data_dir}plate_{plate}_dino_feats.pt', weights_only=False))

      info = pd.concat(info)
      if load_imgs:
        imgs = torch.cat(imgs).reshape(-1, 1, 32, 32)
      if load_masks:
        masks = torch.cat(masks).reshape(-1, 1, 32, 32)
      if load_dino:
         dino_zs = np.concatenate(dino_zs)
      if load_chrometrics:
        chrometrics = pd.concat(chrometrics)

      if strict_discard:
        bad_idx = ~info['qc'].values
        info = info.loc[~bad_idx].reset_index(drop=True)
        if load_dino:
            dino_zs = dino_zs[~bad_idx]
        if load_chrometrics:
            chrometrics = chrometrics.loc[~bad_idx].reset_index(drop=True)
        bad_idx = torch.tensor(bad_idx, dtype=torch.bool)
        if load_imgs:
            imgs = imgs[~bad_idx]
        if load_masks:
            masks = masks[~bad_idx]

      groups = pd.read_csv('meta/patient_diagnosis_groups.csv')
      info = info.merge(groups, on='patient', how='left')

      return imgs, info, masks, chrometrics, dino_zs
