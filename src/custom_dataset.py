from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import os
import h5py
import numpy as np
import cv2
import torch


class ShanghaitechDataset(Dataset):
  def __init__(self, orginal_img_dir, density_map_dir, cvs_file, transform=None, down_sample_rate=4) -> None:
    super().__init__()
    self.orginal_img_dir = orginal_img_dir
    self.density_map_dir = density_map_dir
    self.annotations = pd.read_csv(cvs_file)
    self.transform = transform
    self.down_sample_rate = down_sample_rate

  def __len__(self):
    return len(self.annotations)

  def __getitem__(self, index):
    img_path = os.path.join(self.orginal_img_dir,
                            self.annotations.iloc[index, 0])
    den_path = os.path.join(self.density_map_dir,
                            self.annotations.iloc[index, 1])
    # 确保原图像为RGB图像
    img = Image.open(img_path).convert("RGB")

    # print(img_path)
    # 对训练数据进行增强
    # 转换指的是对当前这个图像进行转换，并不会增加原始图像的数量
    # 同时这个数据集不能做那种flip或者rotation转换，因为这样转换density map会变化
    if self.transform:
      img = self.transform(img)

    den_map = self.load_den_map(den_path)
    # print(den_map)
    # 效果一样
    den_map = torch.unsqueeze(torch.tensor(den_map), dim=0)
    # print(den_map)
    return (img, den_map)

  # 需要向下采样1/4
  # 这个方法应该在transform里表示
  def load_den_map(self, den_path: str):
    with h5py.File(den_path) as hf:
      target = np.asarray(hf["density"])
      # 第二个参数是（width，height）
      target = cv2.resize(target, (int(target.shape[1]/self.down_sample_rate), int(target.shape[0] /
                          self.down_sample_rate)), interpolation=cv2.INTER_CUBIC) * (self.down_sample_rate**2)
      return target
