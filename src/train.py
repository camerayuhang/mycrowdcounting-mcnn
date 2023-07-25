import torch
from torch import nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from custom_dataset import ShanghaitechDataset
from models import mcnn
from utils import save_checkpoint, load_checkpoint, MAE_MSE
import os
import numpy as np
from tqdm import tqdm

# Hyperparameter
learning_rate = 1e-7
decay = 5*1e-4
batch_size = 1
epochs = 100
in_channels = 3
best_loss = 1e7
pretrain_folder = os.path.join(os.path.dirname(__file__), "../pretrained")

# Setting device
# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# Load dataset

# set root dir
__DATASET_ROOT = os.path.join(
    os.path.dirname(__file__), "../dataset/shanghaitech/")
__PART = os.path.join(__DATASET_ROOT, "part_A")
__PART_TRAIN = os.path.join(__PART, "train_data")

img_folder = os.path.join(__PART_TRAIN, "images")
den_folder = os.path.join(__PART_TRAIN, "density-maps")
csv_file = os.path.join(__PART_TRAIN, "shanghaitech.csv")

# data augmentation
my_transforms = transforms.Compose([

    # toTensor就会使数据集缩放到[0,1]
    transforms.ToTensor(),
    # pytoch官方在imagenet上计算的mean和std
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# split dataset into train and test set
shanghaitechDataset = ShanghaitechDataset(
    img_folder, den_folder, csv_file, my_transforms, down_sample_rate=4)

# for reproducible result
generator = torch.Generator().manual_seed(42)

train_dataset, val_dataset = torch.utils.data.random_split(
    shanghaitechDataset, [0.8, 0.2], generator)

train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size, shuffle=True)


def train_loop(dataloader: DataLoader, model: nn.Module, loss_fn, optimizer):
  size = len(dataloader.dataset)
  num_batches = len(dataloader)
  training_loss = 0
  mae, mse = 0, 0
  model.train()
  for batch_idx, (X, Y) in enumerate(tqdm(dataloader)):
    X, Y = X.to(device), Y.to(device)
    pred = model(X)
    loss = loss_fn(pred, Y)
    mae, mse = MAE_MSE(pred, Y)

    # Backpropagation
    loss.backward()  # 利用反向传播来求梯度
    optimizer.step()  # update weights
    optimizer.zero_grad()  # clear gradient before the next iteration

    loss = loss.item()
    # if ((batch_idx+1) % torch.ceil(torch.tensor(num_batches / 5)).item() == 0) or (batch_idx + 1 == num_batches):
    #   current = (batch_idx+1) * len(X)
    #   print(
    #       f"loss: {loss:>7f}  batches: [{batch_idx+1}/{num_batches}]  samples: [{current}/{size}]")

    training_loss += loss
    mae += mae.item()
    mse += mse.item()
    # break

  # print info each epoch
  training_loss /= num_batches
  mae /= num_batches
  mse /= num_batches
  print(f'batches: {num_batches}/{num_batches} - samples: {size}/{size} - loss: {training_loss:.4f} - MAE: {mae} - MSE: {mse}')


def save_best(test_loss, optimizer, model: nn.Module):
  global best_loss
  if test_loss < best_loss:
    best_loss = min(test_loss, best_loss)
    os.makedirs(pretrain_folder, exist_ok=True)
    save_checkpoint({"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(
    )}, os.path.join(pretrain_folder, "mcnn_checkpoint.pth"))


def validation_loop(dataloader: DataLoader, model: nn.Module, loss_fn):
  size = len(dataloader.dataset)
  num_batches = len(dataloader)
  test_loss = 0
  mae, mse = 0, 0
  model.eval()
  # we don't need to compute the grad in the following loss function
  with torch.no_grad():
    for X, Y in tqdm(dataloader):
      X, Y = X.to(device), Y.to(device)

      pred = model(X)
      loss = loss_fn(pred, Y)
      mae, mse = MAE_MSE(pred, Y)

      test_loss += loss.item()
      mae += mae.item()
      mse += mse.item()
      # break

    test_loss /= num_batches
    mae /= num_batches
    mse /= num_batches

    print(
        f'Test Error: \nbatches: {num_batches}/{num_batches} - samples: {size}/{size} - loss: {test_loss:.4f} - MAE: {mae} - MSE: {mse}')
    return test_loss


def main():

  # load model
  model = mcnn(in_channels).to(device)
  # 这个loss function默认情况下，输入的yhat和y可以任意的shape，但是会对里面的所有元素取平均，不管shape是几个维度
  loss_fn = nn.MSELoss()
  optimizer = torch.optim.Adam(
      params=model.parameters(), lr=learning_rate, weight_decay=decay)

  for i in range(epochs):
    print(f'Epoch {i+1}/{epochs}')
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loss = validation_loop(val_dataloader, model, loss_fn)
    save_best(test_loss, model, optimizer)

  print("Done!")


if __name__ == "__main__":
  main()
