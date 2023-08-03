import torch
import os
from torch import nn
import logging
import torchvision.transforms as transforms
from custom_dataset import ShanghaitechDataset
from torch.utils.data import DataLoader
from utils import save_checkpoint, load_checkpoint, MAE_MSE
from tqdm import tqdm
from models import CSRNet
import math


class CSRTrain:
  def __init__(self, learning_rate=1e-7, decay=5*1e-4, batch_size=1, epochs=1) -> None:
    self.__device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    self.__best_loss = 1e7
    self.__mae = 1e7
    self.__mse = 1e7
    self.checkpoint_name = "csr_checkpoint.pth"
    self.pretrain_folder = os.path.join(
        os.path.dirname(__file__), "../pretrained")
    self.checkpoint_path = os.path.join(
        self.pretrain_folder, self.checkpoint_name)

    self.set_logging_config()
    self.set_hyperparameters(
        learning_rate, decay, batch_size, epochs)
    self.load_dataset()
    self.complie()

  # Hyperparameter

  def set_hyperparameters(self, learning_rate, decay, batch_size, epochs):
    self.__learning_rate = learning_rate
    self.__decay = decay
    self.__batch_size = batch_size
    self.__epochs = epochs

  def load_dataset(self):
    # set root dir
    self.__DATASET_ROOT = os.path.join(
        os.path.dirname(__file__), "../dataset/shanghaitech/")
    self.__PART = os.path.join(self.__DATASET_ROOT, "part_A")
    self.__PART_TRAIN = os.path.join(self.__PART, "train_data")

    img_folder = os.path.join(self.__PART_TRAIN, "images")
    den_folder = os.path.join(self.__PART_TRAIN, "density-maps")
    csv_file = os.path.join(self.__PART_TRAIN, "shanghaitech.csv")

    self.define_data_augmentation()

    shanghaitechDataset = ShanghaitechDataset(
        img_folder, den_folder, csv_file, self.transforms, down_sample_rate=8)

    train_dataset, val_dataset = torch.utils.data.random_split(
        shanghaitechDataset, [0.9, 0.1])

    self.train_dataloader = DataLoader(
        train_dataset, batch_size=self.__batch_size, shuffle=True)
    self.val_dataloader = DataLoader(
        val_dataset, self.__batch_size, shuffle=True)

  # data augmentation

  def define_data_augmentation(self):
    torch.manual_seed(20)
    self.transforms = transforms.Compose([
        # toTensor就会使数据集缩放到[0,1]
        transforms.ToTensor(),
        # pytoch官方在imagenet上计算的mean和std
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    self.target_transform = self.transforms

  def complie(self):
    self.model = CSRNet(config=3).to(self.__device)
    self.loss_fn = nn.MSELoss(reduction="sum")
    self.optimizer = torch.optim.Adam(
        params=self.model.parameters(), lr=self.__learning_rate, weight_decay=self.__decay)
    if os.path.exists(self.checkpoint_path):
      self.logger.info(f"loading the checkpoint from {self.checkpoint_path}")
      checkpoint = torch.load(self.checkpoint_path,
                              map_location=torch.device(self.__device))
      load_checkpoint(checkpoint, self.model, self.optimizer)
      self.__best_loss = checkpoint["best_loss"]
      self.__mae = checkpoint["mae"]
      self.__mse = checkpoint["mse"]
      self.logger.info(
          f"loaded the checkpoint with {self.__best_loss} loss, {self.__mae} mae, {self.__mse} mse")

  def train_loop(self, dataloader: DataLoader, model: nn.Module, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    training_loss = 0
    mae, mse = 0, 0
    model.train()
    for batch_idx, (X, Y) in enumerate(tqdm(dataloader)):
      X, Y = X.to(self.__device), Y.to(self.__device)
      pred = model(X)
      loss = loss_fn(pred, Y)
      mae, mse = MAE_MSE(pred, Y)

      # Backpropagation
      loss.backward()  # 利用反向传播来求梯度
      optimizer.step()  # update weights
      optimizer.zero_grad()  # clear gradient before the next iteration

      loss = loss.item()

      training_loss += loss
      mae += mae.item()
      mse += mse.item() ** 2
      # break

    # print info each epoch
    training_loss /= num_batches
    mae /= num_batches
    mse /= num_batches
    mse = math.sqrt(mse)
    self.logger.info(
        f'Train Error: \nbatches: {num_batches}/{num_batches} - samples: {size}/{size} - loss: {training_loss:.4f} - MAE: {mae} - MSE: {mse}')

  def validation_loop(self, dataloader: DataLoader, model: nn.Module, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0
    mae, mse = 0, 0
    model.eval()
    # we don't need to compute the grad in the following loss function
    with torch.no_grad():
      for X, Y in tqdm(dataloader):
        X, Y = X.to(self.__device), Y.to(self.__device)

        pred = model(X)
        loss = loss_fn(pred, Y)
        mae, mse = MAE_MSE(pred, Y)

        test_loss += loss.item()
        mae += mae.item()
        mse += mse.item() ** 2
        # break

      test_loss /= num_batches
      mae /= num_batches
      mse /= num_batches
      mse = math.sqrt(mse)

      self.logger.info(
          f'Test Error: \nbatches: {num_batches}/{num_batches} - samples: {size}/{size} - loss: {test_loss:.4f} - MAE: {mae} - MSE: {mse}')
      return test_loss, mae, mse

  def set_logging_config(self):
    # 创建一个日志记录器
    self.logger = logging.getLogger('my_logger')
    self.logger.setLevel(logging.DEBUG)

    # 创建一个控制台处理器，并设置日志级别为DEBUG
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # 创建一个文件处理器，并设置日志级别为INFO
    file_handler = logging.FileHandler('train_detail.log', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)

    # 定义日志输出格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # 将处理器添加到日志记录器
    self.logger.addHandler(console_handler)
    self.logger.addHandler(file_handler)

  def save(self, loss, mae, mse, model: nn.Module, optimizer):
    if loss < self.__best_loss:
      self.__best_loss = min(loss, self.__best_loss)
      self.__mae = mae
      self.__mse = mse
      os.makedirs(self.pretrain_folder, exist_ok=True)
      save_checkpoint({"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(
      ), "best_loss": self.__best_loss, "mae": self.__mae, "mse": self.__mse}, os.path.join(self.pretrain_folder, self.checkpoint_name))
      self.logger.info(
          f"saveing the checkpoint at the error of {self.__best_loss} loss, {self.__mae} mae, {self.__mse} mse separately")

  def run(self):
    self.logger.info(f"train the model using {self.__device} device")
    for i in range(self.__epochs):
      self.logger.info(
          f'Epoch {i+1}/{self.__epochs}')
      self.logger.info("-------------------------------")
      self.train_loop(dataloader=self.train_dataloader, model=self.model,
                      loss_fn=self.loss_fn, optimizer=self.optimizer)
      loss, mae, mse = self.validation_loop(
          dataloader=self.val_dataloader, model=self.model, loss_fn=self.loss_fn)
      self.save(loss=loss, mae=mae, mse=mse,
                model=self.model, optimizer=self.optimizer
                )


if __name__ == "__main__":
  csrtrain = CSRTrain()
  csrtrain.run()
