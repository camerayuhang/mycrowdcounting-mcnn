import torch


def get_mean_std(loader):
  # var[X] = E[X**2] - E[X]**2
  channels_sum, channels_sqrd_sum, num_batches = 0, 0, 0
  for data, _ in (loader):
    # 第1个维度表示channel数量，把0，2，3维度求均值，最后只会保留channel数量，意味着把一个批次的数据集，在channel的地方求了均值
    channels_sum += torch.mean(data, dim=[0, 2, 3])
    channels_sqrd_sum += torch.mean(data**2, dim=[0, 2, 3])
    num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_sqrd_sum / num_batches - mean**2) ** 0.5

  return mean, std


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
  print("=> Saving checkpoint")
  torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
  print("=> Loading checkpoint")
  model.load_state_dict(checkpoint["state_dict"])
  optimizer.load_state_dict(checkpoint["optimizer"])


def MAE_MSE(pred: torch.Tensor, Y: torch.Tensor):
  diff = torch.sum(pred, dim=[2, 3]) - torch.sum(Y, dim=[2, 3])
  abs_diff = torch.abs(diff)
  square_diff = diff ** 2

  return torch.mean(abs_diff), torch.sqrt(torch.mean(square_diff))
