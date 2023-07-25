# importing libraries
import h5py
import scipy.io as sio
import PIL.Image as Image
import numpy as np
import os
import glob
from scipy.ndimage.filters import gaussian_filter
import scipy
import sys
from joblib import Parallel, delayed
import time
import csv

__DATASET_ROOT = os.path.join(
    os.path.dirname(__file__), "../dataset/shanghaitech/")
__OUTPUT_NAME = __DATASET_ROOT

# function to create density maps for images


def gaussian_filter_density(gt):
  print(gt.shape)
  density = np.zeros(gt.shape, dtype=np.float32)
  gt_count = np.count_nonzero(gt)
  if gt_count == 0:
    return density

  pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))
  leafsize = 2048
  # build kdtree
  tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
  # query kdtree
  distances, locations = tree.query(pts, k=4)

  print('generate density...')
  for i, pt in enumerate(pts):
    pt2d = np.zeros(gt.shape, dtype=np.float32)
    pt2d[pt[1], pt[0]] = 1.
    if gt_count > 1:
      sigma = (distances[i][1]+distances[i][2]+distances[i][3])*0.1
    else:
      sigma = np.average(np.array(gt.shape))/2./2.  # case: 1 point
    density += scipy.ndimage.filters.gaussian_filter(
        pt2d, sigma, mode='constant')
  print('done.')
  return density


# 应该增加参数用来是否向下采样输入图，代码中好像是对密度图向下采样，因为输入图已经标注好了在mat里面
def generate_density_map(img_path):
  print(img_path)
  # replace会返回一个新的路径，不会修改原路径
  mat_path = img_path.replace('.jpg', '.mat').replace(
      'images', 'ground-truth').replace('IMG_', 'GT_IMG_')
  print('mat_path ', mat_path)
  mat = sio.loadmat(mat_path)
  # 使用 plt.imread 函数读取当前图像文件，将图像读取为一个NumPy数组 img。
  # 或者用PIL配合numpy
  imgfile = Image.open(img_path)
  img = np.asarray(imgfile)
  k = np.zeros((img.shape[0], img.shape[1]))
  # 直接嵌套[0,0]把location字段提取出来
  gt = mat["image_info"][0, 0][0, 0][0]
  for i in range(0, len(gt)):
    # 这个训练就是论文中的theta函数，有头的话，离散密度图的pixel为1
    # int(gt[i][1])表示头的y坐标，而图像左上角为0，0，下为y，右为x
    # 图像的二维数组也是从上到下按行越来越大，所以这样赋值1是合理的
    if int(gt[i][1]) < img.shape[0] and int(gt[i][0]) < img.shape[1]:
      # 所以k的索引是行列，行应该对应y坐标
      k[int(gt[i][1]), int(gt[i][0])] = 1
  # 离散变连续密度图
  k = gaussian_filter_density(k)
  # 存储成h5格式
  output_path = img_path.replace(__DATASET_ROOT, __OUTPUT_NAME).replace(
      '.jpg', '.h5').replace('images', 'density-maps')
  density_map_folder = os.path.dirname(output_path)
  if not os.path.exists(density_map_folder):
    # 中间的文件夹也会被创建
    os.makedirs(density_map_folder)
  print("output", output_path)
  sys.stdout.flush()
  with h5py.File(output_path, 'w') as hf:
    # 存储在h5格式文件的density字段里
    hf['density'] = k
  return img_path


def generate_shanghaitech_path(root):
  # now generate the ShanghaiA's ground truth
  print(os.getcwd())  # 获取当前工作目录路径
  print(os.path.abspath('.'))  # 获取当前工作目录路径
  print(os.path.dirname(__file__))
  part_A_train = os.path.join(root, 'part_A/train_data', 'images')
  part_A_test = os.path.join(root, 'part_A/test_data', 'images')
  part_B_train = os.path.join(root, 'part_B/train_data', 'images')
  part_B_test = os.path.join(root, 'part_B/test_data', 'images')

  img_paths_a_train = []
  img_paths_a_test = []
  img_paths_b_train = []
  img_paths_b_test = []

  for img_path in glob.glob(os.path.join(part_A_train, '*.jpg')):
    img_paths_a_train.append(img_path)
  for img_path in glob.glob(os.path.join(part_B_train, '*.jpg')):
    img_paths_b_train.append(img_path)

  for img_path in glob.glob(os.path.join(part_A_test, '*.jpg')):
    img_paths_a_test.append(img_path)
  for img_path in glob.glob(os.path.join(part_B_test, '*.jpg')):
    img_paths_b_test.append(img_path)

  return img_paths_a_train, img_paths_a_test, img_paths_b_train, img_paths_b_test


def generate_mapping_csvfile(img_list: list):
  csv_header = ['orginal_image', "ground_true"]
  rows = []
  for img_path in img_list:
    orginal_image: str = os.path.basename(img_path)
    ground_true = orginal_image.replace("jpg", "h5")
    rows.append({csv_header[0]: orginal_image, csv_header[1]: ground_true})

  csv_file_path = os.path.join(os.path.dirname(
      img_list[0]), "../shanghaitech.csv")
  with open(csv_file_path, "w") as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=csv_header)
    writer.writeheader()
    writer.writerows(rows)


if __name__ == "__main__":
  """
  TODO: this file will preprocess crowd counting dataset
  """

  start_time = time.time()
  a_train, a_test, b_train, b_test = generate_shanghaitech_path(__DATASET_ROOT)
  all_images = [*a_train, *a_test, *b_train, *b_test]

  # If -1 all CPUs are used
  # delayed是个装饰器，用来封装想要并行的函数
  # generate_density_map(p) for p in a_train) 单单这个表达式是表示窜行
  # 先写generator再写并行
  Parallel(n_jobs=-1)(delayed(generate_density_map)(p) for p in all_images)
  print("--- %s seconds ---" % (time.time() - start_time))
  generate_mapping_csvfile(a_train)
  generate_mapping_csvfile(a_test)
  generate_mapping_csvfile(b_train)
  generate_mapping_csvfile(b_test)
