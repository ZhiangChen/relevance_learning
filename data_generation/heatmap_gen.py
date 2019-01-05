import numpy as np
import scipy as sp
from PIL import Image
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from scipy.ndimage.morphology import distance_transform_edt as EDT
from skimage.morphology import binary_erosion as BNE
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from skimage.filters import gaussian as GF
import os
import utils
import matplotlib.pyplot as plt
import glob
import argparse

def housefeat_extractor(label_img, houseLab = 3, global_feat=False, x=0, y=0, l=0, nSamples=1000, radius=1.0):
  '''
  Extract feature of one label image / generate
  :param label_img:
  :param houseLab:
  :return:
  '''
  # some parameters
  depth = 4
  selem = np.ones([7, 7])
  sigma = 3
  mpp_23 = 0.0187
  pos_thr = 0.2

  try:
    H, W = label_img.shape
  except:
    H, W, _ = label_img.shape

  if global_feat and l == 0:
    raise ValueError('Trying to use global feature, but no level specified')

  # houseGT = np.zeros([H, W])
  houseFeat = np.zeros([H, W, depth])
  X, Y = np.meshgrid(np.arange(0, W), np.arange(0, H))
  houseRaw = (label_img == houseLab)
  houseRaw = GF(BNE(houseRaw, selem=selem), sigma=sigma)
  nohouseRaw = np.bitwise_and(label_img > 0, label_img != houseLab)
  houseRaw[nohouseRaw] = 0
  nohouseRaw = BNE(nohouseRaw, selem=selem)

  if global_feat:
    cent = np.r_[X.reshape(-1, 1), Y.reshape(-1, 1)]
    gps = utils.cent2gps(cent, x, y, l)
    X = gps[:, 0].reshape(H, W)
    Y = gps[:, 1].reshape(H, W)
  houseFeat[..., 0] = X
  houseFeat[..., 1] = Y

  j = 2
  label_img_new = np.zeros_like(label_img)
  cnt = 1
  for i in range(1, np.max(label_img)+1):
    raw = (label_img != i)
    if i != houseLab:
      label_img_new[label_img == i] = cnt
      cnt += 1
      ed = EDT(raw)
      if global_feat:
        ed = ed * mpp_23 * 2 ** (23 - l)
      houseFeat[..., j] = ed
      j += 1


  xh, yh = np.where(houseRaw > pos_thr)
  xyh = np.c_[xh, yh, houseRaw[xh, yh]]
  if xyh.shape[0] == 0:
    return np.zeros([H, W]), label_img_new.astype(np.uint8)
  elif xyh.shape[0] < 200:
    nrepeats = int(200 / xyh.shape[0] + 1)
    xyh = np.tile(xyh, (nrepeats, 1))
  np.random.shuffle(xyh)
  xnh, ynh = np.where(nohouseRaw == 1)
  xynh = np.c_[xnh, ynh]
  if xynh.shape[0] < 2000:
    return houseRaw, label_img_new.astype(np.uint8)
    # nrepeats = int(200 / xynh.shape[0] + 1)
    # xynh = np.tile(xynh, (nrepeats, 1))
  np.random.shuffle(xynh)
  X_sample = np.zeros([2 * nSamples, depth])
  y_sample = np.r_[xyh[:nSamples, -1], np.zeros([nSamples,])]
  print(xyh.shape, xynh.shape)
  for i in range(depth):
    xyh_i = (xyh[:nSamples, 0].astype('int'), xyh[:nSamples, 1].astype('int'), i * np.ones([nSamples], dtype=np.int32))
    xynh_i = (xynh[:nSamples, 0].astype('int'), xynh[:nSamples, 1].astype('int'), i * np.ones([nSamples], dtype=np.int32))
    X_sample[:, i] = np.r_[houseFeat[xyh_i].reshape(-1, 1), houseFeat[xynh_i].reshape(-1, 1)].reshape(-1)

  gpr = GPR(kernel=RBF(radius, length_scale_bounds="fixed")*C(10.0, constant_value_bounds="fixed"), alpha=1e-5, normalize_y=False)
  gpr.fit(X_sample/256, y_sample)
  X_pred = np.zeros([H*W, depth])
  y_pred = np.zeros([H, W])
  for i in range(depth):
    X_pred[:, i] = houseFeat[..., i].reshape(-1)
  for i in range(256):
      start = int(i * W * (H / 256))
      end = int(start + W * (H / 256))
      y_pred_tmp = gpr.predict(X_pred[start:end, :]/256)
      y_pred[int(i*(H/256)):int((i+1)*(H/256))] = y_pred_tmp.reshape(int(H/256), int(W))

  y_pred = np.clip(np.tanh(y_pred), 0, 1)
  return y_pred, label_img_new.astype(np.uint8)


if __name__ == '__main__':

  # ap = argparse.ArgumentParser()
  # ap.add_argument('-i', '--input_folder', required=True, dest='input_folder')
  # ap.add_argument('-o', '--output_folder', required=True, dest='output_folder')
  # args = ap.parse_args()
  # idir = args.input_folder
  # odir = args.output_folder
  idir = 'J:/data/multiclass/test_19/test'
  odir = 'J:/data/multiclass/test_19/train'

  folders = ['images', 'annotations']
  targets = ['images', 'masks', 'heatmaps']
  sets = ['training', 'validation']

  if not os.path.exists(odir):
    for set in sets:
      for t_folder in targets:
        os.makedirs(os.path.join(odir, set, t_folder))
  else:
    for set in sets:
      for t_folder in targets:
        try:
          os.makedirs(os.path.join(odir, set, t_folder))
        except:
          pass

  for set in sets:
    i_paths = glob.glob(os.path.join(idir, 'images', set, '*.png'))
    for i_path in i_paths:
      print('processing ' + i_path)
      i_path = i_path.split('\\')[-1]
      i_name = i_path.split('.')[0]
      img = Image.open(os.path.join(idir, 'images', set, i_path))
      img.save(os.path.join(odir, set, 'images', i_path), 'PNG')
      label = np.asarray(Image.open(os.path.join(idir, 'annotations', set, 'label_' + i_path)))
      heatmap, new_label = housefeat_extractor(label, houseLab=2, nSamples=200, radius=1.5)
      # hmg = Image.fromarray(heatmap, mode='F')
      hmg = Image.fromarray(heatmap)
      lmg = Image.fromarray(new_label)
      hmg.save(os.path.join(odir, set, 'heatmaps', i_path[:-4]+'.tiff'), 'TIFF')
      lmg.save(os.path.join(odir, set, 'masks', i_path), 'PNG')


  # imgdir = 'J:/data/multiclass/test_18/test/annotations/training/label_1834.png'
  # label_img = np.asarray(Image.open(imgdir))
  # selem = np.ones([7, 7])
  # sigma = 3
  # plt.imshow(GF(BNE(label_img == 2, selem=selem), sigma=sigma))
  # y = housefeat_extractor(label_img, houseLab=2, nSamples=200, radius=1.5)
  # plt.figure()
  # plt.imshow(y)
  # plt.show()