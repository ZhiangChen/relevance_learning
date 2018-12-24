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
  pos_thr = 0.5

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

  if global_feat:
    cent = np.r_[X.reshape(-1, 1), Y.reshape(-1, 1)]
    gps = utils.cent2gps(cent, x, y, l)
    X = gps[:, 0].reshape(H, W)
    Y = gps[:, 1].reshape(H, W)
  houseFeat[..., 0] = X
  houseFeat[..., 1] = Y

  j = 2
  for i in range(1, np.max(label_img)+1):
    raw = (label_img != i)
    if i != houseLab:
      ed = EDT(raw)
      if global_feat:
        ed = ed * mpp_23 * 2 ** (23 - l)
      houseFeat[..., j] = ed
      j += 1

  xh, yh = np.where(houseRaw > pos_thr)
  xyh = np.c_[xh, yh, houseRaw[xh, yh]]
  np.random.shuffle(xyh)
  xnh, ynh = np.where(nohouseRaw == 1)
  xynh = np.c_[xnh, ynh]
  np.random.shuffle(xynh)
  X_sample = np.zeros([2 * nSamples, depth])
  y_sample = np.r_[xyh[:nSamples, -1], np.zeros([nSamples,])]
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

  y_pred = np.clip(y_pred, 0, 1)
  return y_pred


if __name__ == '__main__':

  imgdir = 'J:/data/multiclass/test_18/test/annotations/training/label_1834.png'
  label_img = np.asarray(Image.open(imgdir))
  selem = np.ones([7, 7])
  sigma = 3
  plt.imshow(GF(BNE(label_img == 2, selem=selem), sigma=sigma))
  y = housefeat_extractor(label_img, houseLab=2, nSamples=200, radius=1.5)
  plt.figure()
  plt.imshow(y)
  plt.show()