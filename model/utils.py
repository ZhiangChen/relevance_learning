import keras.backend as K
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

def mse(y_true, y_pred):
  return K.mean(K.square(y_pred - y_true))


def weighted_L2(y_true, y_pred):
  return K.mean(K.exp(K.clip(y_true, 0, 1) - K.clip(y_pred, 0, 1)) * K.square(y_pred - y_true))


def save_images(net_out, save_dir, visualize=False):
  if not os.path.exists(os.path.join(save_dir, 'masks')):
    os.makedirs(os.path.join(save_dir, 'masks'))
    os.makedirs(os.path.join(save_dir, 'heatmaps'))
  segs = net_out[0]
  heatmaps = net_out[1]
  N = segs.shape[0]
  for i in range(N):
    seg = (np.argmax(segs[i, ...], axis=3) * 127).astype(np.uint8)
    heatmap = heatmaps[i, ..., 0]
    if visualize:
      plt.imshow(seg)
      plt.imshow(heatmap)
    seg_r = Image.fromarray(seg)
    seg_r.save(os.path.join(save_dir, 'masks', str(i)+'.png'))
    heatmap_r = Image.fromarray(heatmap)
    heatmap_r.save(os.path.join(save_dir, 'heatmaps', str(i) + '.png'))