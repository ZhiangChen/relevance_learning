import keras.backend as K
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf

def mse(y_true, y_pred):
  return K.mean(K.square(y_pred - y_true))


def focal(y_true, y_pred, gamma=2., alpha=.25):
  pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
  pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
  return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.sum(
    (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))


def focal_L2(y_true, y_pred, gamma=2., alpha=.25):
  pt_1 = tf.where(tf.greater(y_true, 0.1), y_pred, tf.ones_like(y_pred))
  pt_0 = tf.where(tf.less_equal(y_true, 0.1), y_pred, tf.zeros_like(y_pred))
  return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.sum(
    (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))


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
    seg = (np.argmax(segs[i, ...], axis=2) * 85).astype(np.uint8)
    heatmap = heatmaps[i, ..., 0]
    heatmap = (255 * heatmap).astype(np.uint8)
    if visualize:
      plt.imshow(seg)
      plt.imshow(heatmap)
    seg_r = Image.fromarray(seg)
    seg_r.save(os.path.join(save_dir, 'masks', str(i)+'.png'))
    heatmap_r = Image.fromarray(heatmap)
    heatmap_r.save(os.path.join(save_dir, 'heatmaps', str(i) + '.png'))