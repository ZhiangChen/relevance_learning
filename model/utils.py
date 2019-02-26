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


def focal_L2(y_true, y_pred, gamma=2., alpha=.75):
  pt_1 = tf.where(tf.greater(y_true, 0.1), y_pred, tf.ones_like(y_pred))
  pt_t = tf.where(tf.greater(y_true, 0.1), y_pred, y_true)
  pt_0 = tf.where(tf.less_equal(y_true, 0.1), y_pred, tf.zeros_like(y_pred))
  pt_f = tf.where(tf.less_equal(y_true, 0.1), y_pred, y_true)
  return K.mean(alpha * K.pow(1. - pt_1, gamma) * K.square(pt_t - y_true) +
    (1 - alpha) * K.pow(pt_0, gamma) * K.square(pt_f - y_true))


def weighted_L2(y_true, y_pred):
  return K.sum(tf.multiply(K.exp(3 *(K.clip(y_true, 0, 1) - K.clip(y_pred, 0, 1))), K.square(y_pred - y_true)))


def hm_recall(y_true, y_pred):
  pt_t = tf.where(tf.greater(y_true, 1e-4), y_pred, tf.zeros_like(y_pred))
  return K.switch(K.equal(K.sum(y_true), 0), 0.0, lambda: K.sum(K.clip(pt_t, 0, 1)) / K.sum(y_true))
  # if K.sum(y_true) == 0:
  #   return 0
  # else:
  #   return K.sum(K.clip(pt_t, 0, 1)) / K.sum(y_true)
    # return K.sum(y_true)


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
    heatmap = (255 * np.clip(heatmap, 0, 1)).astype(np.uint8)
    if visualize:
      plt.imshow(seg)
      plt.imshow(heatmap)
    seg_r = Image.fromarray(seg)
    seg_r.save(os.path.join(save_dir, 'masks', str(i)+'.png'))
    heatmap_r = Image.fromarray(heatmap)
    heatmap_r.save(os.path.join(save_dir, 'heatmaps', str(i) + '.png'))


def save_regs(net_out, save_dir, visualize=False):
  if not os.path.exists(os.path.join(save_dir, 'heatmaps')):
    os.makedirs(os.path.join(save_dir, 'heatmaps'))
  heatmaps = net_out
  N = heatmaps.shape[0]
  for i in range(N):
    heatmap = heatmaps[i, ..., 0]
    heatmap = (255 * np.clip(heatmap, 0, 1)).astype(np.uint8)
    if visualize:
      plt.imshow(heatmap)
    heatmap_r = Image.fromarray(heatmap)
    heatmap_r.save(os.path.join(save_dir, 'heatmaps', str(i) + '.png'))