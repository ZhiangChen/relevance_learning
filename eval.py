import numpy as np
from scipy import optimize, ndimage
from skimage.measure import block_reduce
import matplotlib.pyplot as plt
import os
from PIL import Image


def gaussian(height, center_x, center_y, width_x, width_y):
  """Returns a gaussian function with the given parameters"""
  width_x = float(width_x)
  width_y = float(width_y)
  return lambda x, y: height * np.exp(
    -(((center_x - x) / width_x) ** 2 + ((center_y - y) / width_y) ** 2) / 2)

def fitGaussian2D(x):
  def gaussian(height, center_x, center_y, width_x, width_y):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)
    return lambda x, y: height * np.exp(
      -(((center_x - x) / width_x) ** 2 + ((center_y - y) / width_y) ** 2) / 2)


  def moments(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    total = data.sum()
    X, Y = np.indices(data.shape)
    x = (X * data).sum() / total
    y = (Y * data).sum() / total
    col = data[:, int(y)]
    width_x = np.sqrt(np.abs((np.arange(col.size) - y) ** 2 * col).sum() / col.sum())
    row = data[int(x), :]
    width_y = np.sqrt(np.abs((np.arange(row.size) - x) ** 2 * row).sum() / row.sum())
    height = data.max()
    return height, x, y, width_x, width_y


  def fitgaussian(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit"""
    params = moments(data)
    errorfunction = lambda p: np.ravel(gaussian(*p)(*np.indices(data.shape)) -
                                       data)
    p = optimize.leastsq(errorfunction, params)
    (height, x, y, width_x, width_y) = p[0]
    return height, x, y, width_x, width_y

  return fitgaussian(x)


def searchRecall(pred, GT, thr = 0.5):

  targets, hits = 0, 0
  for i in range(GT.shape[0]):
    targets += 1
    for j in range(pred.shape[0]):
      ll = np.exp(-(pred[j, 1] - GT[i, 0]) ** 2 / pred[j,3] ** 2 - (pred[j, 2] - GT[i, 1]) ** 2 / pred[j,4] ** 2)
      if ll > thr:
        hits +=1
        break

  return targets, hits


def decision(hm, split=2, thr=0.5, thr2 = 50):

  H, W = hm.shape
  red = block_reduce(hm > thr2, block_size=(int(H / split), int(W / split)), func=np.mean)

  return red > thr

def decisionEval(pred, GT):

  recall = (np.sum(np.bitwise_and(pred, GT)), np.sum(GT))
  acc = ((np.sum(np.bitwise_and(np.logical_not(pred), np.logical_not(GT))) + np.sum(np.bitwise_and(pred, GT))), GT.size)

  return recall, acc


if __name__ == '__main__':
  # Create the gaussian data
  GT_dir = 'H:/ROBO Master/AerialApp/Winter/Amps_rel/train_18_new/validation/heatmaps'
  pred_dir = 'H:/ROBO Master/AerialApp/Winter/Amps_rel/results_ours/heatmaps'

  GT_images = os.listdir(GT_dir)
  recalls, accs = [0, 0], [0, 0]

  for i in range(len(GT_images)):
    Gi = np.asarray(Image.open(os.path.join(GT_dir, GT_images[i])))
    pi = np.asarray(Image.open(os.path.join(pred_dir, str(i)+'.png')))

    pi_q = pi
    pi_q.setflags(write=True)
    pi_q[pi < 50] = 0
    pi_f = ndimage.gaussian_filter(pi_q, 5)

    des_G, des_p = decision(Gi, split = 4, thr=0.05, thr2 = 25), decision(pi_f, split = 4, thr=0.05, thr2 = 10)
    recall, acc = decisionEval(des_p, des_G)
    recalls[0] += recall[0]
    recalls[1] += recall[1]
    accs[0] += acc[0]
    accs[1] += acc[1]

    # plt.imshow(pi_f)
    # plt.pause(0.5)
    # plt.close()
  print(recalls, accs)