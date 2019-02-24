from utils import cent2gps, seg2det, deg2num
import os
from PIL import Image
import numpy as np
from skimage.transform import resize


def gen_label(level, ix_lvl, iy_lvl, all_gps, thresh):

  label = np.zeros([256, 256])
  gps_ctr = cent2gps(np.array([[128, 128]]), ix_lvl, iy_lvl, level)

  dists = np.sqrt(np.sum((all_gps - np.asarray(gps_ctr)) ** 2, axis=1))
  if np.min(dists) < thresh:
    gps_ctr_x = cent2gps(np.array([[127, 128]]), ix_lvl, iy_lvl, level)
    gps_ctr_y = cent2gps(np.array([[128, 127]]), ix_lvl, iy_lvl, level)
    x, y = np.meshgrid(np.linspace(0, 255, 256), np.linspace(0, 255, 256))
    gps_map_y = gps_ctr[0][0] * np.ones([256, 256]) - (128 - y) * (gps_ctr[0][0] - gps_ctr_y[0][0])
    gps_map_x = gps_ctr[0][1] * np.ones([256, 256]) - (128 - x) * (gps_ctr[0][1] - gps_ctr_x[0][1])
    h = np.where(dists<thresh)[0]
    Rp = 0.5 + np.log(level - 14) / np.log(3)
    # print(Rp)
    for i in range(h.shape[0]):
      label += np.exp(- ((gps_map_x - all_gps[h[i]][1]) ** 2 + (gps_map_y - all_gps[h[i]][0]) ** 2) / (0.05 * thresh * Rp) ** 2)

    label = label.clip(0, 1)
  return label



if __name__ == '__main__':

  idir = '../../amps_data'

  amp_images = os.listdir(os.path.join(idir, 'images'))
  explored_images = []
  all_gps = []

  for amp_image in amp_images:
    if amp_image in explored_images:
      continue
    explored_images.append(amp_image)
    x, y = amp_image.split('_')[0], amp_image.split('_')[1][:-4]
    x, y = int(x), int(y)
    try:
      label = np.asarray(Image.open(os.path.join(idir, 'annotations', 'label_'+amp_image)))
    except:
      continue
    label_cv = np.zeros([512, 512])
    label_cv[:256, :256] = label
    for i in [0, 1]:
      for j in [0, 1]:
        if i == j == 0:
          continue
        neighbor_img = str(x+i) + '_' + str(y+j) + '.png'
        if neighbor_img in amp_images and neighbor_img not in explored_images:
          explored_images.append(neighbor_img)
          try:
            label = np.asarray(Image.open(os.path.join(idir, 'annotations', 'label_' + neighbor_img)))
          except:
            continue
          label_cv[256*j:256*(j+1), 256*i:256*(i+1)] = label
    label_cv = label_cv.astype(np.uint8)

    centroids, bboxes, bboxes_rotated, areas = seg2det(label_cv)
    cts = np.zeros([len(centroids), 2])
    for i in range(len(centroids)):
      cts[i, :] = centroids[i]
    gps = cent2gps(cts, x, y, 23)

    all_gps += gps

  print(np.asarray(all_gps))

  # levels = [23, 22, 21, 20, 19, 18, 17, 16, 15]
  levels = [18, 19, 20]

  for level in levels:
    levelLock = False
    thresh = 0
    data_dir = os.path.join('F:/Data/tiles', str(level))
    out_dir = '../AA' + str(level)

    x_lvls = os.listdir(data_dir)
    for x_lvl in x_lvls:
      print(level, x_lvl)
      y_names = os.listdir(os.path.join(data_dir, x_lvl))
      if not os.path.exists(os.path.join(out_dir, x_lvl)):
        os.makedirs(os.path.join(out_dir, x_lvl))
      for y_name in y_names:
        # print(x_lvl, y_name)
        ix_lvl = int(x_lvl)
        iy_lvl = int(y_name.split('.')[0])
        if not levelLock:
          thresh = 2 * np.sqrt(np.sum((np.asarray(cent2gps(np.array([[256, 256]]), ix_lvl, iy_lvl, level)) - np.asarray(cent2gps(np.array([[0, 0]]), ix_lvl, iy_lvl, level)))**2))
          levelLock = True
        new_label = gen_label(level, ix_lvl, iy_lvl, all_gps, thresh)

        new_label = (255 * new_label).astype(np.uint8)
        label_f = Image.fromarray(new_label)
        label_f.save(os.path.join(out_dir, x_lvl, y_name))