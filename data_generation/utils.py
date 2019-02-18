import numpy as np
import math
import cv2
from sklearn.metrics import pairwise_distances
from skimage.morphology import opening


def cent2gps(cent, im_x, im_y, zoom):
  '''
  Convert a set of points cent(u, v) to lon-lat coordinate
  :param cent: point list
  :param im_x: x dir of img folder
  :param im_y: y img name
  :param zoom: zoom level
  :return:
  '''
  N, d = cent.shape
  if d != 2:
    raise ValueError('Point list not Nx2.')
  gps = np.zeros((N, 2))

  n = 2.0 ** zoom
  lon_deg = im_x / n * 360.0 - 180.0
  lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * im_y / n)))
  lat_deg = math.degrees(lat_rad)

  lon_deg_next = (im_x + 2) / n * 360.0 - 180.0
  lat_rad_next = math.atan(math.sinh(math.pi * (1 - 2 * (im_y + 2) / n)))
  lat_deg_next = math.degrees(lat_rad_next)

  gps[:, 0] = lat_deg * (512 - cent[:, 1])/512 + lat_deg_next * cent[:, 1]/512
  gps[:, 1] = lon_deg * (512 - cent[:, 0])/512 + lon_deg_next * cent[:, 0]/512

  return gps.tolist()


def seg2det(img):

  imgr = img.copy()

  imgr[imgr <= 253] = 0
  ret, thresh = cv2.threshold(imgr, 191, 255, 0)
  # thresh = opening(thresh, selem=np.ones((3, 3)))
  ret, labels = cv2.connectedComponents(thresh)
  labels = labels.astype(np.uint8)

  centroids = []
  bboxes = []
  bboxes_rotated = []
  areas = []

  for i in range(1, int(np.max(labels))+1):

    if np.max(labels) > 10:
      print(np.max(labels))

    ith_obj = labels.copy()
    ith_obj[ith_obj != i] = 0
    if np.sum(ith_obj > 0) < 225:
      continue
    areas.append(np.array([[np.sum(ith_obj > 0)]]))

    hierarchy, contours, _ = cv2.findContours(ith_obj, 1, 2)

    cnt = contours[0]
    M = cv2.moments(cnt)

    centroids.append(np.array([[int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])]]))

    x, y, w, h = cv2.boundingRect(cnt)
    bboxes.append(np.array([[x, y, w, h]]))
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    bboxes_rotated.append(np.int32(box))

  return centroids, bboxes, bboxes_rotated, areas


def deg2num(lat_deg, lon_deg, zoom):
  lat_rad = math.radians(lat_deg)
  n = 2.0 ** zoom
  xtile = int((lon_deg + 180.0) / 360.0 * n)
  ytile = int((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
  return xtile, ytile