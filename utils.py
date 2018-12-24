import numpy as np
import math


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

  lon_deg_next = (im_x + 1) / n * 360.0 - 180.0
  lat_rad_next = math.atan(math.sinh(math.pi * (1 - 2 * (im_y + 1) / n)))
  lat_deg_next = math.degrees(lat_rad_next)

  gps[:, 0] = lat_deg * (256 - cent[:, 0])/256 + lat_deg_next * cent[:, 0]/256
  gps[:, 1] = lon_deg * (256 - cent[:, 1])/256 + lon_deg_next * cent[:, 1]/256

  return gps