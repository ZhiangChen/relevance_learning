from keras.preprocessing.image import *
import numpy as np
import os
import glob
import skimage.io as io
import skimage.transform as trans
import itertools


def adjustData(img, mask, heatmap, flag_multi_class, num_class):
  # Normalize images
  if np.max(img) > 1:
    img = img / 255.0
  # mask to one-hot
  mask = mask[..., 0]
  if flag_multi_class:
    # print(mask.shape)
    new_mask = np.zeros(mask.shape + (num_class,))
    for i in range(num_class):
      new_mask[mask == i, i] = 1
    mask = new_mask
  # heatmap to [0, 1]
  heatmap = np.clip(heatmap, 0.0, 1.0)

  return img, mask, heatmap


def trainGenerator(batch_size, train_path, image_folder, mask_folder, heatmap_folder, aug_dict, image_color_mode="rgb",
                   mask_color_mode="grayscale", image_save_prefix="image", mask_save_prefix="mask",heatmap_save_prefix="heatmap",
                   flag_multi_class=True, num_class=2, save_to_dir=None, target_size=(256, 256), seed=1):
  '''
  can generate image and mask at the same time
  use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
  if you want to visualize the results of generator, set save_to_dir = "your path"
  '''
  image_datagen = ImageDataGenerator(**aug_dict)
  mask_datagen = ImageDataGenerator(**aug_dict)
  heatmap_datagen = ImageDataGenerator(**aug_dict)
  image_generator = image_datagen.flow_from_directory(
    train_path,
    classes=[image_folder],
    class_mode=None,
    color_mode=image_color_mode,
    target_size=target_size,
    batch_size=batch_size,
    save_to_dir=save_to_dir,
    save_prefix=image_save_prefix,
    seed=seed)
  mask_generator = mask_datagen.flow_from_directory(
    train_path,
    classes=[mask_folder],
    class_mode=None,
    color_mode=mask_color_mode,
    target_size=target_size,
    batch_size=batch_size,
    save_to_dir=save_to_dir,
    save_prefix=mask_save_prefix,
    seed=seed)
  heatmap_generator = heatmap_datagen.flow_from_directory(
    train_path,
    classes=[heatmap_folder],
    class_mode=None,
    color_mode='grayscale',
    target_size=target_size,
    batch_size=batch_size,
    save_to_dir=save_to_dir,
    save_prefix=heatmap_save_prefix,
    seed=seed)
  try:
    train_generator = itertools.izip(image_generator, mask_generator, heatmap_generator)
    # print('working with izip')
  except:
    train_generator = zip(image_generator, mask_generator, heatmap_generator)
  # print('trying to adjust data')
  for (img, mask, heatmap) in train_generator:
    img, mask, heatmap = adjustData(img, mask, heatmap, flag_multi_class, num_class)
    # yield (img, [mask, heatmap])
    yield (img, mask)


def testGenerator(test_path, num_image=30, target_size=(256, 256), flag_multi_class=False, as_gray=False):
    for i in range(num_image):
      img = io.imread(os.path.join(test_path, "%d.png" % i), as_gray=as_gray)
      img = img / 255.0
      img = trans.resize(img, target_size)
      img = np.reshape(img, img.shape + (1,)) if (not flag_multi_class) else img
      img = np.reshape(img, (1,) + img.shape)
      yield img
