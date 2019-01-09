from model.net import unet
from keras.callbacks import ModelCheckpoint, TensorBoard
import argparse
import os
import glob
from model.data_loading import *
from model.utils import save_images
import numpy as np

ap = argparse.ArgumentParser()

ap.add_argument('-p', '--use_pfile', default=False, dest='use_pfile', help='use predefined parameter file',  type=bool)
ap.add_argument('-m', '--mode', default='train', choices=['train', 'predict', 'test'])
ap.add_argument('-l', '--logs_dir', default='unet_atlanta.hdf5')
ap.add_argument('-b', '--batch_size', default=2, dest='batch_size', type=int)
ap.add_argument('-n', '--num_epochs', default=10, dest='num_epochs', type=int)
ap.add_argument('-tr', '--train_path', default='./training_data', dest='train_path')
ap.add_argument('-va', '--valid_path', default='./validation_data', dest='valid_path')
ap.add_argument('-te', '--test_path', default='./test_data', dest='test_path')
ap.add_argument('-if', '--image_folder', default='images', dest='image_folder')
ap.add_argument('-mf', '--mask_folder', default='masks', dest='mask_folder')
ap.add_argument('-hf', '--heatmap_folder', default='heatmaps', dest='heatmap_folder')
ap.add_argument('-ht', '--target_height', default=256, dest='h', type=int)
ap.add_argument('-wd', '--target_width', default=256, dest='w', type=int)
ap.add_argument('-is', '--image_save_prefix', default='image', dest='image_save_prefix')
ap.add_argument('-ms', '--mask_save_prefix', default='mask', dest='mask_save_prefix')
ap.add_argument('-hs', '--heatmap_save_prefix', default='heatmap', dest='heatmap_save_prefix')
ap.add_argument('-c', '--class_num', default=2, dest='class_num', type=int)
ap.add_argument('-s', '--save_dir', default='../results', dest='save_dir')
args = ap.parse_args()

# data_gen_args = dict(horizontal_flip=True,)
data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
batch_size = 16
train_path = '../scale6_train/multiclass/heatmap_train/train/training'
valid_path = '../scale6_train/multiclass/heatmap_train/train/validation'
test_path = '../scale6_train/multiclass/heatmap_train/train/validation'
image_folder = 'images'
mask_folder = 'masks'
heatmap_folder = 'heatmaps'
class_num = 3
target_size = (256, 256)
# trainGenerator(batch_size, train_path, image_folder, mask_folder, heatmap_folder, aug_dict, image_color_mode="rgb",
#                    mask_color_mode="grayscale", image_save_prefix="image", mask_save_prefix="mask",heatmap_save_prefix="heatmap",
#                    flag_multi_class=True, num_class=2, save_to_dir=None, target_size=(256, 256), seed=1)
if args.mode == 'train':
  if not args.use_pfile:
    myGene = trainGenerator(args.batch_size, args.train_path, args.image_folder,
                            args.mask_folder,
                            args.heatmap_folder, data_gen_args, image_color_mode="rgb",
                            mask_color_mode="grayscale", image_save_prefix="image", mask_save_prefix="mask",
                            heatmap_save_prefix="heatmap",
                            flag_multi_class=True, num_class=args.class_num, save_to_dir=None,
                            target_size=(args.h, args.w), seed=1)
    myValGene = trainGenerator(args.batch_size, args.valid_path, args.image_folder,
                               args.mask_folder,
                               args.heatmap_folder, data_gen_args, image_color_mode="rgb",
                               mask_color_mode="grayscale", image_save_prefix="image", mask_save_prefix="mask",
                               heatmap_save_prefix="heatmap",
                               flag_multi_class=True, num_class=args.class_num, save_to_dir=None,
                               target_size=(args.h, args.w), seed=1)
  else:
    myGene = trainGenerator(batch_size, train_path, image_folder, mask_folder, heatmap_folder, data_gen_args, image_color_mode="rgb",
                     mask_color_mode="grayscale", image_save_prefix="image", mask_save_prefix="mask", heatmap_save_prefix="heatmap",
                     flag_multi_class=True, num_class=class_num, save_to_dir=None, target_size=target_size, seed=1)
    myValGene = trainGenerator(batch_size, valid_path, image_folder, mask_folder, heatmap_folder, data_gen_args,
                            image_color_mode="rgb",
                            mask_color_mode="grayscale", image_save_prefix="image", mask_save_prefix="mask",
                            heatmap_save_prefix="heatmap",
                            flag_multi_class=True, num_class=class_num, save_to_dir=None, target_size=target_size, seed=1)

  try:
    model = unet(num_classes=3, seg_only=True, pretrained_weights=args.logs_dir)
    print('using pretrained data')
  except:
    model = unet(num_classes=3, seg_only=True)
  model_checkpoint = ModelCheckpoint(args.logs_dir, monitor='loss',verbose=1, save_best_only=False)
  tb = TensorBoard(log_dir = './logs', histogram_freq = 0,
    batch_size = 16, write_graph = True, write_grads = False,
    write_images = False, embeddings_freq = 0,
    embeddings_layer_names = None, embeddings_metadata = None,
    embeddings_data = None, update_freq = 160)
  # for i in range(args.num_epochs):
  model.fit_generator(myGene,steps_per_epoch=300,epochs=args.num_epochs, validation_data=myValGene, validation_steps=300, callbacks=[model_checkpoint, tb])
  # eval = model.evaluate_generator(myValGene, steps=10, verbose=1, workers=1, use_multiprocessing=True)
  print(eval)
  model2 = unet(pretrained_weights=args.logs_dir, num_classes=3, seg_only=False)
  # for j in range(args.num_epochs):
  model2.fit_generator(myGene,steps_per_epoch=300,epochs=args.num_epochs, validation_data=myValGene, validation_steps=300,callbacks=[model_checkpoint, tb])
  # eval = model2.evaluate_generator(myValGene, steps=10, verbose=1, workers=1, use_multiprocessing=True)
  print(eval)

elif args.mode == 'test':
  if not args.use_pfile:
    nImages = len(os.listdir(os.path.join(args.test_path, args.image_folder)))
  else:
    nImages = len(os.listdir(os.path.join(test_path, image_folder)))
  if not args.use_pfile:
    myGene = trainGenerator(args.batch_size, args.test_path, args.image_folder,
                            args.mask_folder,
                            args.heatmap_folder, data_gen_args, image_color_mode="rgb",
                            mask_color_mode="grayscale", image_save_prefix="image", mask_save_prefix="mask",
                            heatmap_save_prefix="heatmap",
                            flag_multi_class=True, num_class=args.class_num, save_to_dir=None,
                            target_size=(args.h, args.w), seed=1)
  else:
    myGene = trainGenerator(batch_size, test_path, image_folder, mask_folder, heatmap_folder, data_gen_args, image_color_mode="rgb",
                     mask_color_mode="grayscale", image_save_prefix="image", mask_save_prefix="mask", heatmap_save_prefix="heatmap",
                     flag_multi_class=True, num_class=class_num, save_to_dir=None, target_size=target_size, seed=1)

  try:
    model = unet(num_classes=3, seg_only=True, pretrained_weights=args.logs_dir)
  except:
    raise ValueError('No model at specified dir')
  model_checkpoint = ModelCheckpoint(args.logs_dir, monitor='loss', verbose=1, save_best_only=True)
  eval = model.evaluate_generator(myGene, verbose=1, workers=1, steps=nImages)
  print(eval)

else:
  if not args.use_pfile:
    nImages = len(os.listdir(os.path.join(args.test_path, args.image_folder)))
  else:
    nImages = len(os.listdir(os.path.join(test_path, image_folder)))
  print(nImages)
  if not args.use_pfile:
    myGene = testGenerator(1, args.test_path, args.image_folder)
  else:
    myGene = testGenerator(1, test_path, image_folder)
  model = unet(num_classes=3, seg_only=True)
  try:
    model.load_weights(args.logs_dir)
  except:
    raise ValueError('No model at specified dir')
  output = model.predict_generator(myGene, workers=1, steps=nImages)
  save_images(output, args.save_dir)