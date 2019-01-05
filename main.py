from model.net import unet
from keras.callbacks import ModelCheckpoint, TensorBoard
import argparse
import os
from model.data_loading import *

ap = argparse.ArgumentParser()

ap.add_argument('-p', '--use_pfile', default=False, dest='use_pfile', help='use predefined parameter file')
ap.add_argument('-m', '--mode', default='train', choices=['train', 'test'])
ap.add_argument('-l', '--logs_dir', default='unet_atlanta.hdf5')
ap.add_argument('-b', '--batch_size', default=2, dest='batch_size')
ap.add_argument('-n', '--num_epochs', default=100, dest='num_epochs')
ap.add_argument('-tr', '--train_path', default='./training_data', dest='train_path')
ap.add_argument('-va', '--valid_path', default='./validation_data', dest='train_path')
ap.add_argument('-te', '--test_path', default='./test_data', dest='test_path')
ap.add_argument('-if', '--image_folder', default='images', dest='image_folder')
ap.add_argument('-mf', '--mask_folder', default='masks', dest='mask_folder')
ap.add_argument('-hf', '--heatmap_folder', default='heatmaps', dest='heatmap_folder')
ap.add_argument('-h', '--target_height', default=256, dest='h')
ap.add_argument('-w', '--target_width', default=256, dest='w')
ap.add_argument('-is', '--image_save_prefix', default='image', dest='image_save_prefix')
ap.add_argument('-ms', '--mask_save_prefix', default='mask', dest='mask_save_prefix')
ap.add_argument('-hs', '--heatmap_save_prefix', default='heatmap', dest='heatmap_save_prefix')
ap.add_argument('-c', '--class_num', default=2, dest='class_num')
args = ap.parse_args()

data_gen_args = dict(horizontal_flip=True,)
batch_size = 30
train_path = './training_data'
valid_path = './validation_data'
test_path = './test_data'
image_folder = 'images'
mask_folder = 'masks'
heatmap_folder = 'heatmaps'
class_num = 3
target_size = (256, 256)
# trainGenerator(batch_size, train_path, image_folder, mask_folder, heatmap_folder, aug_dict, image_color_mode="rgb",
#                    mask_color_mode="grayscale", image_save_prefix="image", mask_save_prefix="mask",heatmap_save_prefix="heatmap",
#                    flag_multi_class=True, num_class=2, save_to_dir=None, target_size=(256, 256), seed=1)
if not args.use_pfile:
  myGene = trainGenerator(batch_size=args.batch_size, train_path=args.train_path, image_folder=args.image_folder, mask_folder=args.mask_folder,
                        heatmap_folder=args.heatmap_folder, aug_dict=data_gen_args, image_color_mode="rgb",
                   mask_color_mode="grayscale", image_save_prefix="image", mask_save_prefix="mask", heatmap_save_prefix="heatmap",
                   flag_multi_class=True, num_class=args.class_num, save_to_dir=None, target_size=(args.h, args.w), seed=1)
  myValGene = trainGenerator(batch_size=args.batch_size, train_path=args.valid_path, image_folder=args.image_folder,
                          mask_folder=args.mask_folder,
                          heatmap_folder=args.heatmap_folder, aug_dict=data_gen_args, image_color_mode="rgb",
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

model = unet(num_classes=3, seg_only=True)
model_checkpoint = ModelCheckpoint(args.logs_dir, monitor='loss',verbose=1, save_best_only=True)
tb = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True,
                            write_grads=False, write_images=True, embeddings_freq=0, embeddings_layer_names=None,
                            embeddings_metadata=None, embeddings_data=None, update_freq='epoch')
for i in range(args.num_epochs):
  model.fit_generator(myGene,steps_per_epoch=300,epochs=1,callbacks=[model_checkpoint])
  eval = model.evaluate_generator(myValGene, steps=10, verbose=1)
  print(eval)
model2 = unet(pretrained_weights=args.logs_dir, num_classes=3, seg_only=False)
for j in range(args.num_epochs):
  model2.fit_generator(myGene,steps_per_epoch=300,epochs=1,callbacks=[model_checkpoint])
  eval = model2.evaluate_generator(myValGene, steps=10, verbose=1)
  print(eval)