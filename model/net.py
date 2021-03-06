from keras.models import *
from keras.layers import *
from keras.optimizers import *
from model.utils import mse, focal_L2, weighted_L2, hm_recall
import os


def unet(pretrained_weights=None, input_size=(256, 256, 3), num_classes=2, seg_only=True, layers_trainable=True):
  inputs = Input(input_size, name='input')

  # Conv part
  conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', trainable=layers_trainable)(inputs)
  conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', trainable=layers_trainable)(conv1)
  pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
  conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', trainable=layers_trainable)(pool1)
  conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', trainable=layers_trainable)(conv2)
  pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
  conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal', trainable=layers_trainable)(pool2)
  conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal', trainable=layers_trainable)(conv3)
  pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
  conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal', trainable=layers_trainable)(pool3)
  conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal', trainable=layers_trainable)(conv4)
  if layers_trainable:
    drop4 = Dropout(0.5)(conv4)
  else:
    drop4 = Dropout(0)(conv4)
  pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
  conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal', trainable=layers_trainable)(pool4)
  conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal', trainable=layers_trainable)(conv5)
  if layers_trainable:
    drop5 = Dropout(0.5)(conv5)
  else:
    drop5 = Dropout(0)(conv5)

  # Branch1 -- Transposed conv for segmentation
  up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal', trainable=layers_trainable)(UpSampling2D(size=(2, 2))(drop5))
  merge6 = concatenate([drop4, up6], axis=3)
  conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal', trainable=layers_trainable)(merge6)
  conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal', trainable=layers_trainable)(conv6)
  up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal', trainable=layers_trainable)(UpSampling2D(size=(2, 2))(conv6))
  merge7 = concatenate([conv3, up7], axis=3)
  conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal', trainable=layers_trainable)(merge7)
  conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal', trainable=layers_trainable)(conv7)
  up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal', trainable=layers_trainable)(UpSampling2D(size=(2, 2))(conv7))
  merge8 = concatenate([conv2, up8], axis=3)
  conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', trainable=layers_trainable)(merge8)
  conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', trainable=layers_trainable)(conv8)
  up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal', trainable=layers_trainable)(UpSampling2D(size=(2, 2))(conv8))
  merge9 = concatenate([conv1, up9], axis=3)
  conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', trainable=layers_trainable)(merge9)
  conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', trainable=layers_trainable)(conv9)
  # Multiclass segmentation output
  conv9 = Conv2D(num_classes, 3, activation='softmax', padding='same', kernel_initializer='he_normal', name='segmentation', trainable=layers_trainable)(conv9)

  # Branch2 -- Transposed conv for heatmap generation
  up6_2 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
  merge6_2 = concatenate([up6, up6_2], axis=3)
  conv6_2 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6_2)
  conv6_2 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6_2)
  up7_2 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6_2))
  merge7_2 = concatenate([up7, up7_2])
  conv7_2 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7_2)
  conv7_2 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7_2)
  up8_2 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7_2))
  merge8_2 = concatenate([up8, up8_2], axis=3)
  conv8_2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8_2)
  conv8_2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8_2)
  up9_2 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8_2))
  merge9_2 = concatenate([up9, up9_2], axis=3)
  conv9_2 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9_2)
  conv9_2 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9_2)
  # Heatmap output
  conv9_2 = Conv2D(1, 3, activation='linear', padding='same', kernel_initializer='he_normal', name='heatmap')(conv9_2)

  model = Model(inputs=inputs, output=[conv9, conv9_2])
  # model = Model(inputs=inputs, output=conv9)
  # model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy',
  #               metrics=['accuracy'])
  if not seg_only:
    model.compile(optimizer=Adam(lr=1e-4), loss={'segmentation': 'binary_crossentropy', 'heatmap':weighted_L2},
                  loss_weights={'segmentation':0, 'heatmap': 1},
                metrics={'segmentation': 'accuracy', 'heatmap': hm_recall})
  else:
    model.compile(optimizer=Adam(lr=1e-4),
                  loss={'segmentation': 'binary_crossentropy', 'heatmap': weighted_L2},
                  loss_weights={'segmentation': 1, 'heatmap': 0},
                  metrics={'segmentation': 'accuracy', 'heatmap': hm_recall})

  # model.summary()

  if (pretrained_weights) and os.path.exists(pretrained_weights):
    print('loading weights '+pretrained_weights)
    model.load_weights(pretrained_weights)

  return model


def regnet(pretrained_weights=None, input_size=(256, 256, 3), layers_trainable=True):
  inputs = Input(input_size, name='input')

  # Conv part
  conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', trainable=layers_trainable)(inputs)
  conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', trainable=layers_trainable)(conv1)
  pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
  conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', trainable=layers_trainable)(pool1)
  conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', trainable=layers_trainable)(conv2)
  pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
  conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal', trainable=layers_trainable)(pool2)
  conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal', trainable=layers_trainable)(conv3)
  pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
  conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal', trainable=layers_trainable)(pool3)
  conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal', trainable=layers_trainable)(conv4)
  if layers_trainable:
    drop4 = Dropout(0.5)(conv4)
  else:
    drop4 = Dropout(0)(conv4)
  pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
  conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal', trainable=layers_trainable)(pool4)
  conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal', trainable=layers_trainable)(conv5)
  if layers_trainable:
    drop5 = Dropout(0.5)(conv5)
  else:
    drop5 = Dropout(0)(conv5)

  # Branch1 -- Transposed conv for segmentation
  up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal', trainable=layers_trainable)(UpSampling2D(size=(2, 2))(drop5))
  merge6 = concatenate([drop4, up6], axis=3)
  conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal', trainable=layers_trainable)(merge6)
  conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal', trainable=layers_trainable)(conv6)
  up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal', trainable=layers_trainable)(UpSampling2D(size=(2, 2))(conv6))
  merge7 = concatenate([conv3, up7], axis=3)
  conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal', trainable=layers_trainable)(merge7)
  conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal', trainable=layers_trainable)(conv7)
  up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal', trainable=layers_trainable)(UpSampling2D(size=(2, 2))(conv7))
  merge8 = concatenate([conv2, up8], axis=3)
  conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', trainable=layers_trainable)(merge8)
  conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', trainable=layers_trainable)(conv8)
  up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal', trainable=layers_trainable)(UpSampling2D(size=(2, 2))(conv8))
  merge9 = concatenate([conv1, up9], axis=3)
  conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', trainable=layers_trainable)(merge9)
  conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', trainable=layers_trainable)(conv9)
  # Multiclass segmentation output
  conv9 = Conv2D(1, 3, activation='linear', padding='same', kernel_initializer='he_normal', name='heatmap', trainable=layers_trainable)(conv9)

  model = Model(inputs=inputs, output=conv9)
  # model = Model(inputs=inputs, output=conv9)
  # model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy',
  #               metrics=['accuracy'])
  model.compile(optimizer=Adam(lr=1e-4), loss={'heatmap':weighted_L2},
              metrics={'heatmap': hm_recall})

  # model.summary()

  if (pretrained_weights) and os.path.exists(pretrained_weights):
    print('loading weights '+pretrained_weights)
    model.load_weights(pretrained_weights)

  return model
