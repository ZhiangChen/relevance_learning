from keras.models import *
from keras.layers import *
from keras.optimizers import *
from model.utils import mse


def unet(pretrained_weights=None, input_size=(256, 256, 3), num_classes=2, seg_only=True):
  inputs = Input(input_size)

  # Conv part
  conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
  conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
  pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
  conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
  conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
  pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
  conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
  conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
  pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
  conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
  conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
  drop4 = Dropout(0.5)(conv4)
  pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
  conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
  conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
  drop5 = Dropout(0.5)(conv5)

  # Branch1 -- Transposed conv for segmentation
  up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
  merge6 = concatenate([drop4, up6], axis=3)
  conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
  conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
  up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
  merge7 = concatenate([conv3, up7], axis=3)
  conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
  conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
  up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
  merge8 = concatenate([conv2, up8], axis=3)
  conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
  conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
  up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
  merge9 = concatenate([conv1, up9], axis=3)
  conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
  conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
  # Multiclass segmentation output
  conv9 = Conv2D(num_classes, 3, activation='softmax', padding='same', kernel_initializer='he_normal', name='segmentation')(conv9)

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
    model.compile(optimizer=Adam(lr=1e-4), loss={'segmentation': 'binary_crossentropy', 'heatmap':'mean_squared_error'},
                  loss_weights={'segmentation':1, 'heatmap': 10},
                metrics={'segmentation': 'accuracy', 'heatmap': mse})
  else:
    model.compile(optimizer=Adam(lr=1e-4),
                  loss={'segmentation': 'binary_crossentropy', 'heatmap': 'mean_squared_error'},
                  loss_weights={'segmentation': 1, 'heatmap': 0},
                  metrics={'segmentation': 'accuracy', 'heatmap': mse})

  # model.summary()

  if (pretrained_weights):
    model.load_weights(pretrained_weights)

  return model
