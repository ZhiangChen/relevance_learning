import keras.backend as K

def mse(y_true, y_pred):
  return K.mean(K.square(y_pred - y_true))


def weighted_L2(y_true, y_pred):
  return K.mean(K.exp(K.clip(y_true, 0, 1) - K.clip(y_pred, 0, 1)) * K.square(y_pred - y_true))