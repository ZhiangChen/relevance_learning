import keras.backend as K

def mse(y_true, y_pred):
  return K.mean(K.square(y_pred - y_true))