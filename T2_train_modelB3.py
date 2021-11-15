
"""   YPL & JLL, 2021.9.13, 9.17, 10.11
From /home/jinn/YPN/YPNetB/train_modelB2.py
Train Keras Functional modelB3 = UNet + Pose Net (PN) = opUNetPNB3
1. Use OP supercombo I/O
2. Task: Regression for Path Prediction
3. Pose: Ego path (x, y); lead car's dRel (relative distance), yRel, vRel (velocity), aRel (acceleration), prob
4. Ground Truth: Y_batch = (25, 112)
5. Loss: mean squared error (mse)
6. Input:
   X_batch from yuv.h: 2 YUV images with 6 channels = (1, 12, 128, 256)
   Y_batch from pathdata.h5, radardata.h5
7. Output: mse loss
8. Make yuv.h5 by /home/jinn/openpilot/tools/lib/hevc2yuvh5B3.py
Run:
(YPN) jinn@Liu:~/YPN/OPNet$ python serverB3.py
(YPN) jinn@Liu:~/YPN/OPNet$ python train_modelB3.py
Input:
/home/jinn/dataB/UHD--2018-08-02--08-34-47--32/yuv.h5, pathdata.h5, radardata.h5
/home/jinn/dataB/UHD--2018-08-02--08-34-47--33/yuv.h5, pathdata.h5, radardata.h5
Output:
/OPNet/saved_model/opUNetPNB3_loss.npy
"""
import os
import h5py
import time
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from T2_modelB3 import get_model
from T2_serverB3 import client_generator

os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

def gen(hwm, host, port):
  for tup in client_generator(hwm=hwm, host=host, port=port):
    X, Y = tup
    X_batch = X
    #Y_batch = Y
    Y_batch = Y.reshape(Y.shape[0], -1)
    #print('#--- X_batch.shape =', X_batch.shape)
    #--- X_batch.shape = (25, 12, 128, 256)
    print('#--- Y_batch.shape =', Y_batch.shape)
    #--- Y_batch.shape = (25, 886)
    print('#--- Y.shape[0] =', Y.shape[0])
    #--- Y.shape[0] = 25
    print('#--- Y.shape =', Y.shape)
    #--- Y.shape = (25, 886)

    yield X_batch, Y_batch

if __name__=="__main__":
  # Free up RAM in case the model definition cells were run multiple times
  tf.keras.backend.clear_session()

  start_time = time.time()
  parser = argparse.ArgumentParser(description='Training test_modelB3')
  parser.add_argument('--host', type=str, default="localhost", help='Data server ip address.')
  parser.add_argument('--port', type=int, default=5557, help='Port of server.')
  parser.add_argument('--val_port', type=int, default=5556, help='Port of server for validation dataset.')
  parser.add_argument('--epoch', type=int, default=10, help='Number of epochs.')
  parser.add_argument('--epochsize', type=int, default=1000, help='How many frames per epoch.')
  parser.add_argument('--skipvalidate', dest='skipvalidate', action='store_true', help='Multiple path output.')

  args = parser.parse_args()

  # Build model
  img_shape = (12, 128, 256)
  num_classes = 3
  model = get_model(img_shape, num_classes)
  model.summary()

  filepath = "./saved_model/T2_yuvB3-BestWeights.hdf5"
  checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,
                               save_best_only=True, mode='min')
  callbacks_list = [checkpoint]

  adam = tf.keras.optimizers.Adam(lr=0.0001)
  #model.load_weights('./saved_model/opEffPNb3-BestWeights.hdf5', by_name=True)
  model.compile(optimizer=adam, loss="mae")#mse

  history = model.fit(
    gen(20, args.host, port=args.port),
    steps_per_epoch=8, epochs=5,
    validation_steps=20*1200/25., verbose=1, callbacks=callbacks_list)

  #print('#--- # of epochs are run =', len(history.history['loss']))
  model.save('./saved_model/T2_yuvB3.h5')

  np.save('./saved_model/T2_yuvB3_loss', np.array(history.history['loss']))
  lossnpy = np.load('./saved_model/T2_yuvB3_loss.npy')
  plt.plot(lossnpy)
  plt.draw() #plt.show() # How to stop server
  #print('#--- modelB3 lossnpy.shape =', lossnpy.shape)
  plt.pause(120)#0.5
  input("Press ENTER to exit ...")
  plt.close()
