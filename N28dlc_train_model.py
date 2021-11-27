import os
import h5py
import time
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from N28dlc_model import get_model
from N28dlc_server import client_generator, BATCH_SIZE

EPOCHS = 50

os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

def gen(hwm, host, port, model):
    for tup in client_generator(hwm=hwm, host=host, port=port):
        X_batch, Y_batch = tup
          #print('#--- X_batch.shape =', X_batch.shape)
          #print('#--- Y_batch.shape =', Y_batch.shape)

          #--- X_batch.shape = (25, 12, 128, 256)
          #--- Y_batch.shape = (25, 112)
        yield X_batch, Y_batch

def custom_loss(y_true, y_pred):
      #print('#---  y_true[0, :] =', y_true[0, :])
      #print('#---  y_pred[0, :] =', y_pred[0, :])
    loss = tf.keras.losses.mse(y_true, y_pred)

    return loss

if __name__=="__main__":
    start = time.time()
    parser = argparse.ArgumentParser(description='Training modelB3')
    parser.add_argument('--host', type=str, default="localhost", help='Data server ip address.')
    parser.add_argument('--port', type=int, default=5557, help='Port of server.')
    parser.add_argument('--port_val', type=int, default=5558, help='Port of server for validation dataset.')
    args = parser.parse_args()

    # Build model
    img_shape = (12, 128, 256)
    num_classes = 6
    model = get_model(img_shape, num_classes)
    #model.summary()

    filepath = "./saved_model/N28dlc_model-BestWeights.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,
                                 save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    #model.load_weights('./saved_model/modelB3-BestWeights.hdf5', by_name=True)
    adam = tf.keras.optimizers.Adam(lr=0.0001)
    model.compile(optimizer=adam, loss=custom_loss)

    history = model.fit(
        gen(20, args.host, port=args.port, model=model),
        steps_per_epoch=32, epochs=EPOCHS,
        validation_data=gen(20, args.host, port=args.port_val, model=model),
        validation_steps=32, verbose=1, callbacks=callbacks_list)
      # steps_per_epoch=1150//BATCH_SIZE = 71  use this for longer training

    model.save('./saved_model/N28dlc_model.h5')

    end = time.time()
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Training Time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

    np.save('./saved_model/N28dlc_model_loss', np.array(history.history['loss']))
    lossnpy = np.load('./saved_model/N28dlc_model_loss.npy')
    plt.plot(lossnpy)  #--- lossnpy.shape = (10,)
    plt.draw() #plt.show()
    plt.pause(0.5)
    input("Press ENTER to exit ...")
    plt.close()
