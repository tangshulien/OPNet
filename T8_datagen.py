"""   YPL & JLL, 2021.9.15, 10.9
Input:
/home/jinn/dataB/UHD--2018-08-02--08-34-47--32/yuv.h5'
/home/jinn/dataB/UHD--2018-08-02--08-34-47--32/pathdata.h5'
/home/jinn/dataB/UHD--2018-08-02--08-34-47--32/radardata.h5'
Output:
X_batch.shape = (25, 12, 128, 256)
Y_batch.shape = (25, 2, 56)
"""
import os
import time
import h5py
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def concatenate(camera_files):
  path_files  = [f.replace('yuv', 'pathdata') for f in camera_files]
  radar_files = [f.replace('yuv', 'radardata') for f in camera_files]
  #---  path_files  = ['/home/jinn/dataB/UHD--2018-08-02--08-34-47--32/pathdata.h5']
  #---  radar_files = ['/home/jinn/dataB/UHD--2018-08-02--08-34-47--32/radardata.h5']
  lastidx = 0
  all_frames = []
  path = []
  long_x = []
  #long_a = []
  lead = []
  #left_lane = []
  #long_v = []
  #desire_state = []
  #right_lane = []  
  #pose = []
  #meta = []
  #desire_pred = []
  
  for cfile, pfile, rfile in zip(camera_files, path_files, radar_files):
  # zip: Traversing Lists in Parallel, https://realpython.com/python-zip-function/
    if os.path.isfile(rfile):
      with h5py.File(pfile, "r") as pf5:
        cf5 = h5py.File(cfile, 'r')
        rf5 = h5py.File(rfile, 'r')
        print("#---datagenB3  cf5['X'].shape       =", cf5['X'].shape)
        #---datagenB3  cf5['X'].shape       = (1150, 6, 128, 256)
        print("#---datagenB3  pf5['Path'].shape    =", pf5['Path'].shape)
        #---datagenB3  pf5['Path'].shape    = (1150, 51)
        print("#---datagenB3  rf5['LeadOne'].shape =", rf5['LeadOne'].shape)
        #---datagenB3  rf5['LeadOne'].shape = (1150, 5)

        Path = pf5['Path'][:]
        path.append(Path)
        leadOne = rf5['LeadOne'][:]      
        lead.append(leadOne)
        Xcf5 = cf5['X'] 
                 
        all_frames.append([lastidx, lastidx+cf5['X'].shape[0], Xcf5])
        lastidx += Xcf5.shape[0]
        
  path = np.concatenate(path, axis=0) #axis=0
  path = np.pad(path, ((0, 0),(0, 142)), 'constant', constant_values=(0,0))
  print("#??? path.shape= ", path.shape) 
  #??? path.shape=  (1150, 193)  
  long_x = (1150, 100)  
  long_x = np.zeros(long_x)  
  print("#??? long_x.shape= ", long_x.shape)
  #??? long_x.shape=  (1150, 100)
  long_a = (1150, 100)
  long_a = np.zeros(long_a) 
  print("#??? long_a.shpe= ", long_x.shape)
  #??? long_a.shape=  (1150, 100)
  lead = np.concatenate(lead, axis=0) #axis=0
  lead = np.pad(lead, ((0, 0),(0, 24)), 'constant', constant_values=(0,0)) 
  print("#??? lead.shape= ", lead.shape)
  #??? lead.shape=  (1150, 29)     
  left_lane = (1150, 193)
  left_lane = np.zeros(left_lane)
  print("#??? left_lane.shape= ", left_lane.shape)
  #??? left_lane.shape=  (1150, 193)            
  long_v = (1150, 100)
  long_v = np.zeros(long_v) 
  print("#??? long_v.shape= ", long_v.shape)  
  #??? long_v.shape=  (1150, 200)
  desire_state = (1150, 4)
  desire_state = np.zeros(desire_state)     
  print("#??? desire_state.shape= ", desire_state.shape) 
  #??? desire_state.shape=  (1150, 8)         
  right_lane = (1150, 193)
  right_lane = np.zeros(right_lane)
  print("#??? right_lane.shape= ", right_lane.shape) 
  #??? right_lane.shape=  (1150, 193)
  residual = (1150, 32)
  residual = np.zeros(residual)
  print("#??? residual.shape= ", residual.shape) 
  #??? residual.shape=  (1150, 32)  
  ''' 
  pose = (1150, 12)
  pose = np.zeros(pose)    
  print("#??? pose.shape= ", pose.shape) 
  #??? pose.shape=  (1150, 12)   
  meta = (1150, 4)
  meta = np.zeros(meta)  
  print("#??? meta.shape= ", meta.shape) 
  #??? meta.shape=  (1150, 4)      
  desire_pred = (1150, 32)
  desire_pred = np.zeros(desire_pred)   
  print("#??? desire_pred.shape= ", desire_pred.shape) 
  #??? desire_pred.shape=  (1150, 32)
  
  out = np.concatenate((path, long_x, long_a ,lead ,left_lane ,long_v ,desire_state ,right_lane ,pose ,meta ,desire_pred), axis=-1)
  '''
  out = np.concatenate((path, lead, long_x, long_a, left_lane, long_v, desire_state, right_lane, residual), axis=-1)
  
  print("#?????out.shape= ", out.shape)
  #?????out.shape=  (1150, 615)  
  out[:, 50:52]/=10
  print('#??????  out[:, 50:52] =', out[:, 50:52].shape)
  #??????  out[:, 50:52] = (1150, 2)
  print("#---datagenB3  total training %d frames" % (lastidx))
  #---datagenB3  total training 1150 frames
  print('#---datagenB3  all_frames =', all_frames)
  #---datagenB3  all_frames = [[0, 1150, <HDF5 dataset "X": shape (1150, 6, 128, 256), type "<f4">]]
  print('#---datagenB3  np.shape(all_frames) =', np.shape(all_frames))
  #---datagenB3  np.shape(all_frames) = (1, 3)
  print('#---datagenB3  lastidx =', lastidx)
  #---datagenB3  lastidx = 1150
  print('#---datagenB3  out.shape =', out.shape)
  #---datagenB3  out.shape = (1150, 1230)
  return all_frames, lastidx, out

def datagen(camera_files, max_time_len, batch_size):  # I did not use max_time_len
  all_frames, lastidx, out = concatenate(camera_files)

  X_batch = np.zeros((batch_size, 12, 128, 256), dtype='float32') # for YUV imgs
  Y_batch = np.zeros((batch_size, 1888), dtype='float32')
  print('#---Y_batch.shape = ', Y_batch.shape)
  #Y_batch.shape =  (25, 1230)

  while True:
    t = time.time()
    count = 0

    while count < batch_size:
      ri = np.random.randint(0, lastidx-1, 1)[-1]
      '''
      random.randint(low=0, high=lastidx, size=1, dtype=int) 
      https://www.sharpsightlabs.com/blog/np-random-randint/
      '''
      for fl, el, Xcf5 in all_frames:
        #---  Xcf5.shape = (1150, 6, 128, 256)

        if fl <= ri and ri < el:
          #print('#---datagenB3  fl, el, ri =', fl, el, ri)
          #print('#---datagenB3  ri-fl =', ri-fl)
          #---  fl, el, ri = 0 1150 345
          #---  ri-fl = 345
          vsX1 = Xcf5[ri-fl]
          vsX2 = Xcf5[ri-fl+1]
          #print('?????vsX1= ', vsX1.shape)
          #print('?????vsX1= ', vsX1.shape)
          #---  vsX2.shape = (6, 128, 256)
          X_batch[count] = np.vstack((vsX1, vsX2))
          #X_batch[count] = np.vstack((vsX1, vsX2))
          #---  X_batch[count].shape = (12, 128, 256)

          vsO1 = out[ri-fl]
          vsO2 = out[ri-fl+1]
          #print('#?????vsO1= ', vsO1)
          #?????vsO1=  [0.01389876 0.03250245 0.05099688 ... 0. 0. 0.]
          print('#?????vsO1.shape= ', vsO1.shape)          
          #???vsO1=  (322,)  
          print('#?????vsO2.shape= ', vsO2.shape)                 
          #???vsO2=  (322,)
          Y_batch[count] = np.append(vsO1, vsO2, axis = 0)
          #Y_batch[count] = np.vstack((vsO1, vsO2))          
          #print('#??? Y_batch[count]=', Y_batch[count])
          #??? Y_batch[count]=[0.00939104 0.0274791 0.04634409...0. 0. 0.] 
          print('#??? Y_batch[count].shape =', Y_batch[count].shape) 
          #??? Y_batch[count].shape = (1230,)            
          #Y_batch[count] = np.concatenate((vsO1, vsO2))      
          #print('#??? Y_batch[count].shape =', Y_batch[count].shape) 
          
          
          break

      print('#---  count =', count)
      #---  count = 24
      count += 1

    t2 = time.time()
    print('#---datagenB3  datagen time =', "%5.2f ms" % ((t2-t)*1000.0))
    #---datagenB3  datagen time = 26.24 ms
    print('#---datagenB3  X_batch.shape =', X_batch.shape)
    #---datagenB3  X_batch.shape = (25, 12, 128, 256)
    print('#---datagenB3  Y_batch.shape =', Y_batch.shape)
    #---datagenB3  Y_batch.shape = (25, 1230)
    yield(X_batch, Y_batch)
    
    
    
