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

def concatenate(camera_files):
  path_files  = [f.replace('yuv', 'pathdata') for f in camera_files]
  radar_files = [f.replace('yuv', 'radardata') for f in camera_files]
  #---  path_files  = ['/home/jinn/dataB/UHD--2018-08-02--08-34-47--32/pathdata.h5']
  #---  radar_files = ['/home/jinn/dataB/UHD--2018-08-02--08-34-47--32/radardata.h5']
  lastidx = 0
  all_frames = []
  path = []
  lead = []

  for cfile, pfile, rfile in zip(camera_files, path_files, radar_files):
  # zip: Traversing Lists in Parallel, https://realpython.com/python-zip-function/
    if os.path.isfile(rfile):
      with h5py.File(pfile, "r") as pf5:
        cf5 = h5py.File(cfile, 'r')
        rf5 = h5py.File(rfile, 'r')
        print("#---datagenB3  cf5['X'].shape       =", cf5['X'].shape)
        print("#---datagenB3  pf5['Path'].shape    =", pf5['Path'].shape)
        print("#---datagenB3  rf5['LeadOne'].shape =", rf5['LeadOne'].shape)

        Path = pf5['Path'][:]
        path.append(Path)

        leadOne = rf5['LeadOne'][:]
        lead.append(leadOne)

        #---  cf5['X'].shape       = (1200, 6, 128, 256)
        #Xcf5 = cf5['X']           # this gives different print('#---  all_frames =', all_frames)
        #Xcf5 = cf5['X'][0:1150]   # this gives different print('#---  all_frames =', all_frames)
        # Conclusion: make cf5['X'].shape = (1150, 6, 128, 256)
        Xcf5 = cf5['X']           # this gives different print('#---  all_frames =', all_frames)
        #print("#---datagenB3  Xcf5.shape =", Xcf5.shape)
        all_frames.append([lastidx, lastidx+cf5['X'].shape[0], Xcf5])
        lastidx += Xcf5.shape[0]

  path = np.concatenate(path, axis=0)
  lead = np.concatenate(lead, axis=0)
  #---  path.shape = (1150, 51)
  #---  lead.shape = (1150, 5)
  #---  len(lead) = 1150

  out = np.concatenate((path, lead), axis=-1)
  out[:, 50:52]/=10
  #print('  #---  out[:, 50:52] =', out[:, 50:52])
  print("#---datagenB3  total training %d frames" % (lastidx))

  print('#---datagenB3  all_frames =', all_frames)
  #print('#---datagenB3  np.shape(all_frames) =', np.shape(all_frames))
  print('#---datagenB3  lastidx =', lastidx)
  print('#---datagenB3  out.shape =', out.shape)
  #---  np.shape(all_frames) = (1, 3)
  #---  all_frames = [[0, 1150, <HDF5 dataset "X": shape (1150, 160, 320, 3), type "<f4">]]
  #---  all_frames = [[0, 1150, <HDF5 dataset "X": shape (1150, 6, 128, 256), type "<f4">]]
  #---  lastidx = 1150
  #---  out.shape = (1150, 56)
  return all_frames, lastidx, out

def datagen(camera_files, max_time_len, batch_size):  # I did not use max_time_len
  all_frames, lastidx, out = concatenate(camera_files)

  X_batch = np.zeros((batch_size, 12, 128, 256), dtype='float32')
  Y_batch = np.zeros((batch_size, 2, 51+5), dtype='float32')
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
          print('#---datagenB3  fl, el, ri =', fl, el, ri)
          print('#---datagenB3  ri-fl =', ri-fl)
          #---  fl, el, ri = 0 1150 345
          #---  ri-fl = 345
          vsX1 = Xcf5[ri-fl]
          vsX2 = Xcf5[ri-fl+1]
          #---  vsX2.shape = (6, 128, 256)
          X_batch[count] = np.vstack((vsX1, vsX2))
          #---  X_batch[count].shape = (12, 128, 256)

          vsO1 = out[ri-fl]
          vsO2 = out[ri-fl+1]
          #---  vsO2.shape = (56,)
          Y_batch[count] = np.vstack((vsO1, vsO2))
          #---  Y_batch[count].shape = (2, 56)
          break

      print('#---  count =', count)
      count += 1

    t2 = time.time()
    print('#---datagenB3  datagen time =', "%5.2f ms" % ((t2-t)*1000.0))
    print('#---datagenB3  X_batch.shape =', X_batch.shape)
    print('#---datagenB3  Y_batch.shape =', Y_batch.shape)

    #---  X_batch.shape = (25, 12, 128, 256)
    #---  Y_batch.shape = (25, 2, 56)
    yield(X_batch, Y_batch)
    
    
    
