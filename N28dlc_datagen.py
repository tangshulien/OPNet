import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

def datagen(camera_files, batch_size):
    X_batch = np.zeros((batch_size, 12, 128, 256), dtype='float32')   # for YUV imgs
    Y_batch = np.zeros((batch_size, 2383), dtype='float32')
    path_files  = [f.replace('yuv', 'pathdata') for f in camera_files]
    radar_files = [f.replace('yuv', 'radardata') for f in camera_files]
      #---  path_files  = ['/home/tom/dataB/UHD--2018-08-02--08-34-47--32/pathdata.h5']
      #---  radar_files = ['/home/tom/dataB/UHD--2018-08-02--08-34-47--32/radardata.h5']
    for cfile, pfile, rfile in zip(camera_files, path_files, radar_files):
        if os.path.isfile(cfile) and os.path.isfile(pfile) and os.path.isfile(rfile):
            print('#---serverB3  OK: cfile, pfile, rfile exist')
        else:
            print('#---serverB3  Error: cfile, pfile, or rfile does not exist')
    batchIndx = 0
    Nplot = 0

    while True:
        for cfile, pfile, rfile in zip(camera_files, path_files, radar_files):
            with h5py.File(cfile, "r") as cf5:
                pf5 = h5py.File(pfile, 'r')
                rf5 = h5py.File(rfile, 'r')
                  #---  cf5['X'].shape       = (1150, 6, 128, 256)
                  #---  pf5['Path'].shape    = (1150, 51)
                  #---  rf5['LeadOne'].shape = (1150, 5)
                cf5X = cf5['X']
                
                pf5Pa = pf5['Path']
                print("#*** pf5Pa.shape =  ", pf5Pa.shape)
                #*** pf5Pa.shape =   (1150, 51)
                path_space = np.zeros([1150, 334])
                print("#*** path_space.shape =  ", path_space.shape)
                #*** path_space.shape =   (1150, 334)
                pf5P = np.append(pf5Pa, path_space, axis=1)
                print("#*** pf5p.shape =  ", pf5P.shape)
                #*** pf5p.shape =   (1150, 385)
                print("#*** pf5p [114][0] =  ", pf5P[114][0])
                print("#*** pf5p [114][19] =  ", pf5P[114][19])
                print("#*** pf5p [114][38] =  ", pf5P[114][38])
                print("#*** pf5p [114][48] =  ", pf5P[114][48])
                print("#*** pf5p [114][49] =  ", pf5P[114][49])
                print("#*** pf5p [114][50] =  ", pf5P[114][50])
                print("#*** pf5p [114][51] =  ", pf5P[114][51])
                print("#*** pf5p [114][52] =  ", pf5P[114][52])
                print("#*** pf5p [114][152] =  ", pf5P[114][152])
                print("#*** pf5p [114][384] =  ", pf5P[114][384])
		#*** pf5p [0][0] =   0.01659233309328556
		#*** pf5p [0][48] =   0.6438997983932495
		#*** pf5p [0][50] =   0.08100473880767822
		#*** pf5p [0][51] =   0.0
		#*** pf5p [0][152] =   0.0
		#*** pf5p [0][384] =   0.0
               
                rf5La = rf5['LeadOne']
                print("#*** rf5L.shape =  ", rf5La.shape)
                #*** rf5L.shape =   (1150, 5)
                lead_space = np.zeros([1150, 53])
                print("#*** lead_space.shape =  ", lead_space.shape)
                #*** lead_space.shape =   (1150, 53)
                rf5L = np.append(rf5La, lead_space, axis=1)
                print("#*** rf5L.shape =  ", rf5L.shape)
                #*** rf5L.shape =   (1150, 58) 
                print("#***  rf5L[114][0] =  ", rf5L[114][0])
                print("#***  rf5L[114][1] =  ", rf5L[114][1])
                print("#***  rf5L[114][2] =  ", rf5L[114][2])
                print("#***  rf5L[114][3] =  ", rf5L[114][3])
                print("#***  rf5L[114][4] =  ", rf5L[114][4])
                print("#***  rf5L[114][5] =  ", rf5L[114][5])
                print("#***  rf5L[114][6] =  ", rf5L[114][6])
                print("#***  rf5L[114][15] =  ", rf5L[114][15])
                print("#***  rf5L[114][25] =  ", rf5L[114][25])
                print("#***  rf5L[114][35] =  ", rf5L[114][35])
                print("#***  rf5L[114][45] =  ", rf5L[114][45])
                print("#***  rf5L[114][57] =  ", rf5L[114][57])
                
                




                              
                space = []
                  #---  cf5X.shape = (1150, 6, 128, 256)

                dataN = len(cf5X)

                count = 0
                while count < batch_size:
                    ri = np.random.randint(0, dataN-2, 1)[-1]   # ri cannot be the last dataN-1
                    print("#---  count, dataN, ri =", count, dataN, ri)
                    for i in range(dataN-1):
                      if ri < dataN-1:
                        vsX1 = cf5X[ri]
                        vsX2 = cf5X[ri+1]
                          #---  vsX2.shape = (6, 128, 256)
                        X_batch[count] = np.vstack((vsX1, vsX2))
                          #---  X_batch[count].shape = (12, 128, 256)

                        pf5P1 = pf5P[ri] 
                        pf5P2 = pf5P[ri+1]
                        rf5L1 = rf5L[ri]
                        rf5L2 = rf5L[ri+1]
                        space = (1497,)
                        space = np.zeros(space)
                        print("#*** pf5P1.shape  =", pf5P1.shape)
                        #*** pf5P1.shape  = (385,)
                        print("#*** pf5P2.shape  =", pf5P2.shape)
                        #*** pf5P2.shape  = (385,)
                        print("#*** rf5L1.shape  =", rf5L1.shape)
                        #*** rf5L1.shape  = (58,)
                        print("#*** rf5L2.shape  =", rf5L2.shape)
                        #*** rf5L2.shape  = (58,)
                        print("#*** space.shape  =", space.shape)
                        #*** space.shape  = (1497,)

                        Y_batch[count] = np.hstack((pf5P1, rf5L1, pf5P2, rf5L2, space))#, space
                          #---  Y_batch[count].shape = (112,)
                        break
                    count += 1

                batchIndx += 1
                print('#---  batchIndx =', batchIndx)

                if Nplot == 0:
                    Yb = Y_batch[0][:]
                    print('#---datagenB3  Yb.shape =', Yb.shape)
                    print('#---datagenB3  Y_batch[0][50:52] =', Y_batch[0][50:52])
                    print('#---datagenB3  Y_batch[0][106:108] =', Y_batch[0][106:108])
                    plt.plot(Yb)
                    plt.show()
                    Nplot += 1

                Y_batch[:, 50:52]/=100   # normalize 50 (valid_len), 51 (lcar's d)
                Y_batch[:, 106:108]/=100

                if Nplot == 1:
                    Yb = Y_batch[0][:]
                    print('#---datagenB3  Y_batch[0][50:52] =', Y_batch[0][50:52])
                    print('#---datagenB3  Y_batch[0][106:108] =', Y_batch[0][106:108])
                    plt.plot(Yb)
                    plt.show()
                    Nplot += 1

                  #print('#---datagenB3  X_batch.shape =', X_batch.shape)
                  #print('#---datagenB3  Y_batch.shape =', Y_batch.shape)
                  #---  X_batch.shape = (25, 12, 128, 256)
                  #---  Y_batch.shape = (25, 112)
                yield(X_batch, Y_batch)
