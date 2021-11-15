
import os
import zmq
import six
import numpy
import random
import logging
import argparse
#from datagenB3 import datagen
from T2_datagen import datagen
from numpy.lib.format import header_data_from_array_1_0

all_dirs = os.listdir('/home/tom/dataB')
all_yuvs = ['/home/tom/dataB/'+i+'/yuv.h5' for i in all_dirs]
#print('#---  all_yuvs =', all_yuvs)

if six.PY3:
  buffer_ = memoryview
else:
  buffer_ = buffer  # noqa

logger = logging.getLogger(__name__)

random.seed(0) # makes the random numbers predictable
random.shuffle(all_yuvs)

def send_arrays(socket, arrays, stop=False):
  if arrays:
    # The buffer protocol only works on contiguous arrays
    arrays = [numpy.ascontiguousarray(array) for array in arrays]
  if stop:
    headers = {'stop': True}
    socket.send_json(headers)
  else:
    headers = [header_data_from_array_1_0(array) for array in arrays]
    socket.send_json(headers, zmq.SNDMORE)
    for array in arrays[:-1]:
      socket.send(array, zmq.SNDMORE)
    socket.send(arrays[-1])

def recv_arrays(socket):
  headers = socket.recv_json()
  if 'stop' in headers:
    raise StopIteration

  arrays = []
  for header in headers:
    data = socket.recv()
    buf = buffer_(data)
    array = numpy.frombuffer(buf, dtype=numpy.dtype(header['descr']))
    array.shape = header['shape']
    if header['fortran_order']:
      array.shape = header['shape'][::-1]
      array = array.transpose()
    arrays.append(array)
  return arrays

def client_generator(port=5557, host="localhost", hwm=20):
  context = zmq.Context()
  socket = context.socket(zmq.PULL)
  socket.set_hwm(hwm)
  socket.connect("tcp://{}:{}".format(host, port))
  logger.info('client started')
  while True:
    data = recv_arrays(socket)
    yield tuple(data)

def start_server(data_stream, port=5557, hwm=20):
  logging.basicConfig(level='INFO')
  context = zmq.Context()
  socket = context.socket(zmq.PUSH)
  socket.set_hwm(hwm)
  socket.bind('tcp://*:{}'.format(port))

  # it = itertools.tee(data_stream)
  it = data_stream
  logger.info('server started')
  while True:
    try:
      data = next(it)
      stop = False
      logger.debug("sending {} arrays".format(len(data)))
    except StopIteration:
      it = data_stream
      data = None
      stop = True
      logger.debug("sending StopIteration")

    send_arrays(socket, data, stop=stop)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Data Server')
  parser.add_argument('--batch', dest='batch', type=int, default=25, help='Batch size')
  parser.add_argument('--time', dest='time', type=int, default=10, help='Number of frames per sample')
  parser.add_argument('--port', dest='port', type=int, default=5557, help='Port of the ZMQ server')
  parser.add_argument('--buffer', dest='buffer', type=int, default=20, help='High-water mark. Increasing this increses buffer and memory usage.')
  parser.add_argument('--validation', dest='validation', action='store_true', default=False, help='Serve validation dataset instead.')
  args, more = parser.parse_known_args()

  train_len  = int(0.5*len(all_yuvs))
  valid_len  = int(0.5*len(all_yuvs))
  train_files = all_yuvs[: train_len]
  valid_files = all_yuvs[train_len: train_len + valid_len]
  print('#---serverB3  len(all_yuvs) =', len(all_yuvs))
  print('#---serverB3  len(train_files) =', len(train_files))
  print('#---serverB3  len(valid_files) =', len(valid_files))

  if args.validation:
    camera_files = valid_files
  else:
    camera_files = train_files

  print('#---serverB3  camera_files =', camera_files)
  data_s = datagen(camera_files, max_time_len=args.time, batch_size=args.batch)
  start_server(data_s, port=args.port, hwm=args.buffer)
