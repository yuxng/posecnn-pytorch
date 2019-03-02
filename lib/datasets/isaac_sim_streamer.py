import socket
import numpy as np
import cv2
from io import BytesIO
import sys
from multiprocessing import Process, Lock, Queue
from multiprocessing.queues import Full
import copy

lock = Lock()

MAX_BUFFER_SIZE = int(4e6) #4 MB
processed_q = Queue(100)


class IsaacSimStreamer(Process):
    def __init__(self, dtypes, shapes, q, host_name='localhost', port_number=8089):
        super(IsaacSimStreamer, self).__init__()
        assert(len(dtypes) == len(shapes))
        self.serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.serversocket.bind(('localhost', 8089))
        self.serversocket.listen(1) # listen to maximum 1 connection
        self.dtypes = dtypes
        self.shapes = shapes
        self.sizes = [len(np.zeros(shape, dtype=dtype).tobytes()) for shape, dtype in zip(self.shapes, self.dtypes)]
        print(self.sizes)
        self.lock = lock
        self.connection = None
        self.q = Queue()
        
        self.buffer = None
        self.current_index = 0
        self.new_item = []
        print('created')
        self.processed_q = q

    
    def get_data(self):
        assert(self.connection is not None), 'connection not initialized yet'
        buf = self.connection.recv(MAX_BUFFER_SIZE)
        if len(buf) == 0:
            return
        self.q.put(copy.deepcopy(buf))
        #print('pushing ', sys.getsizeof(buf), self.q.qsize())


    def process_data(self):
        if self.buffer is None or len(self.buffer) < self.sizes[self.current_index]:
            if self.buffer is None:
                self.buffer = self.q.get()
            else:
                self.buffer += self.q.get()
        
        while len(self.buffer) >= self.sizes[self.current_index]:
            #print('from buffer current_index={}, current_size = {}, buffer_size = {}'.format(self.current_index, self.sizes[self.current_index], len(self.buffer)))
            arr = np.frombuffer(self.buffer[:self.sizes[self.current_index]], dtype=self.dtypes[self.current_index])
            arr = np.reshape(arr, self.shapes[self.current_index])
            self.new_item.append(arr)
            self.buffer = self.buffer[self.sizes[self.current_index]:]
            
            self.current_index += 1
            
            if self.current_index == len(self.sizes):
                try:
                    self.processed_q.put(copy.deepcopy(self.new_item), False)
                    first_time_full = False
                except Full:
                    no_op = True 

                self.current_index = 0
                self.new_item = []


    def run(self):
        print('waiting for connection...')
        self.connection, _ = self.serversocket.accept()
        print('established')
        while True:
            self.get_data()
            self.process_data()

    
    def get_training_data(self):
        while True:
            try:
                data = processed_q.get(True, 1)
                #print('waiting for sim data... qsize = {}'.format(processed_q.qsize()))
                break
            except:
                print('waiting for sim data... qsize = {}'.format(processed_q.qsize()))
                continue
        
        isaac_transform = [[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]]
        isaac_transform = np.asarray(isaac_transform, dtype=np.float32)
        data[2] = np.matmul(isaac_transform, data[2])
        #print(isaac_transform.shape, data[2].shape)
        
        return data
        


if __name__ == '__main__':
    height = 480
    width = 640
    num_objects = 5
    listener = IsaacSimStreamer([np.uint8, np.uint8, np.float32], [[height, width, 3], [height, width], [num_objects, 4, 4]], processed_q)
    listener.start()
    print('here')
    while True:
        data = listener.get_training_data()
        print(data[0].shape)
        cv2.imshow('network image', data[0])
        cv2.waitKey(1000)

    listener.join()
    print('end')






    





'''serversocket.listen(1) # become a server socket, maximum 1 connections


connection, address = serversocket.accept()
print('accepted', connection, address)


total_buf = None
current_index = 0
while True:
    buf = connection.recv(MAX_BUFFER_SIZE)
    if len(buf) == 0:
        continue
    if total_buf is None:
        total_buf = copy.deeepcopy(buf)
    else:
        total_buf += copy.deepcopy(buf)
    print(type(buf))
    print(sys.getsizeof(buf))
    

    #bytes_io = BytesIO(buf)
    #data = np.load(bytes_io).item()
    #print(data.keys())'''


