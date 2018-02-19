from collections import deque
import numpy as np
import random


class ReplayMemory( object ):

    def __init__( self, buffer_size, batch_size ):
        self.buffer_size = buffer_size
        self.buffer = deque( maxlen=self.buffer_size )
        self.batch_size = batch_size
        self._size = 0

    def add( self, s1, a1, r1, s2 ):
        experience = ( s1, a1, r1, s2 )
        self.buffer.append( experience )
        self._size += 1
        if self._size > self.buffer_size:
            self._size = self.buffer_size

    def sample( self ):
        batch = []
        count = min( self.batch_size, self._size )
        batch = random.sample( self.buffer, count )

        s1 = np.array( [ arr[0] for arr in batch ] )
        a1 = np.array( [ arr[1] for arr in batch ] )
        r1 = np.array( [ arr[2] for arr in batch ] )
        s2 = np.array( [ arr[3] for arr in batch ] )
        return s1, a1, r1, s2

    def size( self ):
        return self._size

