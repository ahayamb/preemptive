import numpy as np
from keras.utils import Sequence


class DataTransformerGenerator(Sequence):

    def __init__(self, x, y, batch_size, transformer=None, shuffled=False):
        self.__x = x
        self.__y = y
        self.__transformer = transformer
        self.__batch_size = batch_size
        self.__shuffled = shuffled
    
    def __len__(self):
        length = len(self.__x)
        return int(np.ceil(length / self.__batch_size))
    
    def __getitem__(self, idx):
        lower_bound = idx * self.__batch_size
        upper_bound = (idx + 1) * self.__batch_size
        batch_x = self.__transformer(self.__x[lower_bound:upper_bound]) if self.__transformer else self.__x[lower_bound:upper_bound]
        batch_y = self.__y[lower_bound:upper_bound]

        return batch_x, batch_y
    
    def on_epoch_end(self):
        if self.__shuffled:
            idx = np.arange(len(self.__x))
            np.random.shuffle(idx)
            self.__x = self.__x[idx]
            self.__y = self.__y[idx]
