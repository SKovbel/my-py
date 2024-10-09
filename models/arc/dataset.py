import json
import numpy as np
import tensorflow as tf

from config import Config
from sklearn.preprocessing import StandardScaler

class Dataset:
    def __init__(self):
        self.scaler = StandardScaler()
        self.keys = []
        self.data = {}
        self.load()

    def __expand(self, input):
        height, width = min(len(input), Config.HEIGHT), min(len(input[0]), Config.WIDTH)
        output = np.zeros((Config.HEIGHT, Config.WIDTH), dtype=np.int64)
        output[:height, :width] = [row[0:width] for row in input[0:height]]
        return output

    def load(self, train_path='arc-agi_training_challenges.json', test_path='arc-agi_training_solutions.json'):
        with open(f'{Config.DATA_DIR}/{train_path}', 'r') as train_file, open(f'{Config.DATA_DIR}/{test_path}', 'r') as test_file:
            test = json.load(test_file)
            train = json.load(train_file)
            for key in train:
                self.keys.append(key)
                self.data[key] = {
                    'X_train': [],
                    'y_train': [],
                    'X_test': self.__expand(train[key]['test'][0]['input']),
                    'y_test': self.__expand(test[key][0])
                }
                for index in range(0, len(train[key]['train'])):
                    self.data[key]['X_train'].append(self.__expand(train[key]['train'][index]['input']))
                    self.data[key]['y_train'].append(self.__expand(train[key]['train'][index]['output']))

    def samples(self, key_id=0):
        return self.data if key_id is None else self.data[self.keys[key_id]]

    def debug(self, key_id=0, img_id=0):
        print(self.samples(key_id)['X_train'][img_id])

    # Datasets:
    def vector(self, as_tensor=False):
        x, y = [], []
        for key in self.data:
            for sample in self.data[key]['X_train']:
                x.append([element for row in sample for element in row])
            for sample in self.data[key]['y_train']:
                y.append([element for row in sample for element in row])

        if as_tensor:
            return tf.data.Dataset.from_tensor_slices((x, y))
        return np.array(x), np.array(y)

    def series(self, type='vector', pad=True, as_tensor=False):
        x_series, y_series = [], []
        max_deep = 0
        if type == 'vector':
            for key in self.data:
                sub_x = []
                sub_y = []
                for sample in self.data[key]['X_train']:
                    sub_x.append([element for row in sample for element in row])
                for sample in self.data[key]['y_train']:
                    sub_y.append([element for row in sample for element in row])
                max_deep = len(sub_x) if len(sub_x) > max_deep else max_deep
                x_series.append(sub_x)  
                y_series.append(sub_y)

        # @todo use another method
        if pad:
            for i in range(len(x_series)):
                for _ in range(max_deep - len(x_series[i])):
                    x_series[i].append(Config.WIDTH * Config.HEIGHT * [0])
                    y_series[i].append(Config.WIDTH * Config.HEIGHT * [0])

        if as_tensor:
            return tf.data.Dataset.from_tensor_slices((x_series, y_series))
        elif pad:
            return np.array(x_series), np.array(y_series)
        return x_series, y_series

    def channels(self, type='whc', as_tensor=False):
        x, y = [], []
        if type == 'whc':
            for key in self.data:
                for sample in self.data[key]['X_train']:
                    x.append((sample[..., None] == np.arange(1, Config.CHANNEL + 1)).astype(int))
                for sample in self.data[key]['y_train']:
                    y.append((sample[..., None] == np.arange(1, Config.CHANNEL + 1)).astype(int))        
        elif type == 'whc0':
            for key in self.data:
                for sample in self.data[key]['X_train']:
                    x.append((sample[..., None] == np.arange(Config.CHANNEL)).astype(int))
                for sample in self.data[key]['y_train']:
                    y.append((sample[..., None] == np.arange(Config.CHANNEL)).astype(int))
        elif type == 'cwh':
            for key in self.data:
                for sample in self.data[key]['X_train']:
                    x.append((sample[None, :, :] == np.arange(1, Config.CHANNEL + 1)[:, None, None]).astype(int))
                for sample in self.data[key]['y_train']:
                    y.append((sample[None, :, :] == np.arange(1, Config.CHANNEL + 1)[:, None, None]).astype(int))

        if as_tensor:
            return tf.data.Dataset.from_tensor_slices((x, y))
        return np.array(x), np.array(y)

    def sparse(self, as_tensor=False):
        #x, y = self.vector()
        x, y = self.series(type='vector')

        x_indices = tf.where(tf.not_equal(x, 0))
        x_values = tf.gather_nd(x, x_indices)

        y_indices = tf.where(tf.not_equal(y, 0))
        y_values = tf.gather_nd(y, y_indices)
        if as_tensor:
            x_sparse = tf.sparse.SparseTensor(x_indices, x_values, tf.shape(x))
            y_sparse = tf.sparse.SparseTensor(y_indices, y_values, tf.shape(y))
            return x_sparse, y_sparse
        return np.column_stack((x_indices.numpy(), x_values.numpy())), np.column_stack((y_indices.numpy(), y_values.numpy()))
        # return (x_indices.numpy(), x_values.numpy()), (y_indices.numpy(), y_values.numpy())
    
    def gausse(self):
        x, y = self.series(type='vector')
        
        x_mean = np.mean(x, axis=2, keepdims=True)
        x_std = np.std(x, axis=2, keepdims=True)
        x = np.where(x_std == 0, x - x_mean, (x - x_mean) / x_std)

        y_mean = np.mean(y, axis=2, keepdims=True)
        y_std = np.std(y, axis=2, keepdims=True)
        y = np.where(y_std == 0, y - y_mean, (y - y_mean) / y_std)

        return np.array(x), np.array(y)

    def hot_stop(self):
        pass

    def tokenizer(self):
        pass


if __name__ == '__main__':
    from plot import Plot

    ds = Dataset()
    dataset_vector = ds.vector()
    dataset_series = ds.series()
    dataset_sparse = ds.sparse()
    dataset_gausse = ds.gausse()
    dataset_channel = ds.channels()

    #print(dataset_gausse[0])
    print(dataset_sparse[0][0])

    print('vector', dataset_vector[0].shape, dataset_vector[1].shape)
    print('series', dataset_series[0].shape, dataset_series[1].shape)
    print('sparse', dataset_sparse[0][0].shape, dataset_sparse[0][1].shape, dataset_sparse[1][0].shape, dataset_sparse[1][1].shape)   
    print('gausse', dataset_gausse[0].shape, dataset_gausse[1].shape)
    print('channel', dataset_channel[0].shape, dataset_channel[1].shape)

    exit(0)
    plot = Plot()
    plot.plot_samples(ds.samples(key_id=0))
    plot.plot_sample(ds.samples(key_id=0), img_id=0)
    plot.plot_test(ds.samples(key_id=0))
