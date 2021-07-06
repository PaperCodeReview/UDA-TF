import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds

from augment import RandAugment


AUTO = tf.data.experimental.AUTOTUNE


def set_dataset(data_path, dataset):
    if dataset == 'imagenet':
        trainset = pd.read_csv(
            os.path.join(
                data_path, '{}_trainset.csv'.format(dataset)
            )).values.tolist()
        valset = pd.read_csv(
            os.path.join(
                data_path, '{}_valset.csv'.format(dataset)
            )).values.tolist()
        return np.array(trainset, dtype='object'), np.array(valset, dtype='object')

    else:
        assert dataset in ['cifar10', 'svhn'], 'dataset must be selected in cifar10 and svhn.'
        totalset = tfds.load('svhn_cropped' if dataset == 'svhn' else dataset)
        trainset, valset = totalset['train'], totalset['testset']
        return trainset, valset


class DataLoader:
    def __init__(self, args, mode, datalist, batch_size, shuffle=True):
        self.args = args
        self.mode = mode
        self.datalist = datalist
        self.imglist = self.datalist[:,0].tolist()
        self.batch_size = batch_size
        self.shuffle = shuffle

        if self.args.dataset in ['imagenet']:
            self.dataloader = self._dataloader()
        else:
            pass

    def __len__(self):
        return len(self.datalist)

    def fetch_dataset(self, path):
        x = tf.io.read_file(path)
        return tf.data.Dataset.from_tensors(x)

    def augmentation(self, img, shape):
        augset = RandAugment(self.args, self.mode)
        img_list = []
        for _ in range(2): # query, key
            aug_img = tf.identity(img)
            aug_img = augset(aug_img, shape) # moco v1
            img_list.append(aug_img)
        return img_list

    def dataset_parser(self, value):
        shape = tf.image.extract_jpeg_shape(value)
        img = tf.io.decode_jpeg(value, channels=3)
        query, key = self.augmentation(img, shape)
        return {'query': query, 'key': key}
        
    def _dataloader(self):
        if self.args.dataset == 'imagenet':
            dataset = tf.data.Dataset.from_tensor_slices(self.imglist)
        else:
            dataset = self.datalist
        dataset = dataset.repeat()
        if self.shuffle:
            dataset = dataset.shuffle(len(self.datalist))

        dataset = dataset.interleave(self.fetch_dataset, num_parallel_calls=AUTO)
        dataset = dataset.map(self.dataset_parser, num_parallel_calls=AUTO)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(AUTO)
        return dataset