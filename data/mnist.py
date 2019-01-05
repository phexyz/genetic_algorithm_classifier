import numpy as np
import urllib
import gzip
import pickle
import os

class MNIST(object):

    def __init__(self):
        self.filename = [["training_images","train-images-idx3-ubyte.gz"],
                    ["test_images","t10k-images-idx3-ubyte.gz"],
                    ["training_labels","train-labels-idx1-ubyte.gz"],
                    ["test_labels","t10k-labels-idx1-ubyte.gz"]]

        self.data_dir = "./MNIST/"
        self.download_mnist()
        self.save_mnist()

    def download_mnist(self):
        base_url = "http://yann.lecun.com/exdb/mnist/"
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        for name in self.filename:
            print("Downloading "+name[1]+"...")
            urllib.urlretrieve(base_url+name[1], self.data_dir+name[1])
        print("Download complete.")

    def save_mnist(self):
        mnist = {}
        for name in self.filename[:2]:
            with gzip.open(self.data_dir+name[1], 'rb') as f:
                mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1,28,28,1)
        for name in self.filename[-2:]:
            with gzip.open(self.data_dir+name[1], 'rb') as f:
                mnist[name[0]] = self.make_one_hot(np.frombuffer(f.read(), np.uint8, offset=8))
        with open(self.data_dir+"mnist.pkl", 'wb') as f:
            pickle.dump(mnist,f)
        print("Save complete.")

    def load(self):
        """This function loads (training images, training labels, test images, test labels)"""
        with open(self.data_dir+"mnist.pkl",'rb') as f:
            mnist = pickle.load(f)
        return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]

    def load_train_dataset(self):
        """This function loads (training images, training labels)"""
        with open(self.data_dir+"mnist.pkl",'rb') as f:
            mnist = pickle.load(f)
        return mnist["training_images"], mnist["training_labels"]

    def load_test_dataset(self):
        """This function loads (test images, test labels)"""
        with open(self.data_dir+"mnist.pkl",'rb') as f:
            mnist = pickle.load(f)
        return mnist["test_images"], mnist["test_labels"]

    def make_one_hot(self, labels):
        """This function makes one label given a list"""

        return (np.arange(10)==labels[:,None]).astype(np.integer)
