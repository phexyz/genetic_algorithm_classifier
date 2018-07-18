import os
import tensorflow as tf
from random import shuffle
from math import floor


def get_file_list_from_dir(datadir, extension=None):

    all_files = os.listdir(datadir)
    if extension == None:
        return list(filter( lambda x: os.path.isdir(os.path.join(datadir, x)), all_files))
    data_files = list(filter(lambda file: file.endswith(extension), all_files))
    return data_files

def tfrecord_auto_traversal():

    directory_to_be_searched = "./flowers/" # The directory to be traversed

    tfrecord_list = get_file_list_from_dir(directory_to_be_searched, ".tfrecord")

    print("files that end with tfrecord", tfrecord_list)

    return tfrecord_list


def split_training_and_testing_sets():

    data_dir = "./flowers/"
    folder_list = get_file_list_from_dir("./flowers/")

    for folder in folder_list:

        file_list = get_file_list_from_dir(os.path.join(data_dir, folder), ".jpg")
        shuffle(file_list)
        split = 0.7
        split_index = int(floor(len(file_list) * split))

        training = file_list[:split_index]
        testing = file_list[split_index:]

        for file in training:

            if not os.path.exists("./train/{0}/".format(folder)):
                os.makedirs("./train/{0}/".format(folder))
            os.rename(os.path.join(data_dir, folder, file), "./train/{0}/{1}".format(folder, file))

        for file in testing:

            if not os.path.exists("./test/{0}/".format(folder)):
                os.makedirs("./test/{0}/".format(folder))

            os.rename(os.path.join(data_dir, folder, file), "./test/{0}/{1}".format(folder, file))


if __name__ == "__main__":
    split_training_and_testing_sets()
