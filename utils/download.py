from s3_functions import *

download_folder_from_s3("flowers/", "flowers/")
download_from_s3("vgg16_weights.npz","vgg16_weights.npz")
download_folder_from_s3("model", "model")