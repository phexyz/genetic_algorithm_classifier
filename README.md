# flower_classifier
This is a flower classifier for the kaggle competition

# Download dataset and trained model
python download.py

# Use model
'''
python main.py --train True/False \
               --test True/False \
               --epoch number_epoch_to_be_trained \
               --learning_rate learning_rate_in_training \
               --checkpoint_dir directory_to_save_model \
               --input_images_dir input_image_path

'''

--input_images_dir flag can receive a directory as well as the path to a single image.

