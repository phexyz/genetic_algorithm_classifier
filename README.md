# Flower Classifier
This is a flower classifier for the kaggle competition. The best accuracy is currently 85%.

## How to use the model
### Install required packages
```
pip -r install requirements.ext
```


### Download dataset and trained model
```
python download.py
```

### Train the model
```
python main.py --train True \
               --test False \
               --epoch number_epoch_to_be_trained \
               --learning_rate learning_rate_in_training \
               --checkpoint_dir directory_to_save_model 
```

### Test the model
```
python main.py --train False \
               --test True \
               --epoch number_epoch_to_be_trained \
               --learning_rate learning_rate_in_training \
               --checkpoint_dir model_directory
```
### Use the model to evaluate images
```
python main.py --input_images_dir input_images'_directory \ 
               --checkpoint_dir model_directory 
```
--input_images_dir flag can receive a directory as well as the path to a single image.

Have fun with flowers!

