# Transfer learning in TensorFlow

My attempt on using transfer learning on my dataset and retraining it on the inception_v3 model. 

<center>
![Inception-v3 Architecture](g3doc/inception_v3_architecture.png)
</center>

## Steps on accomplishing the process

### Step 1

Create a dataset of images on which you would retrain the model with. In my instance I gathered images of 3 different flags (philippines, british and america).
Arrange the images in a file format such as the one cited in https://github.com/tensorflow/models/blob/master/inception/README.md 

like this. create both a train and validation directory each containing the classes you need. 

```shell
  $TRAIN_DIR/dog/image0.jpeg
  $TRAIN_DIR/dog/image1.jpg
  $TRAIN_DIR/dog/image2.png
  ...
  $TRAIN_DIR/cat/weird-image.jpeg
  $TRAIN_DIR/cat/my-image.jpeg
  $TRAIN_DIR/cat/my-image.JPG
  ...
  $VALIDATION_DIR/dog/imageA.jpeg
  $VALIDATION_DIR/dog/imageB.jpg
  $VALIDATION_DIR/dog/imageC.png
  ...
  $VALIDATION_DIR/cat/weird-image.PNG
  $VALIDATION_DIR/cat/that-image.jpg
  $VALIDATION_DIR/cat/cat.JPG
  ...
```

and run the build_image_data.py script provided 


```shell

python build_image_data.py \
  --train_directory /home/Desktop/datasets/flags/train/ \
  --validation_directory /home/Desktop/datasets/flags/train \
  --output_directory /home/Desktop/datasets/flags_tfrecords \
  --labels_file /home/Desktop/datasets/flags/labels.txt \
  --train_shards 1 \
  --validation_shards 1 \
  --num_threads 1

```
any of the following arguments can be changed depending on where you stored your images and labels.
this result should produce a bunch of tfrecords with train and validation. 

### Step 2

tfrecords can be checked with the labels file using the script combine_test.py 
but the data_dir location in the script should be changed as well as the number of validation and train images. the num of classes should be adjusted to fit the new dataset. 

To complete the requirments the inception model should be downloaded if training from scratch. But for fine tuning the model the checkpoint for the model need to be downloaded as to know the saved weights of the model.


### Step 3

The models repository could be cloned or forked from https://github.com/tensorflow/models.git

You can approach this through the slim or inception tutorial

I followed the inception tutorial and is able to run the flowers_train on a different dataset. simply change the num_classes in 

```shell

bazel-bin/inception/flowers_train  
--train_dir /home/elements/Desktop/THESIS/transfer learning on tensorflow/train_dir   
--data_dir /home/elements/Desktop/datasets/flags_tfrecords 
--pretrained_model_checkpoint_path /home/elements/Desktop/inception-v3-model/inception-v3/model.ckpt-157585  
--fine_tune=True  
--initial_learning_rate=0.001 
--input_queue_memory_factor=1

```

Following the slim model and its tutorial, the code for train_image_classifier.py can also be used to retrain the inception_v3 model. 


```shell

python train_image_classifier.py 
--train_dir /home/elements/Desktop/THESIS/train_dir 
--dataset_dir /home/elements/Desktop/datasets/flags_tfrecords 
--dataset_split_name=train 
--model_name=inception_v3 
--checkpoint_path /home/elements/Desktop/THESIS/checkpoints/inception_v3.ckpt 
--checkpoint_exclude_scopes=InceptionV3/Logits,InceptionV3/AuxLogits/Logits 
--trainable_scopes=InceptionV3/Logits,InceptionV3/AuxLogits/Logits


```

http://stackoverflow.com/questions/38947658/tensorflow-saving-into-loading-a-graph-from-a-file

http://stackoverflow.com/documentation/tensorflow/5000/save-and-restore-a-model-in-tensorflow#t=201608150606555812336

https://accounts.google.com/o/oauth2/auth?client_id=717762328687-p17pldm5fteklla3nplbss3ai9slta0a.apps.googleusercontent.com&scope=profile+email&redirect_uri=https%3a%2f%2fstackauth.com%2fauth%2foauth2%2fgoogle&state=%7b%22sid%22%3a1%2c%22st%22%3a%228ccf498b1dd9726abb4fc65a9fb48841d2ac8f839a38894253f92439cce9f01d%22%2c%22ses%22%3a%226ca4e02e7ab34b4b8de99685650a0b0e%22%7d&response_type=code

A lot of functions in the scripts provided have been deprecated. SO you will receive errors when using the latest model of tensor flow. but nonetheless the script will still run.

## still updating



