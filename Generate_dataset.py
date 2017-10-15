from Model_interface.Model import Model
import tensorflow as tf
import os
import cv2
import numpy as np
from Dataset_IO.Segmentation.Dataset_reader_segmentation import Dataset_reader_segmentation
from Dataset_IO.Segmentation.Dataset_writer_segmentation import Dataset_writer_segmentation

_BATCH_SIZE_        = 5
_IMAGE_WIDTH_       = 1918
_IMAGE_HEIGHT_      = 1280
_IMAGE_CSPACE_      = 3
_CLASSES_           = 1
_IMAGE_PATH_      = 'G:\\Datasets\\\KaggleResources\\trains'
_MASK_PATH_       = 'G:\\Datasets\\KaggleResources\\train_masks'
_DATASET_NAME_    = 'Testkaggle'
_OUTPUT_FOLDER_   = 'G:\\Datasets'


Seg_writer = Dataset_writer_segmentation(Dataset_filename = _OUTPUT_FOLDER_ + '\\' +_DATASET_NAME_ + '.tfrecords', \
    image_shape=[_IMAGE_WIDTH_, _IMAGE_HEIGHT_, _IMAGE_CSPACE_]) #Initializing Dataset Constructor 

Seg_writer.filename_constructor(image_path = _IMAGE_PATH_, mask_path = _MASK_PATH_) #Feeding the Training images and Training masks

with tf.Session() as sess:
    Seg_writer.write_record() #Writing dataset 
