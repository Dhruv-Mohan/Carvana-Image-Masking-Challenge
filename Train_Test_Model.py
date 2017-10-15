from Model_interface.Model import Model
import tensorflow as tf
from Dataset_IO.Segmentation.Dataset_reader_segmentation import Dataset_reader_segmentation
from Dataset_IO.Segmentation.Dataset_writer_segmentation import Dataset_writer_segmentation
#from tensorflow.examples.tutorials.mnist import input_data
import os
import cv2
import numpy as np

_SUMMARY_           = True
_BATCH_SIZE_        = 1
_IMAGE_WIDTH_       = 1024
_IMAGE_HEIGHT_      = 1024
_IMAGE_CSPACE_      = 3
_CLASSES_           = 1
_OUTPUT_IMAGE_WIDTH_ = 1918
_OUTPUT_IMAGE_HEIGHT_ = 1280
_OUTPUT_IMAGE_CSPACE_ = 3
_MODEL_NAME_        = 'FRRN_C'
_ITERATIONS_        = 500000
_LEARNING_RATE_     = 0.0005
_SAVE_DIR_          = 'G:/TFmodels/frrncexp'
_SAVE_INTERVAL_     = 5000
_RESTORE_           = False
_TEST_              =  False
_DATASET_PATH_ = 'G:\\Datasets\\KaggleCar\\kaggle.tfrecords'
_TEST_PATH_      = 'G:/Datasets/KaggleResources/test/'
_SUBMISSION_NAME_ = 'frrucsubmission.csv'

def writer_pre_proc_seg_test(images):
    print('Adding image Preproc')
    resized_images = tf.image.resize_images(images, size=[_IMAGE_HEIGHT_,_IMAGE_WIDTH_])
    resized_images=tf.image.per_image_standardization(resized_images)
    #reshaped_images = tf.cast(resized_images,tf.uint8)
    #rgb_image_float = tf.image.convert_image_dtype(resized_images, tf.float32) 
    reshaped_images = tf.reshape(resized_images, [-1,_IMAGE_HEIGHT_*_IMAGE_WIDTH_*_IMAGE_CSPACE_])
    return reshaped_images


def writer_pre_proc(images):
    print('Adding image Preproc')
    resized_images = tf.image.resize_images(images, size=[_IMAGE_HEIGHT_,_IMAGE_WIDTH_])
    reshaped_images = tf.reshape(resized_images, [-1,_IMAGE_HEIGHT_*_IMAGE_WIDTH_*_IMAGE_CSPACE_])
    return reshaped_images

def writer_pre_proc_mask(images):
    print('Adding mask Preproc')
    resized_images = tf.image.resize_images(images, size=[_IMAGE_HEIGHT_,_IMAGE_WIDTH_])
    gray_images = tf.image.rgb_to_grayscale(resized_images)
    reshaped_images = tf.reshape(gray_images, [-1,_IMAGE_HEIGHT_*_IMAGE_WIDTH_])
    return reshaped_images

def writer_pre_proc_weight(images):
    print('Adding weight Preproc')
    images = tf.image.per_image_standardization(images)
    images = tf.expand_dims(images,-1)
    resized_images = tf.image.resize_images(images, size=[_IMAGE_HEIGHT_,_IMAGE_WIDTH_])
    gray_images = (tf.image.rgb_to_grayscale(resized_images)+2)* 5
    gray_images = gray_images/ tf.reduce_mean(gray_images)
    reshaped_images = tf.reshape(gray_images, [-1,_IMAGE_HEIGHT_*_IMAGE_WIDTH_])
    return reshaped_images

def writer_pre_proc_test(image, mean_image):
    recast_image = tf.image.convert_image_dtype(image,tf.float32)
    normalized_image = recast_image - mean_image
    return tf.reshape(normalized_image, [-1,_IMAGE_HEIGHT_*_IMAGE_WIDTH_*_IMAGE_CSPACE_])

def construct_segmap(image):
    image = cv2.resize(image, (_OUTPUT_IMAGE_WIDTH_, _OUTPUT_IMAGE_HEIGHT_), interpolation=cv2.INTER_LANCZOS4)
    ret,thresh = cv2.threshold(image,0.5,1,cv2.THRESH_BINARY)
    return thresh


def rle_encode(mask_image):
    pixels = mask_image.flatten()
    # We avoid issues with '1' at the start or end (at the corners of 
    # the original image) by setting those pixels to '0' explicitly.
    # We do not expect these to be non-zero for an accurate mask, 
    # so this should not harm the score.
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    return runs


def rle_to_string(runs):
    return ' '.join(str(x) for x in runs)



def main():

    with tf.device('/cpu:0'):
        dummy_reader = Dataset_reader_segmentation(_DATASET_PATH_)
        dummy_reader.pre_process_image(writer_pre_proc)
        dummy_reader.pre_process_mask(writer_pre_proc_mask)
        dummy_reader.pre_process_weights(writer_pre_proc_weight)

    init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())

    #Construct model
    with tf.name_scope('FRRN_C'):
        Simple_DNN = Model(Model_name=_MODEL_NAME_, Summary=_SUMMARY_, \
            Batch_size=_BATCH_SIZE_, Image_width=_IMAGE_WIDTH_, Image_height=_IMAGE_HEIGHT_, Image_cspace=_IMAGE_CSPACE_, Classes=_CLASSES_, Save_dir=_SAVE_DIR_)

        Simple_DNN.Construct_Model()


    #Set loss
        Simple_DNN.Set_loss()


    #Set optimizer
        if not _TEST_:
            with tf.name_scope('Train'):
                Simple_DNN.Set_optimizer(starter_learning_rate= _LEARNING_RATE_)


        #Construct op to check accuracy
        Simple_DNN.Construct_Accuracy_op()
        Simple_DNN.Construct_Predict_op()

    #Training/Testing block
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        if _TEST_:
            test_path = _TEST_PATH_
            test_images = os.listdir(test_path)
            test_image_path = tf.placeholder(tf.string)
            test_image = tf.image.decode_image(tf.read_file(test_image_path)) #So, i've read a test image, i need to pre-process it now
            test_image.set_shape([_OUTPUT_IMAGE_WIDTH_, _OUTPUT_IMAGE_HEIGHT_, _OUTPUT_IMAGE_CSPACE_])
            test_image = writer_pre_proc_seg_test(test_image)

    with tf.Session(config=config) as session:
        if _TEST_:
            Simple_DNN.Construct_Writers()
            Simple_DNN.Try_restore()
            total_images = len(test_images)
            with open(_SUBMISSION_NAME_, 'w') as submitfile:
                    submitfile.write('img,rle_mask\n')
                    for index, imag in enumerate(test_images):
                        printProgressBar(index+1, total_images)
                        test_imag = session.run([test_image], feed_dict={test_image_path:os.path.join(test_path,imag)})[0]
                        prediction = Simple_DNN.Predict(test_imag)            
                        seg_map = construct_segmap(np.squeeze(prediction))
                        submitfile.write(imag+','+rle_to_string(rle_encode(seg_map))+'\n')


        if not _TEST_:
            Simple_DNN.Construct_Writers()
            Simple_DNN.Train_Iter(iterations=_ITERATIONS_, save_iterations=_SAVE_INTERVAL_, data=dummy_reader, restore=_RESTORE_)


def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 50, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('       Progress: |%s| %s%% %s' % (bar, percent, suffix), end="\r")
    # Print New Line on Complete


if __name__ == "__main__":
    main()