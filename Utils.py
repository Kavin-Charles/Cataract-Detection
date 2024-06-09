import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from Config import *


#model's unit definitions
class Tools:
    # Defined functions for all the basic tensorflow components that we needed for building a model.
    # function definitions are in the respective comments

    def add_weights(self, shape):
        # a common method to create all sorts of weight connections
        # takes in shapes of previous and new layer as a list e.g. [2,10]
        # starts with random values of that shape.
        return tf.Variable(tf.truncated_normal(shape=shape, stddev=0.05))

    def add_biases(self, shape):
        # a common method to add create biases with default=0.05
        # takes in shape of the current layer e.g. x=10
        return tf.Variable(tf.constant(0.05, shape=shape))

    def conv_layer(self, layer, kernel, input_shape, output_shape, stride_size):
        #convolution occurs here.
        #create weights and biases for the given layer shape
        weights = self.add_weights([kernel, kernel, input_shape, output_shape])
        biases = self.add_biases([output_shape])
        #stride=[image_jump,row_jump,column_jump,color_jump]=[1,1,1,1] mostly
        stride = [1, stride_size, stride_size, 1]
        #does a convolution scan on the given image
        layer = tf.nn.conv2d(layer, weights, strides=stride, padding='SAME') + biases
        return layer

    def pooling_layer(self, layer, kernel_size, stride_size):
        # basically it reduces the complexity involved by only taking the important features alone
        # many types of pooling is there.. average pooling, max pooling..
        # max pooling takes the maximum of the given kernel
        #kernel=[image_jump,rows,columns,depth]
        kernel = [1, kernel_size, kernel_size, 1]
        #stride=[image_jump,row_jump,column_jump,color_jump]=[1,2,2,1] mostly
        stride = [1, stride_size, stride_size, 1]
        return tf.nn.max_pool(layer, ksize=kernel, strides=stride, padding='SAME')

    def flattening_layer(self, layer):
        #make it single dimensional
        input_size = layer.get_shape().as_list()
        new_size = input_size[-1] * input_size[-2] * input_size[-3]
        return tf.reshape(layer, [-1, new_size]), new_size

    def fully_connected_layer(self, layer, input_shape, output_shape):
        #create weights and biases for the given layer shape
        weights = self.add_weights([input_shape, output_shape])
        biases = self.add_biases([output_shape])
        #most important operation
        layer = tf.matmul(layer, weights) + biases  # mX+b
        return layer

    def activation_layer(self, layer):
        # we use Rectified linear unit Relu. it's the standard activation layer used.
        # there are also other layer like sigmoid,tanh..etc. but relu is more efficent.
        # function: 0 if x<0 else x.
        return tf.nn.relu(layer)


import os
import cv2
import random


#tools for image processing and data handing.
class utils:
    image_count = []
    count_buffer = []
    class_buffer = all_classes[:]

    def __init__(self):
        self.image_count = []
        self.count_buffer = []
        for i in os.walk(data_path):
            if len(i[2]):
                self.image_count.append(len(i[2]))
        self.count_buffer = self.image_count[:]

    # processing images into arrays and dispatch as batches whenever called.
    def batch_dispatch(self, batch_size=batch_size):
        global batch_counter
        if sum(self.count_buffer):

            class_name = random.choice(self.class_buffer)
            choice_index = all_classes.index(class_name)
            choice_count = self.count_buffer[choice_index]
            if choice_count == 0:
                class_name = all_classes[self.count_buffer.index(max(self.count_buffer))]
                choice_index = all_classes.index(class_name)
                choice_count = self.count_buffer[choice_index]

            slicer = batch_size if batch_size < choice_count else choice_count
            img_ind = self.image_count[choice_index] - choice_count
            indices = [img_ind, img_ind + slicer]
            images = self.generate_images(class_name, indices)
            labels = self.generate_labels(class_name, slicer)

            self.count_buffer[choice_index] = self.count_buffer[choice_index] - slicer
        else:
            images, labels = (None,) * 2
        return images, labels

    #gives one hot for the respective labels
    def generate_labels(self, class_name, number_of_samples):
        one_hot_labels = [0] * number_of_classes
        one_hot_labels[all_classes.index(class_name)] = 1
        one_hot_labels = [one_hot_labels] * number_of_samples
        #one_hot_labels=tf.one_hot(indices=[all_classes.index(class_name)]*number_of_samples,depth=number_of_classes)
        return one_hot_labels

    # image operations
    def generate_images(self, class_name, indices):
        batch_images = []
        choice_folder = os.path.join(data_path, class_name)
        selected_images = os.listdir(choice_folder)[indices[0]:indices[1]]
        for image in selected_images:
            img = cv2.imread(os.path.join(choice_folder, image))
            batch_images.append(img)
        return batch_images
