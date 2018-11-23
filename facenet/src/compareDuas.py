"""Performs face alignment and calculates L2 distance between the embeddings of images."""

# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import tensorflow as tf
import numpy as np
import sys
import os
import argparse
import time
import facenet
import align.detect_face
import copy

def main(args):
    
    t1 = time.time()
    images = load_data(args.image_files)
    t = time.time() - t1
    print("Load data: {0} seconds".format(t))
    
    with tf.Graph().as_default():
        with tf.Session() as sess:
            
            t = time.time()
            # Load the model
            facenet.load_model("D:/Anaconda3/Restos/models/vggface2.pb")
            t = time.time() - t
            print("Load model: {0} seconds".format(t))
            
            t = time.time()
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            
            t = time.time()
            # Run forward pass to calculate embeddings
            feed_dict = { images_placeholder: images, phase_train_placeholder:False }
            emb = sess.run(embeddings, feed_dict=feed_dict)
            t = time.time() - t
            print("Run forward: {0} seconds".format(t))
            print(emb)
            #filename_base, file_extension = os.path.splitext(args.image_files[0])
           # filename_base = "{}.npy".format(filename_base)
           # np.save(filename_base,emb[0,:])
            #a = np.load(filename_base)

           # filename_base, file_extension = os.path.splitext(args.image_files[1])
           # filename_base = "{}.npy".format(filename_base)
            #np.save(filename_base,emb[1,:])
           # b = np.load(filename_base)
            dist = np.sqrt(np.sum(np.square(np.subtract(emb[0,:], emb[1,:]))))
            #dist = np.sqrt(np.sum(np.square(np.subtract(a,b))))
            if dist <= args.threshold:
                print('True',dist)
            else:
                print('False',dist)
    t = time.time() - t1
    print("Time taken : {0} seconds".format(t))


def load_data(images_path):

    tmp_image_paths = copy.deepcopy(images_path)
    img_list = []
    for image in tmp_image_paths:
        img = misc.imread(os.path.expanduser(image), mode='RGB')
        prewhitened = facenet.prewhiten(img)
        img_list.append(prewhitened)
    

    images = np.stack(img_list)
    return images

      

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('image_files', type=str, nargs='+', help='Images to compare')
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--margin', type=int,
        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=0.7)
    parser.add_argument('--threshold', type=float,
        help='Upper bound of distance between two images to ...  .', default=0.8)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))