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
    images_path = []
    for folder in sorted(os.listdir(args.image_folder)):
        for path in sorted(os.listdir(args.image_folder + '/' + folder)):
            images_path.append(args.image_folder + '/' + folder + '/' + path)
    
    #images = load_and_align_data(args.image_files, args.image_size, args.margin, args.gpu_memory_fraction)
    start = time.time()
    images = load_data(images_path, args.gpu_memory_fraction)

    print('Load data completo')
    
    with tf.Graph().as_default():
        # config = tf.ConfigProto()
        # config.gpu_options.per_process_gpu_memory_fraction = 0.1
        # sess = tf.Session(config=config)
        gpu_options = tf.GPUOptions(allow_growth = True)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with tf.Session() as sess:
            
            # Load the model
            facenet.load_model(args.model)
            print('Load model completo')
            
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            print('Get input and output completo')
            
            # Run forward pass to calculate embeddings
            feed_dict = { images_placeholder: images, phase_train_placeholder:False }
            emb = sess.run(embeddings, feed_dict=feed_dict)
            print('Run forward completo')
            
            nrof_images = len(images_path)
            print('Images:', nrof_images)
            for i in range(nrof_images):
                print('%1d: %s' % (i, images_path[i]))
            print('')
            
            nrof_comp = 0
            vp = 0
            vn = 0
            fp = 0
            fn = 0        
            ## Print distance matrix
           # print('Distance matrix')
           # print('    ', end='')
           # for i in range(nrof_images):
           #     print(';%1d' % i, end='')
           # print('')
            for i in range(nrof_images):
           #     print('%1d;' % i, end='')
                for j in range(nrof_images):
                    if j < i:
                        nrof_comp += 1
                        #if nrof_comp%1000000 == 0:
                        #    print(nrof_comp)
                        #    print('')
                        dist = np.sqrt(np.sum(np.square(np.subtract(emb[i,:], emb[j,:]))))
                        image1 = images_path[i].split("/")[-2:-1]
                        image2 = images_path[j].split("/")[-2:-1]
                        if image1 == image2:
                            if dist <= args.threshold:
                    #            print('OK - %1.4f vp;' % dist, end='')
                                vp += 1
                            else:
                     #           print('ERRO - %1.4f dn;' % dist, end='')
                                fn += 1
                        else:
                            if dist > args.threshold:
                      #          print('OK - %1.4f vn;' % dist, end='')
                                vn += 1
                            else:
                     #           print('ERRO - %1.4f fp;' % dist, end='')
                                fp += 1
                        #print('%1.4f' % dist, end='')     
              #   print('')

    print('Threshold:', args.threshold)
    print('Numero de Images:', nrof_images)
    print('Numero de Comparacoes:', nrof_comp)
    print('Acertos:', vp+vn)
    print('Verdadeiro Positivo:', vp)
    print('Verdadeiro Negativo:', vn)
    print('Falso Positivo:', fp)
    print('Falso Negativo:', fn)
    p = ((vp+vn)/nrof_comp)*100
    print('Precisao: %1.4f %%' % p)
    end = time.time()
    seconds = end - start
    print("Time taken : {0} seconds".format(seconds))

      
def load_data(images_path, gpu_memory_fraction):

    tmp_image_paths = copy.deepcopy(images_path)
    img_list = []
    for image in tmp_image_paths:
        img = misc.imread(os.path.expanduser(image), mode='RGB')
        prewhitened = facenet.prewhiten(img)
        img_list.append(prewhitened)
    
    #output_filename_n = image + '.jpg'
                #misc.imsave(output_filename_n, img)

    images = np.stack(img_list)
    return images

            
def load_and_align_data(image_paths, image_size, margin, gpu_memory_fraction):

    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor

    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
  
    #tmp_image_paths = image_paths.copy()

    tmp_image_paths = copy.deepcopy(image_paths)
    img_list = []
    for image in tmp_image_paths:
        img = misc.imread(os.path.expanduser(image), mode='RGB')
        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        if len(bounding_boxes) < 1:
          image_paths.remove(image)
          print("can't detect face, remove ", image)
          continue
        det = np.squeeze(bounding_boxes[0,0:4])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0]-margin/2, 0)
        bb[1] = np.maximum(det[1]-margin/2, 0)
        bb[2] = np.minimum(det[2]+margin/2, img_size[1])
        bb[3] = np.minimum(det[3]+margin/2, img_size[0])
        cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
        aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
        prewhitened = facenet.prewhiten(aligned)
        
    img_list.append(prewhitened)
    images = np.stack(img_list)
    return images

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('model', type=str, 
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('image_folder', type=str, help='Images to compare')
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--margin', type=int,
        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=0.7)
    parser.add_argument('--threshold', type=float,
        help='Upper bound of distance between two images to ...  .', default=0.7)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
