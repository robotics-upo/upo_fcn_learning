#!/usr/bin/python2.7

#import rospy
import numpy as np
#import pylab as plt
import matplotlib.pyplot as plt

#keras
from keras.models import load_model, Sequential, Model
from keras.preprocessing.image import load_img, img_to_array, array_to_img
from keras import backend as K
from keras.utils import plot_model

#openCV
import cv2

#for directories and files manipulation
from os import listdir
from os.path import isfile, join




#--------------------------------------------------------------------------
def preprocess_image(image_path):
  # resize and format pictures
  # into appropriate tensors.
  img = load_img(image_path)
  img = img_to_array(img)
  img = np.expand_dims(img, axis=0) 
  #print img
  img = img.astype('float32') / 255.
  return img



#--------------------------------------------------------------------------
def deprocess_image(x):
  # Util function to convert a tensor into a valid image.
  if K.image_data_format() == 'channels_first':
    x = x.reshape((1, x.shape[2], x.shape[3]))
    x = x.transpose((1, 2, 0))
  else:
    x = x.reshape((x.shape[1], x.shape[2], 1))
    #x /= 2.
    #x += 0.5
    x *= 255.
    x = np.clip(x, 0, 255).astype('uint8')
  return x


#--------------------------------------------------------------------------
def preprocess_image_grey(image_path, norm=False, six=False):
  # resize and format pictures
  # into appropriate tensors.
  img = load_img(image_path, grayscale=True)
  img = img_to_array(img)
  #print('img.dtype: %s' % img.dtype)
 
  if(norm==True):

    if(six==True):
      img[img <=15] = 0     #free: 0
      img[img >=235] = 1    #path: 255
      img[img >=184] = 0.8  #obstacles: 204
      img[img >=133] = 0.4  #people front: 153
      img[img >=82] = 0.2   #people back: 102
      img[img >=43] = 0.6   # goal: 63
    else:
      img[img <=15] = 0     #free: 0
      img[img >=235] = 1    #path: 255
      img[img >=184] = 0.25  #obstacles: 204
      img[img >=82] = 0.5  #people front: 153, back: 102
      img[img >=43] = 0.75   # goal: 63
  else:
    if(six==True):
      img[img <=15] = 0    #free: 0
      img[img >=235] = 3   #path: 255
      img[img >=184] = 2   #obstacles: 204
      img[img >=133] = -2  #people front: 153
      img[img >=82] = -1   #people back: 102
      img[img >=43] = 1    # goal: 63
    else:
      img[img <=15] = 0    #free: 0
      img[img >=235] = 2   #path: 255
      img[img >=184] = -1   #obstacles: 204
      img[img >=82] = -2  #people front: 153, back: 102
      img[img >=43] = 1    # goal: 63
    
  img = np.expand_dims(img, axis=0) 
  #img = img.astype('float32') / 255.
  #print img.shape

  #print img
  return img


#--------------------------------------------------------------------------
def deprocess_image_grey(x):

  img = x
  # Util function to convert a tensor into a valid image.
  if K.image_data_format() == 'channels_first':
    img = img.reshape((1, img.shape[2], img.shape[3]))
    img = img.transpose((1, 2, 0))
  else:
    img = img.reshape((img.shape[1], img.shape[2], 1))
      
  #x /= 2.
  #x += 0.5
  img *= 255.
  img = np.clip(img, 0, 255).astype('uint8')

  return img


#--------------------------------------------------------------------------
def filter_image_grey(x):

   
  # Util function to convert a tensor into a valid image.
  if K.image_data_format() == 'channels_first':
    x = x.reshape((1, x.shape[2], x.shape[3]))
    x = x.transpose((1, 2, 0))
  else:
    x = x.reshape((x.shape[1], x.shape[2], 1))
 
  x[x < 0.05] = 0     
  #x /= 2.
  #x += 0.5
  x *= 255.
  x = np.clip(x, 0, 255).astype('uint8')

  return x




#--------------------------------------------------------------------------
def load_eval_data(eval_dir, label_dir, load_npy=False, save_npy=False, norm=False, six=False):
  # Evaluation
 
  eval_data = []
  eval_labels = [] 
  if load_npy==True:
    eval_data = np.load((eval_dir + 'data.npy'))
    eval_labels = np.load((label_dir + 'labels.npy'))
    print('Data loaded from:')
    print((eval_dir + 'data.npy'))
    print((eval_dir + 'labels.npy'))

  else:
    inputdata = eval_dir
    labeldata = label_dir
    evalfiles = [g for g in listdir(inputdata) if isfile(join(inputdata, g))]
    i = 1
    for g in evalfiles:
      evalfile = inputdata + g
      evallabel = labeldata + g
      image_in = preprocess_image_grey(evalfile, norm, six)
      image_label = preprocess_image_grey(evallabel, norm, six)
      if i == 1:
        eval_data = np.array(image_in)
        eval_labels = np.array(image_label)
      else: 
        eval_data = np.append(eval_data, image_in, axis=0) #if axis is not specified, the data is flattened
        eval_labels = np.append(eval_labels, image_label, axis=0)

      if (i % 100) == 0: 
      	print("Image %i loaded..." % i)
      i = i+1
    print("\nTotal images loaded: %i \n" % (i-1))

  #print("Eval dimensions: %d " % self.eval_data.ndim)
  print("Eval Shape: ")
  print(eval_data.shape)
  #print("Eval Size: %d " % self.eval_data.size)
  ##print(self.data)
  #print("Eval Labels dimensions: %d " % self.eval_labels.ndim)
  #print("Eval labels Shape: ")
  #print(self.eval_labels.shape)
  #print("Eval labels Size: %d " % self.eval_labels.size)

  if(save_npy==True):
    file_data = eval_dir + 'data.npy'
    np.save(file_data, eval_data)
    file_labels = eval_dir + 'labels.npy'
    np.save(file_labels, eval_labels)
    print('Saving eval data in files:')
    print(file_data)
    print(file_labels)

  return [eval_data, eval_labels]



if __name__ == '__main__':


  test_dir = '/home/noe/catkin_ws/src/upo_cnn_learning/scripts/prediction_test/'
  input_dir = test_dir + 'input_test_500/'
  save_dir = test_dir + 'output/'
  label_dir = test_dir + 'labels_test_500/'
  model = load_model(test_dir + 'my_model.h5')
  print "Loaded_model:"
  print model.summary()
  plot_model(model, to_file=(test_dir+'model.png'), show_shapes=True)
  norm_img_data = True
  use_six = True

  [eval_data, eval_labels] = load_eval_data(input_dir, label_dir, load_npy=False, save_npy=False, norm=norm_img_data, six=use_six)
  score = model.evaluate(eval_data, eval_labels, batch_size=5)
  print('Score:')
  print(score)
  np_score = np.array(score)
  #np.savetxt((test_dir+"test_metrics.txt"), np_score, delimiter=",")


  infiles = [f for f in listdir(input_dir) if isfile(join(input_dir, f))]
  i = 1
  for f in infiles:
    infile = input_dir + f
    label = label_dir + f
    print('Predicting output for image: %s' % f)
    image = preprocess_image_grey(infile, norm_img_data)
    y = model.predict(image, verbose=1)
    #print("prediction shape: ")
    #print y.shape
    #print("shape after reshape: ")
    #print z.shape
    #print("Image:")
    #for i in range(len(z[0,:,0])):
    #  print(z[0,i,:])
    #print z
    i = deprocess_image_grey(y)
    plt.imsave((save_dir + f + "_color.jpeg"), i[:,:,0])
    #y2 = model.predict(image, verbose=1)
    #k = filter_image_grey(y2)
    #plt.imsave((save_dir + f + "_filtered.jpeg"), k[:,:,0])
    z = y.reshape((1, y.shape[2], y.shape[3]))
    z = array_to_img(z)
    z.save(save_dir + f, "JPEG")


     
