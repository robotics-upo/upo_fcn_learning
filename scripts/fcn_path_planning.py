#!/usr/bin/python2.7
###!/usr/bin/env python

import rospy
import numpy as np
#import pylab as plt
import matplotlib.pyplot as plt

#keras
from keras.models import load_model, Sequential, Model
from keras.preprocessing.image import load_img, img_to_array, array_to_img
from keras.layers import Dense,Dropout,Flatten,Conv2D, Activation, Conv2DTranspose, Input, MaxPooling2D, UpSampling2D, AveragePooling2D
from keras.layers.merge import Add, Concatenate 
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, TensorBoard
from keras import optimizers
from keras import backend as K
from keras.utils import plot_model

#openCV
import cv2

#for directories and files manipulation
from os import listdir
from os.path import isfile, join


class TestCNN(object):


  #--------------------------------------------------------------------------
  def __init__(self, r, s):
    # load the data
    self.dir = r
    self.data_dir = r + 'input/'
    self.labels_dir = r + 'labels/'  
    #self.eval_data_dir = e
    #self.eval_label_dir = r + 'eval/labels/'
    self.save_dir = s
    np.set_printoptions(precision=3, threshold=10000, linewidth=10000)
   



  #--------------------------------------------------------------------------
  def preprocess_image_grey(self, image_path, norm=False, six=False):
    # resize and format pictures
    # into appropriate tensors.
    img = load_img(image_path, grayscale=True)
    img = img_to_array(img)
 
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
  def deprocess_image_grey(self, x):

    #print("Deprocess image grey:")
    #for i in range(len(x[0,0,:,0])):
    #        print(x[0,0,i,:])

    # Util function to convert a tensor into a valid image.
    if K.image_data_format() == 'channels_first':
      x = x.reshape((1, x.shape[2], x.shape[3]))
      x = x.transpose((1, 2, 0))
    else:
      x = x.reshape((x.shape[1], x.shape[2], 1))
    
    
    x *= 255.
    x = np.clip(x, 0, 255).astype('uint8')

    return x




  #--------------------------------------------------------------------------
  def load_data(self, load_npy=False, save_npy=False, norm=False, six=False):

    if load_npy==True:
      self.data = np.load((self.dir+ 'data.npy'))
      self.labels = np.load((self.dir+ 'labels.npy'))
      print('Data loaded from:')
      print((self.dir + 'data.npy'))
      print((self.dir + 'labels.npy'))

    else:
      infiles = [f for f in listdir(self.data_dir) if isfile(join(self.data_dir, f))]
      i = 1
      for f in infiles:
        infile = self.data_dir + f
        label = self.labels_dir + f
        image_in = self.preprocess_image_grey(infile, norm, six)
        image_label = self.preprocess_image_grey(label, norm, six)
        if i == 1:
          self.data = np.array(image_in)
          self.labels = np.array(image_label)
        
        else: 
          self.data = np.append(self.data, image_in, axis=0) #if axis is not specified, the data is flattened
          self.labels = np.append(self.labels, image_label, axis=0)

        if (i % 100) == 0: 
      	  print("Image %i loaded..." % i)
        i = i+1
      print("\nTotal images loaded: %i \n" % (i-1))


    print("Data dimensions: %d " % self.data.ndim)
    print("Shape: ")
    print(self.data.shape)
    print("Size: %d " % self.data.size)
    print("Labels dimensions: %d " % self.labels.ndim)
    print("Shape: ")
    print(self.labels.shape)
    print("Size: %d " % self.labels.size)

    if(save_npy==True):
      file_data = self.dir + 'data.npy'
      np.save(file_data, self.data)
      file_labels = self.dir + 'labels.npy'
      np.save(file_labels, self.labels)
      print('Saving data in files:')
      print(file_data)
      print(file_labels)






#***************************************************************************
#************** NETWORK ARCHITECTURES **************************************
#***************************************************************************


  #--------------------------------------------------------------------------
  #standard FCN (paper Wulfmeier, Watch this: Scalable..)
  def define_standardFCN(self):  
    self.model = Sequential()
    self.model.add(Conv2D(64, (5, 5), input_shape=(3, 200, 200), activation='relu', padding='same'))  
    self.model.add(BatchNormalization())
    self.model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    self.model.add(BatchNormalization())
    self.model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    self.model.add(BatchNormalization())
    #self.model.add(Conv2D(32, (3, 3), activation='relu', padding='same')) #
    #self.model.add(BatchNormalization()) #
    self.model.add(Conv2D(32, (1, 1), activation='relu', padding='same'))
    self.model.add(BatchNormalization())
    self.model.add(Conv2D(3, (1, 1), padding='same')) #activation='relu'

    ## Compile model
    sgd = optimizers.SGD(lr=0.01, clipnorm=1., clipvalue=0.0)
    self.model.compile(optimizer=sgd, loss='mse', metrics=['mae', 'mse'])
    #self.model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse'])
    self.model.summary()




  #--------------------------------------------------------------------------
  #pooling FCN (paper Wulfmeier, Watch this: Scalable..)
  def define_poolingFCN(self):
    input_img = Input(shape=(1, 200, 200))  
    x = Conv2D(64, (5, 5), activation='relu', padding='same')(input_img)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(32, (1, 1), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(1, (1, 1), padding='same')(x)
    net = BatchNormalization()(x)
    self.model = Model(input_img, net)

    ## Compile model
    #sgd = optimizers.SGD(lr=0.01, clipnorm=1., clipvalue=0.0)
    #self.model.compile(optimizer=sgd, loss='mse', metrics=['mae', 'mse'])
    self.model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse'])
    self.model.summary()




  #--------------------------------------------------------------------------
  #Multi-scale FCN (paper Wulfmeier, Watch this: Scalable..)
  def define_MSFCN(self):
    input_img = Input(shape=(1, 200, 200))  
    x = Conv2D(64, (5, 5), activation='relu', padding='same')(input_img)
    #x = BatchNormalization()(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    #x = BatchNormalization()(x)
    
    # first branch
    b1 = MaxPooling2D((2, 2), padding='same')(x)  # 100x100
    b1 = Conv2D(32, (3, 3), activation='relu', padding='same')(b1)
    #b1 = BatchNormalization()(b1)
    b1 = Conv2D(32, (1, 1), activation='relu', padding='same')(b1)
    #b1 = BatchNormalization()(b1)
    b1 = UpSampling2D((2, 2))(b1) # 200x200

    # second branch
    b2 = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    #b2 = BatchNormalization()(b2) #last
    b2 = Conv2D(16, (1, 1), activation='relu', padding='same')(b2)
    #b2 = BatchNormalization()(b2) #last

    # concatenate
    w = Concatenate(axis=1)([b1,b2])
    net = Conv2D(1, (1, 1), padding='same')(w)
    #net = BatchNormalization()(net)

    self.model = Model(input_img, net)

    ## Compile model
    #sgd = optimizers.SGD(lr=0.01, clipnorm=1., clipvalue=0.0)
    #self.model.compile(optimizer=sgd, loss='mse', metrics=['mae', 'mse'])
    self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['mae', 'mse', 'binary_accuracy'])
    self.model.summary()




#--------------------------------------------------------------------------
  #Multi-scale FCN 
  def define_MSFCN6(self):
    input_img = Input(shape=(1, 200, 200))  
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    #x = BatchNormalization()(x)
    #x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    #x = BatchNormalization()(x)
    
    # first branch
    b1 = MaxPooling2D((2, 2), padding='same')(x)  # 100x100
    b1 = Conv2D(16, (5, 5), activation='relu', padding='same')(b1)
    #b1 = BatchNormalization()(b1)
    b1 = MaxPooling2D((2, 2), padding='same')(b1)  # 50x50
    b1 = Conv2D(16, (7, 7), activation='relu', padding='same')(b1)
    b1 = MaxPooling2D((2, 2), padding='same')(b1)  # 25x25
    #b1 = BatchNormalization()(b1)
    b1 = Conv2D(16, (9, 9), activation='relu', padding='same')(b1)
    b1 = UpSampling2D((8, 8))(b1) # 200x200

    # second branch
    #b2 = Conv2D(32, (5, 5), activation='relu', padding='same')(input_img)
    #b2 = BatchNormalization()(b2) #last
    #b2 = Conv2D(16, (3, 3), activation='relu', padding='same')(b2)
    #b2 = BatchNormalization()(b2) #last

    # concatenate
    w = Concatenate(axis=1)([b1,input_img])
    w = Conv2D(16, (3, 3), activation='relu', padding='same')(w)
    #w = Conv2D(16, (1, 1), activation='relu', padding='same')(w)
    net = Conv2D(1, (1, 1), activation='relu', padding='same')(w)
    #net = BatchNormalization()(net)

    self.model = Model(input_img, net)

    ## Compile model
    #sgd = optimizers.SGD(lr=0.01, clipnorm=1., clipvalue=0.0)
    #self.model.compile(optimizer=sgd, loss='mse', metrics=['mae', 'mse'])
    self.model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse'])
    self.model.summary()




#--------------------------------------------------------------------------
  #Multi-scale FCN 
  def define_MSFCN7(self):
    input_img = Input(shape=(1, 200, 200))  
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = Dropout(0.1)(x)
    #x = BatchNormalization()(x)
    #x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    #x = BatchNormalization()(x)
    
    # first branch
    b1 = MaxPooling2D((2, 2), padding='same')(x)  # 100x100
    b1 = Conv2D(16, (5, 5), activation='relu', padding='same')(b1)
    b1 = Dropout(0.1)(b1)
    #b1 = BatchNormalization()(b1)
    b1 = MaxPooling2D((2, 2), padding='same')(b1)  # 50x50
    b1 = Conv2D(16, (7, 7), activation='relu', padding='same')(b1)
    b1 = Dropout(0.1)(b1)
    b1 = MaxPooling2D((2, 2), padding='same')(b1)  # 25x25
    #b1 = BatchNormalization()(b1)
    b1 = Conv2D(16, (9, 9), activation='relu', padding='same')(b1)
    b1 = Dropout(0.1)(b1)
    b1 = UpSampling2D((8, 8))(b1) # 200x200

    # second branch
    #b2 = Conv2D(32, (5, 5), activation='relu', padding='same')(input_img)
    #b2 = BatchNormalization()(b2) #last
    #b2 = Conv2D(16, (3, 3), activation='relu', padding='same')(b2)
    #b2 = BatchNormalization()(b2) #last

    # concatenate
    w = Concatenate(axis=1)([b1,input_img])
    w = Conv2D(16, (3, 3), activation='relu', padding='same')(w)
    w = Dropout(0.1)(w)
    #w = Conv2D(16, (1, 1), activation='relu', padding='same')(w)
    net = Conv2D(1, (1, 1), activation='relu', padding='same')(w)
    #net = BatchNormalization()(net)

    self.model = Model(input_img, net)

    ## Compile model
    #sgd = optimizers.SGD(lr=0.01, clipnorm=1., clipvalue=0.0)
    #self.model.compile(optimizer=sgd, loss='mse', metrics=['mae', 'mse'])
    self.model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse'])
    self.model.summary()





#--------------------------------------------------------------------------
  #Multi-scale FCN 
  def define_MSFCN8(self):
    
    input_img = Input(shape=(1, 200, 200))  
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = Dropout(0.1)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)  # 100x100
    x = Conv2D(16, (5, 5), activation='relu', padding='same')(x)
    x = Dropout(0.1)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)  # 50x50
    x = Conv2D(16, (7, 7), activation='relu', padding='same')(x)
    x = Dropout(0.1)(x)
    x = Conv2D(16, (7, 7), activation='relu', padding='same')(x)
    x = Dropout(0.1)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)  # 25x25
    x = Conv2D(16, (9, 9), activation='relu', padding='same')(x)
    x = Dropout(0.1)(x)
    x = Conv2D(16, (9, 9), activation='relu', padding='same')(x)
    x = Dropout(0.1)(x)
    x = Conv2D(16, (9, 9), activation='relu', padding='same')(x)
    x = Dropout(0.1)(x)
    x = UpSampling2D((8, 8))(x) # 200x200

    w = Concatenate(axis=1)([x,input_img])
    w = Conv2D(32, (3, 3), activation='relu', padding='same')(w)
    w = Dropout(0.1)(w)
    w = Conv2D(32, (3, 3), activation='relu', padding='same')(w)
    w = Dropout(0.1)(w)
    net = Conv2D(1, (1, 1), activation='relu', padding='same')(w)

    ## Compile model
    self.model = Model(input_img, net)
    self.model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse'])
    self.model.summary()





  #--------------------------------------------------------------------------
  # Residual network 
  def define_residualCN(self):
    input_img = Input(shape=(1, 200, 200))  
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    y = BatchNormalization()(x)
    
    # first branch
    x = MaxPooling2D((2, 2), padding='same')(y)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(32, (1, 1), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)

    # Add 1
    #img = Conv2D(16, (1, 1), activation='relu', padding='same')(input_img)
    x = Conv2D(1, (1, 1), padding='same', activation='relu')(x)
    w = Add()([x,input_img]) #Concatenate()
    net = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(w)
    net = BatchNormalization()(net)

    self.model = Model(input_img, net)

    ## Compile model
    sgd = optimizers.SGD(lr=0.01, clipnorm=1., clipvalue=0.0)
    self.model.compile(optimizer=sgd, loss='mse', metrics=['mae', 'mse'])
    #self.model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse'])
    self.model.summary()




  #--------------------------------------------------------------------------
  # deep convolutional Inverse Graphics Network (https://arxiv.org/pdf/1503.03167.pdf)
  def define_dc_ign(self):  
    input_img = Input(shape=(1, 200, 200))  
    x = Conv2D(96, (7, 7), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (5, 5), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    encoder = Conv2D(32, (5, 5), activation='relu', padding='same')(x)

    x = Conv2D(32, (7, 7), activation='relu', padding='same')(encoder)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (7, 7), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(96, (7, 7), activation='relu', padding='same')(x)
    decoder = Conv2D(1, (7, 7), activation='sigmoid', padding='same')(x)
    self.model = Model(input_img, decoder)

    ## Compile model
    #sgd = optimizers.SGD(lr=0.01, clipnorm=1., clipvalue=0.0)
    #self.model.compile(optimizer=sgd, loss='mse', metrics=['mae', 'mse'])
    self.model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse'])
    self.model.summary()




  #--------------------------------------------------------------------------
  #autoencoder (link: Building autoencoders in Keras)
  def define_conv_autoencoder(self):  
    # Encoder
    input_img = Input(shape=(3, 200, 200))  
    x = Conv2D(50, (3, 3), activation='relu', padding='same')(input_img)
    #x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same', data_format='channels_last')(x)
    x = Conv2D(50, (3, 3), activation='relu', padding='same')(x)
    encoder = MaxPooling2D((2, 2), padding='same')(x)
    # Decoder
    x = Conv2D(50, (3, 3), activation='relu', padding='same')(encoder)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(50, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    #x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    decoder = Conv2D(3, (1, 1), activation='relu', padding='same')(x) #activation='sigmoid'
    self.model = Model(input_img, decoder)

    #sgd = optimizers.SGD(lr=0.01, clipnorm=1., clipvalue=0.0)
    #self.model.compile(optimizer=sgd, loss='mse', metrics=['mae', 'mse'])
    self.model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse'])
    self.model.summary() 




  #--------------------------------------------------------------------------
  # sparse auntoencoder
  def define_sparse_autoencoder(self):  
    # Encoder
    input_img = Input(shape=(1, 200, 200))  
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = UpSampling2D((2, 2))(x)
    encoder = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    # Decoder
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoder)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    decoder = Conv2D(1, (1, 1), activation='sigmoid', padding='same')(x) 
    self.model = Model(input_img, decoder)

    #sgd = optimizers.SGD(lr=0.01, clipnorm=1., clipvalue=0.0)
    #self.model.compile(optimizer=sgd, loss='mse', metrics=['mae', 'mse'])
    self.model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse']) 
    self.model.summary()



#*************************************************************************************
#*************************************************************************************




  #--------------------------------------------------------------------------
  def training(self, n_epochs, val_split, batchsize, shuffledata):    
   
    #early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    ## Fit the model
    hist = self.model.fit(self.data, self.labels, validation_split=val_split, epochs=n_epochs, batch_size=batchsize, shuffle=shuffledata) #, callbacks=[early_stopping])  
    print "Trainning finished:"
    print(hist.history)
    self.model.save(self.save_dir + 'my_model.h5')
    print('Saving model learned: %s\n' % (self.save_dir + 'my_model.h5'))
    # Loss
    loss = hist.history["loss"]
    np_loss = np.array(loss)
    np.savetxt((saveroute+"loss_history.txt"), np_loss, delimiter=",")
    # Val_loss
    #val_loss = hist.history["val_loss"]
    #np_val_loss = np.array(val_loss)
    #np.savetxt((saveroute+"val_loss_history.txt"), np_val_loss, delimiter=",")
    # Mean squared error
    mse = hist.history["mean_squared_error"]
    np_mse = np.array(mse)
    np.savetxt((saveroute+"mse_history.txt"), np_mse, delimiter=",")
    # Mean absolute error
    mae = hist.history["mean_absolute_error"]
    np_mae = np.array(mae)
    np.savetxt((saveroute+"mae_history.txt"), np_mae, delimiter=",")
    
    
    plt.figure(1)
    plt.subplot(211)
    plt.plot(loss, '-b')
    plt.ylabel('loss (mse)')
    plt.xlabel('epochs')
    #plt.subplot(312)
    #plt.plot(val_loss, '-r')
    #plt.ylabel('val_loss')
    #plt.xlabel('epochs')
    plt.subplot(212)
    plt.plot(mae, '-g')
    plt.ylabel('mae')
    plt.xlabel('epochs')
    #plt.show()
    plt.savefig((self.save_dir + 'loss.eps'))






#--------------------------------------------------------------------------
if __name__ == '__main__':
 
  rospy.init_node('cnn_traj')
  print('--Starting cnn_traj--')

  #K.set_image_dim_ordering('th')

  normalize_img_data = rospy.get_param('~normalize_image_values', True)
  use_six_categories = rospy.get_param('~use_six_categories', True)
  create_npy_files = rospy.get_param('~create_npy_files', True)
  load_npy_files = rospy.get_param('~load_npy_files', False)

  #Directory where the learning dataset is.
  #Inside we have to put the folders 'input' and 'labels'
  dataset_dir = rospy.get_param('~dataset_dir', '/home/noe/catkin_ws/src/upo_fcn_learning/captures/real_trajs_set1/')
  #Directory where we want to store the results
  save_dir = rospy.get_param('~save_dir', '/home/noe/catkin_ws/src/upo_fcn_learning/captures/real_trajs_set1/')

  
  test_node = TestCNN(dataset_dir, save_dir)
 
 

  # load the training data
  print('\n---LOADING DATA IMAGES---')
  test_node.load_data(load_npy=load_npy_files, save_npy=create_npy_files, norm=normalize_img_data, six=use_six_categories)

  # Define the FCN
  print('\n---SETTING THE NET ARCHITECTURE---')
  test_node.define_MSFCN8()
  

  # train the network
  print('\n---TRAINING THE NETWORK---')
  test_node.training(80, 0, 10, True) #n_epochs, val_split, batchsize, shuffledata






