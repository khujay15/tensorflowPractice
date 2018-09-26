import cv2                 # working with, mainly resizing, images
import numpy as np         # dealing with arrays
import os                  # dealing with directories
from random import shuffle # mixing up or currently ordered data that might lead our network astray in training.
import tensorflow as tf
TRAIN_CAT = './input/PetImages/Cat'
TRAIN_DOG = './input/PetImages/Dog'
TEST_DIR = './input/test'
IMG_SIZE = 96
LR = 1e-3

MODEL_NAME = 'dogsvscats3-{}-{}.model'.format(LR, '2conv-basic') # just so we remember which saved model is which, sizes must match



def label_img(img):
    word_label = img.split('.')[-2]
    # conversion to one-hot array [cat,dog]
    #                            [much cat, no dog]
    if word_label.find('cat') > 0: return [1,0]
    #                             [no cat, very doggo]
    elif word_label.find('dog') > 0: return [0,1]



def create_train_data():
    training_data = []
    for img in (os.listdir(TRAIN_CAT)):
        
        label = [1,0]
        try:
            path = os.path.join(TRAIN_CAT,img)
            img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        except Exception as e:
            print(str(path))
        training_data.append([np.array(img),np.array(label)])
    for img2 in (os.listdir(TRAIN_DOG)):
        label = [0,1]
        try:
            path = os.path.join(TRAIN_DOG,img2)
            img2 = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
            img2 = cv2.resize(img2, (IMG_SIZE,IMG_SIZE))
        except Exception as e:
            print(str(path))
        training_data.append([np.array(img2),np.array(label)])
        
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data


def process_test_data():
    
    testing_data = []
    for img in (os.listdir(TEST_DIR)):
        
        path = os.path.join(TEST_DIR,img)
        img_num = img.split('.')[0]
        try:
            img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
            testing_data.append([np.array(img), img_num])
        except Exception as e:
            print(str(e))
            print(str(path))
       
        
        
    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data




train_data = np.load('train_data.npy')


tf.reset_default_graph()
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

convnet = conv_2d(convnet, 64, 3, padding='same',activation='relu')
convnet = conv_2d(convnet, 64, 3, padding='same',activation='relu')
convnet = max_pool_2d(convnet, 2,strides=2)

convnet = conv_2d(convnet, 128, 3, padding='same',activation='relu')
convnet = conv_2d(convnet, 128, 3, padding='same',activation='relu')
convnet = max_pool_2d(convnet, 2,strides=2)

convnet = conv_2d(convnet, 256, 3, padding='same',activation='relu')
convnet = conv_2d(convnet, 256, 3, padding='same',activation='relu')
convnet = conv_2d(convnet, 256, 3, padding='same',activation='relu')
convnet = max_pool_2d(convnet, 2,strides=2)

convnet = conv_2d(convnet, 512, 3, padding='same',activation='relu')
convnet = conv_2d(convnet, 512, 3, padding='same',activation='relu')
convnet = conv_2d(convnet, 512, 3, padding='same',activation='relu')
convnet = max_pool_2d(convnet, 2,strides=2)


convnet = fully_connected(convnet, 2048, activation='relu')
convnet = dropout(convnet, 0.5)
network = fully_connected(convnet, 2048, activation='relu')
network = dropout(convnet, 0.5)


convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet)



#if os.path.exists('/home/ncp/workspace/catVSdog/{}.meta'.format(MODEL_NAME)):
#    model.load(MODEL_NAME)
#    print('model loaded!')

train = train_data[:-500]
test = train_data[-500:]

X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
test_y = [i[1] for i in test]

model.fit({'input': X}, {'targets': Y}, n_epoch=11, validation_set=({'input': test_x}, {'targets': test_y}), 
    snapshot_step=500, show_metric=True, batch_size=32, run_id=MODEL_NAME)


model.save(MODEL_NAME)


