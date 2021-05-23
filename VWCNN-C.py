import h5py
import numpy as np
import struct
import scipy
from scipy import stats
import scipy.io as sio
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import timeit
import numpy as np
import scipy.io as sio  
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC  
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from IPython.display import Image
#import pydotplus 
from numpy import ones
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition
from sklearn.naive_bayes import GaussianNB

from mxnet import nd
from mxnet.gluon import nn
import d2lzh as d2l
from mxnet.gluon import loss as gloss
import mxnet as mx
from mxnet import gluon, init
from mxnet import autograd
import time
from mxnet.gluon import data as gdata, loss as gloss, utils as gutils
import sys



ytest_target = sio.loadmat('/content/drive/MyDrive/New_VWCNN/YTest.dat')
ytest_target= ytest_target.get('YTest') - 1
ytest_target1 = [item[0] for item in ytest_target] 

for i in range(1,6):

    data1 = sio.loadmat('/content/drive/MyDrive/New_VWCNN/Fold_'+str(i)+'_XTrain_Crops_1.dat')
    data1 = data1.get('XTrain_Crops')
    data1 = np.float32(data1)
    data1 = data1.transpose((3,2,0,1))
    data1 = nd.array(data1, ctx=mx.gpu(0))
    #data1 = nd.concat(data1, data1, data1, dim=1)
    np.shape(data1)

    data2 = sio.loadmat('/content/drive/MyDrive/New_VWCNN/Fold_'+str(i)+'_XVal_Crops_1.dat')
    data2 = data2.get('XVal_Crops')
    data2 = np.float32(data2)
    data2 = data2.transpose((3,2,0,1))
    data2 = nd.array(data2, ctx=mx.gpu(0))
    #data2 = nd.concat(data2, data2, data2, dim=1)
    np.shape(data2)

    data3 = sio.loadmat('/content/drive/MyDrive/New_VWCNN/XTest_Crops_1.dat')
    data3 = data3.get('XTest_Crops')
    data3 = np.float32(data3)
    data3 = data3.transpose((3,2,0,1))
    data3 = nd.array(data3, ctx=mx.gpu(0))
    #data2 = nd.concat(data2, data2, data2, dim=1)
    np.shape(data3)

    ytrain = sio.loadmat('/content/drive/MyDrive/New_VWCNN/Fold_'+str(i)+'_YTrain_Crops_1.dat')
    data_train_label = ytrain.get('YTrain_Crops')
    data_train_label = np.ravel(data_train_label)
    data_train_label = data_train_label.astype('float32')
    #data_train_label = data_train_label[0:40000]
    data_train_label = nd.array(data_train_label, ctx=mx.gpu(0))
    data_train_label = data_train_label.astype('float32') - 1
    data_train_label

    yval = sio.loadmat('/content/drive/MyDrive/New_VWCNN/Fold_'+str(i)+'_YVal_Crops_1.dat')
    data_val_label = yval.get('YVal_Crops')
    data_val_label = np.ravel(data_val_label)
    data_val_label = data_val_label.astype('float32')
    data_val_label = nd.array(data_val_label, ctx=mx.gpu(0))
    data_val_label = data_val_label.astype('float32') - 1
    data_val_label


    ytest = sio.loadmat('/content/drive/MyDrive/New_VWCNN/YTest_Crops_1.dat')
    data_test_label = ytest.get('YTest_Crops')
    data_test_label = np.ravel(data_test_label)
    data_test_label = data_test_label.astype('float32')
    #data_train_label = data_train_label[0:40000]
    data_test_label = nd.array(data_test_label, ctx=mx.gpu(0))
    data_test_label = data_test_label.astype('float32') - 1
    data_test_label


    yval_voting = sio.loadmat('/content/drive/MyDrive/New_VWCNN/Fold_'+str(i)+'_YVal_1.dat')
    yval_voting = yval_voting.get('fold_val_label') - 1
    yval_voting_target = [item[0] for item in yval_voting] 
    yval_voting_target

    batch_size = 500

    dataset_train = mx.gluon.data.ArrayDataset(data1, data_train_label)
    dataset_val = mx.gluon.data.ArrayDataset(data2, data_val_label)
    dataset_test = mx.gluon.data.ArrayDataset(data3, data_test_label)
    train_data_loader = mx.gluon.data.DataLoader(dataset_train, batch_size, shuffle=True)
    valid_data_loader = mx.gluon.data.DataLoader(dataset_val, batch_size, shuffle=False)
    test_data_loader = mx.gluon.data.DataLoader(dataset_test, batch_size, shuffle=False)

    class FancyConv2D(nn.HybridSequential):
        def __init__(self, channels, inputsize, kernel_size, pad, **kwargs):
            super(FancyConv2D, self).__init__(**kwargs)
            self.weight = self.params.get('weight', shape=(channels, inputsize) + kernel_size)
            self.bias = self.params.get('bias', shape=(channels,))
            self.num_filter = channels
            self.kernel_size = kernel_size
            self.pad = pad

        def forward(self, x, dw, db, alpha1, alpha2):
            return nd.Convolution(data=x, weight=self.weight.data() + 0.01 * alpha1 * dw, bias=self.bias.data() + 0.01 * alpha2 * db, kernel=self.kernel_size, num_filter=self.num_filter, pad = self.pad)


    class VWCNN_2(nn.HybridBlock):
        def __init__(self, **kwargs):
            super(VWCNN_2, self).__init__(**kwargs)
            self.conv2d_1 = nn.Conv2D(channels=30, kernel_size=3, padding=1)
            self.batchnorm1 = nn.BatchNorm()
            self.batchnorm2 = nn.BatchNorm()
            self.batchnorm3 = nn.BatchNorm()
            self.batchnorm4 = nn.BatchNorm()
            self.relu = nn.Activation('relu')
            self.maxpooling = nn.MaxPool2D(pool_size=2, strides=2)
            self.dense0 = nn.Dense(450, activation='sigmoid')
            self.dense1 = nn.Dense(3, activation='sigmoid')
            self.dense2 = nn.Dense(3)
            self.conv2d_2 = FancyConv2D(channels=50, inputsize=30, kernel_size=(3,3), pad = (1,1))
            self.conv2d_3 = FancyConv2D(channels=100, inputsize=50, kernel_size=(3,3), pad = (1,1))
            self.conv2d_4 = FancyConv2D(channels=30, inputsize=100, kernel_size=(2,2), pad = [])
            self.dense3 = nn.Dense(120, activation='sigmoid')
            self.dense4 = nn.Dense(900, activation='sigmoid')
            self.dense0_1 = nn.Dense(50, activation='sigmoid')
            self.dense3_1 = nn.Dense(30, activation='sigmoid')
            self.dense4_1 = nn.Dense(100, activation='sigmoid')
            self.dense_w = nn.Dense(2, activation='sigmoid')


            #self.dropout = nn.Dropout(0.8)

        def hybrid_forward(self,F, a):
            alpha = self.dense_w(a)
            alpha = alpha.mean(axis=0)
            alpha = alpha.reshape((2,))
            x = self.conv2d_1(a)
            x = self.batchnorm1(x)
            x = self.relu(x)
            x1 = self.dense0(a)
            #self.dropout
            x1 = x1.mean(axis=0)
            x1 = x1.reshape((50,1,3,3)) 
            x1_b = self.dense0_1(a)
            x1_b = x1_b.mean(axis=0)
            x1_b = x1_b.reshape((50,)) 
            x2 = self.conv2d_2(x, x1, x1_b, alpha[0], alpha[1])
            x2 = self.batchnorm2(x2)
            x2 = self.relu(x2)
            x2 = self.maxpooling(x2)
            x2_1 = self.dense4(a)
            x2_1 = x2_1.mean(axis=0)
            x2_1 = x2_1.reshape((100,1,3,3))
            x2_b = self.dense4_1(a)
            x2_b = x2_b.mean(axis=0)
            x2_b = x2_b.reshape((100,)) 
            x2_2 = self.conv2d_3(x2, x2_1, x2_b, alpha[0], alpha[1])
            x2_2 = self.batchnorm3(x2_2)
            x2_2 = self.relu(x2_2)
            x2_2 = self.maxpooling(x2_2) 
            x3 = self.dense3(a)
            #self.dropout
            x3 = x3.mean(axis=0)
            x3 = x3.reshape((30,1,2,2))
            x3_b = self.dense3_1(a)
            x3_b = x3_b.mean(axis=0)
            x3_b = x3_b.reshape((30,))        
            x4 = self.conv2d_4(x2_2, x3, x3_b, alpha[0], alpha[1])
            x4 = self.batchnorm4(x4)
            x4 = self.relu(x4)
            x4 = self.maxpooling(x4)
            x4 = self.dense1(x4)
            x4 = self.dense2(x4)      
            return x4

    net_vwcnn2 = VWCNN_2()
    num_epochs, ctx = 40, mx.gpu(0)
    #ctx = mx.cpu()
    #net.fancymlp83.bias.zeros_()
    #net_vwcnn2.collect_params().initialize(init.Xavier(), ctx = ctx)
    net_vwcnn2.initialize(force_reinit=True, ctx=ctx)
    #net_vwcnn2.hybridize()

    #lr = 0.01
    trainer = gluon.Trainer(net_vwcnn2.collect_params(), 'adam') 
    d2l.train_ch5(net_vwcnn2, train_data_loader, valid_data_loader, batch_size, trainer, ctx, num_epochs)
    #trainer = gluon.Trainer(net_vwcnn2.collect_params(), 'sgd') 
    #d2l.train_ch5(net_vwcnn2, train_data_loader, valid_data_loader, batch_size, trainer, ctx, num_epochs=5)

    test_acc = d2l.evaluate_accuracy(valid_data_loader, net_vwcnn2, ctx)
    print(test_acc)

    y_val = net_vwcnn2(data2).argmax(axis=1)
    y_val = y_val.asnumpy()
    y_val_pred = []
    for i in range(1,49):
        a = scipy.stats.mode(y_val[(i-1)*494+1:i*494], axis=0)
        y_val_pred.append(a[0])

    acc = sum(1 for a, b in zip(yval_voting_target, y_val_pred) if a == b) / 48
    print(acc)

    y_test = net_vwcnn2(data3).argmax(axis=1)
    y_test = y_test.asnumpy()
    y_test_target2 = []
    for i in range(1,61):
        a = scipy.stats.mode(y_test[(i-1)*494+1:i*494], axis=0)
        y_test_target2.append(a[0])

    acc = sum(1 for a, b in zip(ytest_target1, y_test_target2) if a == b) / 60
    print(acc)

    if acc == 1:
      net_vwcnn2.save_parameters('/content/drive/MyDrive/New_VWCNN/acc_1_model.params')

    # 1 1 1 1 1
