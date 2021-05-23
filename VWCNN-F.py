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
from mxnet.gluon.nn import Activation
import threading

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

    class _NumpyArrayScope(object):
      _current = threading.local()

      def __init__(self, is_np_array):  # pylint: disable=redefined-outer-name
          self._old_scope = None
          self._is_np_array = is_np_array

      def __enter__(self):
          if not hasattr(_NumpyArrayScope._current, "value"):
              _NumpyArrayScope._current.value = _NumpyArrayScope(False)
          self._old_scope = _NumpyArrayScope._current.value
          _NumpyArrayScope._current.value = self
          return self

      def __exit__(self, ptype, value, trace):
          assert self._old_scope
          _NumpyArrayScope._current.value = self._old_scope
    def is_np_array():
      return _NumpyArrayScope._current.value._is_np_array if hasattr(_NumpyArrayScope._current, "value") else False


    class New_Dense(nn.HybridSequential):
      def __init__(self, units, activation=None, use_bias=True, flatten=True,
                    dtype='float32', weight_initializer=None, bias_initializer='zeros',
                    in_units=0, **kwargs):
            super(New_Dense, self).__init__(**kwargs)
            self._flatten = flatten
            with self.name_scope():
                self._units = units
                self._in_units = in_units
                self.weight = self.params.get('weight', shape=(units, in_units),
                                              init=weight_initializer, dtype=dtype,
                                              allow_deferred_init=True)
                if use_bias:
                    self.bias = self.params.get('bias', shape=(units,),
                                                init=bias_initializer, dtype=dtype,
                                                allow_deferred_init=True)
                else:
                    self.bias = None
                if activation is not None:
                    self.act = Activation(activation, prefix=activation+'_')
                else:
                    self.act = None

      def hybrid_forward(self, F, x, dw, weight, bias=None):
            fc = F.npx.fully_connected if is_np_array() else F.FullyConnected
            act = fc(x, weight - 0.001 * dw, bias, no_bias=bias is None, num_hidden=self._units,
                    flatten=self._flatten, name='fwd')
            if self.act is not None:
                act = self.act(act)
            return act


      def __repr__(self):
            s = '{name}({layout}, {act})'
            shape = self.weight.shape
            return s.format(name=self.__class__.__name__,
                            act=self.act if self.act else 'linear',
                            layout='{0} -> {1}'.format(shape[1] if shape[1] else None, shape[0]))
            


    class VWCNN_2(nn.HybridBlock):
            def __init__(self, **kwargs):
                super(VWCNN_2, self).__init__(**kwargs)
                self.features = nn.HybridSequential()
                self.features.add(nn.Conv2D(channels=30, kernel_size=3, padding=1),
                    nn.BatchNorm(),
                    nn.Activation('relu'),
                    nn.Conv2D(channels=50, kernel_size=3, padding=1),
                    nn.BatchNorm(),
                    nn.Activation('relu'),
                    nn.MaxPool2D(pool_size=2, strides=2),
                    nn.Conv2D(channels=100, kernel_size=3, padding=1),
                    nn.BatchNorm(),
                    nn.Activation('relu'),
                    nn.MaxPool2D(pool_size=2, strides=2),
                    nn.Conv2D(channels=30, kernel_size=2, padding=0),
                    nn.BatchNorm(),
                    nn.Activation('relu'),
                    nn.MaxPool2D(pool_size=2, strides=2))
                self.dense1 = New_Dense(units=3, activation='sigmoid', in_units=30)      
                self.dense0 = nn.Dense(90, activation='sigmoid')
                self.dense2 = New_Dense(units=3, in_units=3)  
                self.dense00 = nn.Dense(9)
                
                

            

                #self.dropout = nn.Dropout(0.8)

            def hybrid_forward(self,F, a):
                x = self.features(a)
                x1 = self.dense0(a)
                #print(x1.shape)
                x1 = x1.mean(axis=0)
                x1 = x1.reshape((3,30)) 
                x2 = self.dense00(a)
                x2 = x2.mean(axis=0)
                x2 = x2.reshape((3,3))
                x = self.dense1(x, x1)
                x = self.dense2(x, x2)     
                return x
    net_vwcnn3 = VWCNN_2()
    num_epochs, ctx = 40, mx.gpu(0)
    #ctx = mx.cpu()
    #net.fancymlp83.bias.zeros_()
    #net_vwcnn2.collect_params().initialize(init.Xavier(), ctx = ctx)
    net_vwcnn3.initialize(force_reinit=True, ctx=ctx)
    #net_vwcnn2.hybridize()

    #lr = 0.01
    trainer = gluon.Trainer(net_vwcnn3.collect_params(), 'adam') 
    d2l.train_ch5(net_vwcnn3, train_data_loader, valid_data_loader, batch_size, trainer, ctx, num_epochs)

    test_acc = d2l.evaluate_accuracy(valid_data_loader, net_vwcnn3, ctx)
    print(test_acc)

    y_val = net_vwcnn3(data2).argmax(axis=1)
    y_val = y_val.asnumpy()
    y_val_pred = []
    for i in range(1,49):
        a = scipy.stats.mode(y_val[(i-1)*494+1:i*494], axis=0)
        y_val_pred.append(a[0])

    acc = sum(1 for a, b in zip(yval_voting_target, y_val_pred) if a == b) / 48
    print(acc)

    y_test = net_vwcnn3(data3).argmax(axis=1)
    y_test = y_test.asnumpy()
    y_test_target2 = []
    for i in range(1,61):
        a = scipy.stats.mode(y_test[(i-1)*494+1:i*494], axis=0)
        y_test_target2.append(a[0])

    acc = sum(1 for a, b in zip(ytest_target1, y_test_target2) if a == b) / 60
    print(acc)

    if acc == 1:
      net_vwcnn3.save_parameters('/content/drive/MyDrive/New_VWCNN/vwcnn_F.params')
