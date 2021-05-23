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
import pandas as pd

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
from gluoncv.model_zoo import get_model
import threading

table = {'ABSZ':0,'CPSZ':1, 'FNSZ':2, 'GNSZ':3, 'SPSZ':4, 'TCSZ':5, 'TNSZ':6}

ctx = mx.gpu(0)

for i in range(5):
    i = i + 1
    print('######### vw-resnet: fold '+str(i)+' ############')
    
    data1 = np.load('data_new_train_val_test/train_data_cross_'+str(i)+'.npy')
    data1 = np.float32(data1)
    data1 = nd.array(data1, ctx=ctx)
    data1 = nd.concat(data1, data1, data1, dim=1)


    data2 = np.load('data_new_train_val_test/val_data_cross_'+str(i)+'.npy')
    data2 = np.float32(data2)
    data2 = nd.array(data2, ctx=ctx)
    data2 = nd.concat(data2, data2, data2, dim=1)


    data_train_label = np.load('data_new_train_val_test/train_label_after_cross_'+str(i)+'.npy')
    data_train_label = pd.Categorical(data_train_label)
    data_val_label = np.load('data_new_train_val_test/val_label_after_cross_'+str(i)+'.npy')
    data_val_label = pd.Categorical(data_val_label)

    val_label = []
    for item in data_val_label:
        val_label.append(table[item])

    train_label = []
    for item in data_train_label:
        train_label.append(table[item])


    batch_size = 250

    dataset_train = mx.gluon.data.ArrayDataset(data1, train_label)
    dataset_val = mx.gluon.data.ArrayDataset(data2, val_label)
    train_data_loader = mx.gluon.data.DataLoader(dataset_train, batch_size, shuffle=True)
    valid_data_loader = mx.gluon.data.DataLoader(dataset_val, batch_size, shuffle=False)
    
    data3 = np.load('data_new_train_val_test/test_data.npy')
    data3 = np.float32(data3)
    data3 = nd.array(data3, ctx=ctx)
    data3 = nd.concat(data3, data3, data3, dim=1)


    data_test_label = np.load('data_new_train_val_test/test_label_after.npy')
    data_test_label = pd.Categorical(data_test_label)

    test_label = []
    for item in data_test_label:
        test_label.append(table[item])


    dataset_test = mx.gluon.data.ArrayDataset(data3, test_label)
    test_data_loader = mx.gluon.data.DataLoader(dataset_test, batch_size, shuffle=False)

    

    def evaluate_accuracy(data_iter, net, ctx):
        acc_sum, n = nd.array([0], ctx=ctx), 0
        for X, y in data_iter:
            X, y = X.as_in_context(ctx), y.as_in_context(ctx).astype('float32')
            acc_sum += (net(X).argmax(axis=1) == y).sum()
            n += y.size
        return acc_sum.asscalar() / n


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
        return _NumpyArrayScope._current.value._is_np_array if hasattr(
            _NumpyArrayScope._current, "value") else False

    def _conv3x3(channels, stride, in_channels):
            return nn.Conv2D(channels, kernel_size=3, strides=stride, padding=1,
                         use_bias=False, in_channels=in_channels)

    class FancyConv2D(nn.HybridBlock):
        def __init__(self, channels, stride, inputsize, kernel_size=(3,3), padding=1, **kwargs):
            super(FancyConv2D, self).__init__(**kwargs)
            self.weight = self.params.get('weight', shape=(channels, inputsize) + kernel_size)
         #   self.bias = self.params.get('bias', shape=(channels,))
            self.num_filter = channels
            self.kernel_size = kernel_size
            self.pad = (padding, padding)
            self.stride = (stride, stride)

        def forward(self, x, dw):
            return nd.Convolution(data=x, weight=self.weight.data() + 0.0005*dw, 
                                      stride = self.stride, no_bias=True, kernel=self.kernel_size, 
                                      num_filter=self.num_filter, pad = self.pad)


    class BasicBlockV2(nn.HybridBlock):  
        def __init__(self, channels, stride, downsample=False, in_channels=0, **kwargs):
            super(BasicBlockV2, self).__init__(**kwargs)
            self.bn1 = nn.BatchNorm()
            self.conv1 = _conv3x3(channels, stride, in_channels)
            self.bn2 = nn.BatchNorm()
            self.conv2 = FancyConv2D(channels, 1, channels)
            if downsample:
                self.downsample = nn.Conv2D(channels, 1, stride, use_bias=False,
                                            in_channels=in_channels)
            else:
                self.downsample = None

        def hybrid_forward(self, F, x, dw):
            residual = x
            x = self.bn1(x)
            act = F.npx.activation if is_np_array() else F.Activation
            x = act(x, act_type='relu')
            if self.downsample:
                residual = self.downsample(x)
            x = self.conv1(x)

            x = self.bn2(x)
            x = act(x, act_type='relu')
            x = self.conv2(x, dw)

            return x + residual


    def dense(num_channels):
        blk = nn.HybridSequential() 
        blk.add(nn.Dense(num_channels))
        return blk


    class ResNetV2(nn.HybridBlock):
        def __init__(self, layers, channels, classes=7, thumbnail=False, **kwargs):
            super(ResNetV2, self).__init__(**kwargs)
            assert len(layers) == len(channels) - 1
            self.features = nn.HybridSequential()
            self.features.add(nn.BatchNorm(scale=False, center=False))
            if thumbnail:
                self.features.add(_conv3x3(channels[0], 1, 0))
            else:
                self.features.add(nn.Conv2D(channels[0], 7, 2, 3, use_bias=False))
                self.features.add(nn.BatchNorm())
                self.features.add(nn.Activation('relu'))
                self.features.add(nn.MaxPool2D(3, 2, 1))

            self.block1 = BasicBlockV2(64, 1, False, 64)  #64 1 False 64 
            self.block2 = BasicBlockV2(64, 1, False, 64)  #64 1 False 64 

            self.block3 = BasicBlockV2(128, 2, True, 64)  #128 2 True 64 
            self.block4 = BasicBlockV2(128, 1, False, 128)  #128 1 False 128 

            self.block5 = BasicBlockV2(256, 2, True, 128)  #256 2 True 128 
            self.block6 = BasicBlockV2(256, 1, False, 256)  #256 1 False 256 

            self.block7 = BasicBlockV2(512, 2, True, 256)  #512 2 True 256 
            self.block8 = BasicBlockV2(512, 1, False, 512)  #512 1 False 512 


            self.dense1 = dense(64*9)
            self.dense2 = dense(64*9)
            self.dense3 = dense(128*9)
            self.dense4 = dense(128*9)
            self.dense5 = dense(256*9)
            self.dense6 = dense(256*9)
            self.dense7 = dense(512*9)
            self.dense8 = dense(512*9)


            self.features1 = nn.HybridSequential()
            self.features1.add(nn.BatchNorm())
            self.features1.add(nn.Activation('relu'))
            self.features1.add(nn.GlobalAvgPool2D())
            self.features1.add(nn.Flatten())

            self.output = nn.Dense(classes, in_units=channels[-1])

        def hybrid_forward(self, F, x):
            dw1 = self.dense1(x).mean(axis=0)
            dw1 = dw1.reshape((1,64,3,3))
            dw2 = self.dense2(x).mean(axis=0)
            dw2 = dw2.reshape((1,64,3,3))
            dw3 = self.dense3(x).mean(axis=0)
            dw3 = dw3.reshape((1,128,3,3))
            dw4 = self.dense4(x).mean(axis=0)
            dw4 = dw4.reshape((1,128,3,3))
            dw5 = self.dense5(x).mean(axis=0)
            dw5 = dw5.reshape((1,256,3,3))
            dw6 = self.dense6(x).mean(axis=0)
            dw6 = dw6.reshape((1,256,3,3))
            dw7 = self.dense7(x).mean(axis=0)
            dw7 = dw7.reshape((1,512,3,3))
            dw8 = self.dense8(x).mean(axis=0)
            dw8 = dw8.reshape((1,512,3,3))


            x = self.features(x)
            x = self.block1(x, dw1)
            x = self.block2(x, dw2)
            x = self.block3(x, dw3)
            x = self.block4(x, dw4)
            x = self.block5(x, dw5)
            x = self.block6(x, dw6)
            x = self.block7(x, dw7)
            x = self.block8(x, dw8)

            x = self.features1(x)
            x = self.output(x)
            return x

    net = ResNetV2([3, 4, 6, 3],[64, 64, 128, 256, 512])


    epochs = 10
    net.initialize(force_reinit=True, ctx=ctx, init=init.Xavier())
    #net.collect_params().reset_ctx(ctx)
    #net.hybridize()

    trainer = gluon.Trainer(net.collect_params(), 'adam', {'wd': 0.001})
    d2l.train_ch5(net, train_data_loader, valid_data_loader, batch_size, trainer, ctx, num_epochs=epochs)
    trainer = gluon.Trainer(net.collect_params(), 'sgd')
    d2l.train_ch5(net, train_data_loader, valid_data_loader, batch_size, trainer, ctx, num_epochs=5)

    file_name = "models/final_"+str(i)+".params"
    net.save_parameters(file_name)
    
    test_acc = evaluate_accuracy(valid_data_loader, net, ctx)
    print(test_acc)
    
    def get_y_pred(data_iter, net, ctx):
        y_pred, y_true, n = [], [], 0
        for X, y in data_iter:
            X, y = X.as_in_context(ctx), y.as_in_context(ctx).astype('float32')
            y_pred.append(net(X).argmax(axis=1)) 
            y_true.append(y) 
            n += y.size
        return y_pred, y_true, n

    y_pred, y_true, n = get_y_pred(valid_data_loader, net, ctx)

    each_lenth = np.load('data_new_train_val_test/each_lenth_val_'+str(i)+'.npy')
    if n == sum(each_lenth):
        print('The length is correct.')

    y_pred1 = []
    for items in y_pred:
        y_pred1 += list(items.asnumpy())

    y_true1 = []
    for items in y_true:
        y_true1 += list(items.asnumpy())

    acc = sum(1 for a, b in zip(y_pred1, y_true1) if a == b) / len(y_true1)
    print('acc before voting: ', acc)

    accumulated = np.load('data_new_train_val_test/accumulated_val_'+str(i)+'.npy')
    y_pred_after = []
    for j in range(len(each_lenth)):
        vote = y_pred1[accumulated[j]:accumulated[j]+each_lenth[j]]
        pred1 = int(scipy.stats.mode(vote)[0])
        y_pred_after.append(pred1)

    y_true_after = []
    for j in range(len(each_lenth)):
        vote = y_true1[accumulated[j]:accumulated[j]+each_lenth[j]]
        pred1 = int(scipy.stats.mode(vote)[0])
        y_true_after.append(pred1)

    y_true_before = np.load('data_new_train_val_test/val_label_before_cross_'+str(i)+'.npy')
    y_true_before = pd.Categorical(y_true_before)

    table = {'ABSZ':0,'CPSZ':1, 'FNSZ':2, 'GNSZ':3, 'SPSZ':4, 'TCSZ':5, 'TNSZ':6}
    y_true = []

    for item in y_true_before:
        y_true.append(table[item])

    acc = sum(1 for a, b in zip(y_pred_after, y_true_after) if a == b) / len(y_true_after)
    print('acc after voting: ', acc)

    if y_true == y_true_after:
        print('Majority voting is done correctly.')

    from sklearn.metrics import f1_score
    #y_pred = y_pred.as_in_context(mx.cpu())
    f1 = f1_score(y_true_after, y_pred_after, average='weighted')
    print('weighted f1-score: ', f1)
    
    ###########################################print test result##########################################################################
    print('test results:')
    test_acc = evaluate_accuracy(test_data_loader, net, ctx)
    print(test_acc)

    y_pred, y_true, n = get_y_pred(test_data_loader, net, ctx)


    each_lenth = np.load('data_new_train_val_test/each_lenth_test.npy')
    if n == sum(each_lenth):
        print('The length is correct.')

    y_pred1 = []
    for items in y_pred:
        y_pred1 += list(items.asnumpy())

    y_true1 = []
    for items in y_true:
        y_true1 += list(items.asnumpy())

    acc = sum(1 for a, b in zip(y_pred1, y_true1) if a == b) / len(y_true1)
    print('acc before voting: ', acc)

    accumulated = np.load('data_new_train_val_test/accumulated_test.npy')
    y_pred_after = []
    for j in range(len(each_lenth)):
        vote = y_pred1[accumulated[j]:accumulated[j]+each_lenth[j]]
        pred1 = int(scipy.stats.mode(vote)[0])
        y_pred_after.append(pred1)

    y_true_after = []
    for j in range(len(each_lenth)):
        vote = y_true1[accumulated[j]:accumulated[j]+each_lenth[j]]
        pred1 = int(scipy.stats.mode(vote)[0])
        y_true_after.append(pred1)

    y_true_before = np.load('data_new_train_val_test/test_label_before.npy')
    y_true_before = pd.Categorical(y_true_before)

    table = {'ABSZ':0,'CPSZ':1, 'FNSZ':2, 'GNSZ':3, 'SPSZ':4, 'TCSZ':5, 'TNSZ':6}
    y_true = []

    for item in y_true_before:
        y_true.append(table[item])

    acc = sum(1 for a, b in zip(y_pred_after, y_true_after) if a == b) / len(y_true_after)
    print('acc after voting: ', acc)

    if y_true == y_true_after:
        print('Majority voting is done correctly.')

    from sklearn.metrics import f1_score
    #y_pred = y_pred.as_in_context(mx.cpu())
    f1 = f1_score(y_true_after, y_pred_after, average='weighted')
    print('weighted f1-score: ', f1)

