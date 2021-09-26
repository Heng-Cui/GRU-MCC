import re
from numpy import *
import scipy.io as sio
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import random
import os
import torch
import torch.backends.cudnn as cudnn
from torch import nn
from torchvision import datasets, transforms
import numpy as np
from data_loader import GetLoader
from torch.autograd import Function
from torch import optim
import torch.nn.functional as F
from pytorchtools import EarlyStopping
from sklearn import svm, metrics
from model import DenseNet
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib import cm
import itertools

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)#/len(kernel_val)


def mmd_rbf_accelerate(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
        kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    loss = 0
    for i in range(batch_size):
        s1, s2 = i, (i+1)%batch_size
        t1, t2 = s1+batch_size, s2+batch_size
        loss += kernels[s1, s2] + kernels[t1, t2]
        loss -= kernels[s1, t2] + kernels[s2, t1]
    return loss / float(batch_size)

def mmd_rbf_noaccelerate(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY -YX)
    return loss

def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy
class GradientReverseLayer(Function):
    """
    Gradient Reversal Layer implementation.
    Taken from https://discuss.pytorch.org/t/solved-reverse-gradients-in-backward-pass/3589/3
    """

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        output = grad.neg() * ctx.alpha
        return output, None




class CALayer(nn.Module):
    def __init__(self, channel, ratio=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, 32, bias=False),
            nn.ReLU(True),
            nn.Linear(32, channel, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _ = x.size()
        avg_out = self.avg_pool(x).view(b, c)
        max_out = self.max_pool(x).view(b, c)
        avg_out = self.fc(avg_out).view(b, c, 1)
        max_out = self.fc(max_out).view(b, c, 1)
        out = avg_out + max_out
        y = self.sigmoid(out)
        return x * y.expand_as(x)

class ECALayer(nn.Module):

    def __init__(self, channel, k_size=5):
        super(ECALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.transpose(-1, -2)).transpose(-1, -2)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, 32, bias=False),
            nn.Tanh(),
            nn.Linear(32, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)



class RNNModel(nn.Module):
    """
    Architecture of the Neural Network.
    Taken from http://sites.skoltech.ru/compvision/projects/grl/files/suppmat.pdf
    """

    def __init__(self):
        super(RNNModel, self).__init__()
        self.feature = nn.GRU(
            input_size=5,  # 图片每行的数据像素点
            hidden_size=16,  # rnn hidden unit
            num_layers=1,  # 有几层 RNN layers
            batch_first=True,
            #dropout = 0.3,
            #bidirectional=True,
        )
        #self.se = SELayer(channel = 62)
        #self.se = SELayer()

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_drop3', nn.Dropout(0.5))
        self.class_classifier.add_module('c_fc1', nn.Linear(62*16, 512))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(512))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout(0.5))
        #self.class_classifier.add_module('c_fc2', nn.Linear(512, 256))
        #self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(256))
        #self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        #self.class_classifier.add_module('c_drop2', nn.Dropout(0.3))
        self.class_classifier.add_module('c_fc3', nn.Linear(512, 3))
        # self.class_classifier.add_module('c_softmax', nn.LogSoftmax(dim=1))


    def forward(self, input_data):
        out, h = self.feature(input_data, None)
        #out = self.se(out)
        features = out.reshape(out.size(0), -1)
        classifier = self.class_classifier(features)

        return classifier, features

class GRUModel(nn.Module):
    """
    Architecture of the Neural Network.
    Taken from http://sites.skoltech.ru/compvision/projects/grl/files/suppmat.pdf
    """

    def __init__(self):
        super(GRUModel, self).__init__()
        self.feature = nn.GRU(
            input_size=5,  # 图片每行的数据像素点
            hidden_size=16,  # rnn hidden unit
            num_layers=1,  # 有几层 RNN layers
            batch_first=True,
            #dropout = 0.3,
            #bidirectional=True,
        )
        #self.se = SELayer(channel = 62)
        #self.se = SELayer()

        self.fc1 = nn.Sequential(
            #nn.Dropout(0.5),
            nn.Linear(62 * 16, 512),
            nn.BatchNorm1d(512, affine=True),
            nn.ReLU(inplace=True),
            #nn.Dropout(0.5),
        )
        self.fc2 = nn.Linear(512, 3)


    def forward(self, input_data):
        out, h = self.feature(input_data, None)
        #out = self.se(out)
        features = out.reshape(out.size(0), -1)
        features = self.fc1(features)
        classifier = self.fc2(features)

        return classifier, features


class MLPModel(nn.Module):
    """
    Architecture of the Neural Network.
    Taken from http://sites.skoltech.ru/compvision/projects/grl/files/suppmat.pdf
    """

    def __init__(self):
        super(MLPModel, self).__init__()
        self.feature = nn.Sequential()
        #self.feature.add_module('f_drop1', nn.Dropout(0.5))
        self.feature.add_module('f_fc1', nn.Linear(62*5, 128))
        self.feature.add_module('f_relu1', nn.ReLU(True))
        #self.feature.add_module('f_bn1', nn.BatchNorm1d(256))
        #self.feature.add_module('f_drop1', nn.Dropout(0.3))
        #self.feature.add_module('f_fc2', nn.Linear(256, 128))
        #self.feature.add_module('f_relu2', nn.ReLU(True))
        self.feature.add_module('f_bn2', nn.BatchNorm1d(128))

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_drop1', nn.Dropout(0.5))
        #self.class_classifier.add_module('c_fc1', nn.Linear(128, 64))
        #self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        # self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(128))
        #self.class_classifier.add_module('c_drop2', nn.Dropout(0.5))
        self.class_classifier.add_module('c_fc2', nn.Linear(128, 3))
        # self.class_classifier.add_module('c_softmax', nn.LogSoftmax(dim=1))


    def forward(self, input_data):
        features = self.feature(input_data)
        classifier = self.class_classifier(features)
        return classifier, features



def normalize(data):
    '''
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    data = min_max_scaler.fit_transform(data)
    '''
    data -= np.mean(data, axis=0)
    data /= np.std(data, axis=0)

    return data

def get_cross_sub_lstmdata(sub_id):
    seq_length = 10
    fe = 'de_LDS'
    input_train = np.empty([0,seq_length,310], dtype=np.float32)
    output_train = list()
    input_test = np.empty([0,seq_length,310], dtype=np.float32)
    output_test = list()
    label = sio.loadmat("/data/ch/data/SEED/label.mat")
    label = label['label'].reshape(-1)
    for sub in range(1, 15):
        if sub != sub_id:
            sub = "%02d" % sub
            for section in range(1, 4):
                input_train_s = list()
                data_path = '/data/ch/data/SEED/seed_ExtractedFeatures/s' + str(sub) + '_' + str(section) + '.mat'
                data = sio.loadmat(data_path)
                for i in range(1, 16):
                    input_train1 = data[fe + str(i)].transpose((1, 2, 0)).reshape((-1,310))
                    #input_train_s = np.vstack([input_train_s, input_train1])
                    for j in range(0, len(input_train1), 1):
                        if j + seq_length <= len(input_train1):
                            input_train_s.append(input_train1[j:j + seq_length])
                            output_train.append(label[i - 1] + 1)
                input_train_s = np.array(input_train_s, dtype=np.float32)
                input_train_s = normalize(input_train_s)
                input_train = np.vstack([input_train, input_train_s])


    sub_id = "%02d" % sub_id
    for section in range(1, 4):
        input_test_s = list()
        data_path = '/data/ch/data/SEED/seed_ExtractedFeatures/s' + str(sub_id) + '_' + str(section) + '.mat'
        data = sio.loadmat(data_path)
        for i in range(1, 16):
            input_test1 = data[fe + str(i)].transpose((1, 2, 0)).reshape((-1,310))
            for j in range(0, len(input_test1), 1):
                if j + seq_length <= len(input_test1):
                    input_test_s.append(input_test1[j:j + seq_length])
                    output_test.append(label[i - 1] + 1)
        input_test_s = np.array(input_test_s, dtype=np.float32)
        input_test_s = normalize(input_test_s)
        input_test = np.vstack([input_test, input_test_s])


    output_train = np.array(output_train)
    output_test = np.array(output_test)
    # output_train = np_utils.to_categorical(output_train, num_classes=3)
    # output_test = np_utils.to_categorical(output_test, num_classes=3)
    print(input_train.shape)
    print(input_test.shape)
    print(output_train.shape)
    print(output_test.shape)

    return input_train, input_test, output_train, output_test




def get_cross_sub_data2(sub_id):
    seq_length = 1
    fe = 'de_LDS'
    input_train = np.empty([0,310], dtype=np.float32)
    output_train = list()
    input_test = np.empty([0,310], dtype=np.float32)
    output_test = list()
    label = sio.loadmat("/data/ch/data/SEED/label.mat")
    label = label['label'].reshape(-1)
    for sub in range(1, 15):
        if sub != sub_id:
            sub = "%02d" % sub
            for section in range(1, 4):
                input_train_s = np.empty([0, 310], dtype=np.float32)
                data_path = '/data/ch/data/SEED/seed_ExtractedFeatures/s' + str(sub) + '_' + str(section) + '.mat'
                data = sio.loadmat(data_path)
                for i in range(1, 16):
                    input_train1 = data[fe + str(i)].transpose((1, 2, 0)).reshape((-1,310))
                    input_train_s = np.vstack([input_train_s, input_train1])
                    for j in range(0, len(input_train1), 1):
                        if j + seq_length <= len(input_train1):
                            output_train.append(label[i - 1] + 1)

                input_train_s = normalize(input_train_s)
                input_train = np.vstack([input_train, input_train_s])


    sub_id = "%02d" % sub_id
    for section in range(1, 4):
        input_test_s = np.empty([0, 310], dtype=np.float32)
        data_path = '/data/ch/data/SEED/seed_ExtractedFeatures/s' + str(sub_id) + '_' + str(section) + '.mat'
        data = sio.loadmat(data_path)
        for i in range(1, 16):
            input_test1 = data[fe + str(i)].transpose((1, 2, 0)).reshape((-1,310))
            input_test_s = np.vstack([input_test_s, input_test1])
            for j in range(0, len(input_test1), 1):
                if j + seq_length <= len(input_test1):
                    output_test.append(label[i - 1] + 1)
        input_test_s = normalize(input_test_s)
        input_test = np.vstack([input_test, input_test_s])


    output_train = np.array(output_train)
    output_test = np.array(output_test)
    # output_train = np_utils.to_categorical(output_train, num_classes=3)
    # output_test = np_utils.to_categorical(output_test, num_classes=3)
    print(input_train.shape)
    print(input_test.shape)
    print(output_train.shape)
    print(output_test.shape)

    return input_train, input_test, output_train, output_test

def get_cross_sub_data(sub_id):
    seq_length = 1
    fe = 'de_LDS'
    input_train = np.empty([0,62, 5], dtype=np.float32)
    output_train = list()
    input_test = np.empty([0,62, 5], dtype=np.float32)
    output_test = list()
    label = sio.loadmat("/data/ch/data/SEED/label.mat")
    label = label['label'].reshape(-1)
    section = 1
    for sub in range(1, 15):
        if sub != sub_id:
            sub = "%02d" % sub
            data_path = '/data/ch/data/SEED/seed_ExtractedFeatures/s' + str(sub) + '_' + str(section) + '.mat'
            data = sio.loadmat(data_path)
            input_train_s = np.empty([0, 62, 5], dtype=np.float32)
            for i in range(1, 16):
                input_train1 = data[fe + str(i)].transpose((1, 0, 2))
                input_train_s = np.vstack([input_train_s, input_train1])
                for j in range(0, len(input_train1), 1):
                    if j + seq_length <= len(input_train1):
                        output_train.append(label[i - 1] + 1)

            input_train_s = normalize(input_train_s)
            input_train = np.vstack([input_train, input_train_s])


    sub_id = "%02d" % sub_id
    data_path = '/data/ch/data/SEED/seed_ExtractedFeatures/s' + str(sub_id) + '_' + str(section) + '.mat'
    data = sio.loadmat(data_path)
    for i in range(1, 16):
        input_test1 = data[fe + str(i)].transpose((1, 0, 2))
        input_test = np.vstack([input_test, input_test1])
        for j in range(0, len(input_test1), 1):
            if j + seq_length <= len(input_test1):
                output_test.append(label[i - 1] + 1)
    input_test = normalize(input_test)


    output_train = np.array(output_train)
    output_test = np.array(output_test)
    # output_train = np_utils.to_categorical(output_train, num_classes=3)
    # output_test = np_utils.to_categorical(output_test, num_classes=3)
    print(input_train.shape)
    print(input_test.shape)
    print(output_train.shape)
    print(output_test.shape)

    return input_train, input_test, output_train, output_test

def get_cross_sub_data1(sub_id):
    seq_length = 1
    fe = 'de_LDS'
    input_train = np.empty([0,62, 5], dtype=np.float32)
    output_train = list()
    input_test = np.empty([0,62, 5], dtype=np.float32)
    output_test = list()
    label = sio.loadmat("/data/ch/data/SEED/label.mat")
    label = label['label'].reshape(-1)
    for sub in range(1, 15):
        if sub != sub_id:
            sub = "%02d" % sub
            for section in range(1, 4):
                input_train_s = np.empty([0, 62, 5], dtype=np.float32)
                data_path = '/data/ch/data/SEED/seed_ExtractedFeatures/s' + str(sub) + '_' + str(section) + '.mat'
                data = sio.loadmat(data_path)
                for i in range(1, 16):
                    input_train1 = data[fe + str(i)].transpose((1, 0, 2))
                    input_train_s = np.vstack([input_train_s, input_train1])
                    for j in range(0, len(input_train1), 1):
                        if j + seq_length <= len(input_train1):
                            output_train.append(label[i - 1] + 1)

                input_train_s = normalize(input_train_s)
                input_train = np.vstack([input_train, input_train_s])


    sub_id = "%02d" % (sub_id)
    for section in range(1, 4):
        input_test_s = np.empty([0, 62, 5], dtype=np.float32)
        data_path = '/data/ch/data/SEED/seed_ExtractedFeatures/s' + str(sub_id) + '_' + str(section) + '.mat'
        data = sio.loadmat(data_path)
        for i in range(1, 16):
            input_test1 = data[fe + str(i)].transpose((1, 0, 2))
            input_test_s = np.vstack([input_test_s, input_test1])
            for j in range(0, len(input_test1), 1):
                if j + seq_length <= len(input_test1):
                    output_test.append(label[i - 1] + 1)
        input_test_s = normalize(input_test_s)
        input_test = np.vstack([input_test, input_test_s])


    output_train = np.array(output_train)
    output_test = np.array(output_test)

    #input_train = input_train.reshape((-1, 310))
    #input_test = input_test.reshape((-1, 310))
    # output_train = np_utils.to_categorical(output_train, num_classes=3)
    # output_test = np_utils.to_categorical(output_test, num_classes=3)
    print(input_train.shape)
    print(input_test.shape)
    print(output_train.shape)
    print(output_test.shape)

    return input_train, input_test, output_train, output_test




def get_cross_sub_2Ddata(sub_id):
    seq_length = 1
    fe = 'de_LDS'
    input_train = np.empty([0,5, 62], dtype=np.float32)
    output_train = list()
    input_test = np.empty([0,5, 62], dtype=np.float32)
    output_test = list()
    label = sio.loadmat("/data/ch/data/SEED/label.mat")
    label = label['label'].reshape(-1)
    for sub in range(1, 15):
        if sub != sub_id:
            sub = "%02d" % sub
            for section in range(1, 4):
                input_train_s = np.empty([0,5, 62], dtype=np.float32)
                data_path = '/data/ch/data/SEED/seed_ExtractedFeatures/s' + str(sub) + '_' + str(section) + '.mat'
                data = sio.loadmat(data_path)
                for i in range(1, 16):
                    input_train1 = data[fe + str(i)].transpose((1, 2, 0))
                    input_train_s = np.vstack([input_train_s, input_train1])
                    for j in range(0, len(input_train1), 1):
                        if j + seq_length <= len(input_train1):
                            output_train.append(label[i - 1] + 1)

                input_train_s = normalize(input_train_s)
                input_train = np.vstack([input_train, input_train_s])


    sub_id = "%02d" % sub_id
    for section in range(1, 4):
        input_test_s = np.empty([0,5, 62], dtype=np.float32)
        data_path = '/data/ch/data/SEED/seed_ExtractedFeatures/s' + str(sub_id) + '_' + str(section) + '.mat'
        data = sio.loadmat(data_path)
        for i in range(1, 16):
            input_test1 = data[fe + str(i)].transpose((1, 2, 0))
            input_test_s = np.vstack([input_test_s, input_test1])
            for j in range(0, len(input_test1), 1):
                if j + seq_length <= len(input_test1):
                    output_test.append(label[i - 1] + 1)
        input_test_s = normalize(input_test_s)
        input_test = np.vstack([input_test, input_test_s])

    input_train = get_1D_to_2D_data(input_train)
    input_test = get_1D_to_2D_data(input_test)

    output_train = np.array(output_train)
    output_test = np.array(output_test)

    #input_train = input_train.reshape((-1, 1, 62, 5))
    #input_test = input_test.reshape((-1, 1, 62, 5))
    # output_train = np_utils.to_categorical(output_train, num_classes=3)
    # output_test = np_utils.to_categorical(output_test, num_classes=3)
    print(input_train.shape)
    print(input_test.shape)
    print(output_train.shape)
    print(output_test.shape)

    return input_train, input_test, output_train, output_test



def get_data(sub_id):
    seq_length = 1
    fe = 'de_LDS'
    input_train = list()
    output_train = list()
    input_test = list()
    output_test = list()
    label = sio.loadmat("/data/ch/data/SEED/label.mat")
    label = label['label'].reshape(-1)
    sub_id = "%02d" % sub_id
    sub = "s" + str(sub_id)
    data_path1 = '/data/ch/data/SEED/seed_ExtractedFeatures/' + sub + '_1.mat'
    data1 = sio.loadmat(data_path1)
    for i in range(1, 10):
        input_train1 = data1[fe + str(i)].transpose((1, 2, 0))
        for j in range(0, len(input_train1), 1):
            if j + seq_length <= len(input_train1):
                input_train.append(input_train1[j:j + seq_length])
                output_train.append(label[i - 1] + 1)
    for i in range(10, 16):
        input_test1 = data1[fe + str(i)].transpose((1, 2, 0))
        for j in range(0, len(input_test1), 1):
            if j + seq_length <= len(input_test1):
                input_test.append(input_test1[j:j + seq_length])
                output_test.append(label[i - 1] + 1)

    input_train = np.array(input_train, dtype=np.float32)
    output_train = np.array(output_train)
    input_test = np.array(input_test, dtype=np.float32)
    output_test = np.array(output_test)
    # output_train = np_utils.to_categorical(output_train, num_classes=3)
    # output_test = np_utils.to_categorical(output_test, num_classes=3)
    print(input_train.shape)
    print(input_test.shape)
    X_train = input_train.reshape((-1, seq_length, 310))
    X_test = input_test.reshape((-1, seq_length, 310))
    '''
    minda = np.tile(np.min(X_train, axis=2).reshape((X_train.shape[0], X_train.shape[1], 1)),
                    (1, 1, X_train.shape[2]))
    maxda = np.tile(np.max(X_train, axis=2).reshape((X_train.shape[0], X_train.shape[1], 1)),
                    (1, 1, X_train.shape[2]))
    X_train = (X_train - minda) / (maxda - minda)

    minda2 = np.tile(np.min(X_test, axis=2).reshape((X_test.shape[0], X_test.shape[1], 1)),
                     (1, 1, X_test.shape[2]))
    maxda2 = np.tile(np.max(X_test, axis=2).reshape((X_test.shape[0], X_test.shape[1], 1)),
                     (1, 1, X_test.shape[2]))
    X_test = (X_test - minda2) / (maxda2 - minda2)
    '''
    if seq_length == 1:
        X_train = X_train.reshape(-1, 310)
        X_test = X_test.reshape(-1, 310)

    X_train -= np.mean(X_train, axis=0)
    X_train /= np.std(X_train, axis=0)  # 归一化
    X_test -= np.mean(X_test, axis=0)
    X_test /= np.std(X_test, axis=0)  # 归一化
    print(X_train.shape)
    print(X_test.shape)
    print(output_train.shape)
    print(output_test.shape)
    return X_train, X_test, output_train, output_test


def get_cross_session_data(sub_id):
    seq_length = 1
    fe = 'de_LDS'
    input_train = np.empty([0,310], dtype=np.float32)
    output_train = list()
    input_test = np.empty([0,310], dtype=np.float32)
    output_test = list()
    label = sio.loadmat("/data/ch/data/SEED/label.mat")
    label = label['label'].reshape(-1)
    sub_id = "%02d" % sub_id
    sub = "s" + str(sub_id)
    '''
    data_path1 = '/data/ch/data/SEED/seed_ExtractedFeatures/' + sub + '_1.mat'
    data1 = sio.loadmat(data_path1)

    input_train_s = np.empty([0, 310], dtype=np.float32)
    for i in range(1, 16):
        input_train1 = data1[fe + str(i)].transpose((1, 2, 0)).reshape((-1,310))
        input_train_s = np.vstack([input_train_s, input_train1])
        for j in range(0, len(input_train1), 1):
            if j + seq_length <= len(input_train1):
                output_train.append(label[i - 1] + 1)
    input_train_s = normalize(input_train_s)
    input_train = np.vstack([input_train, input_train_s])
    '''
    data_path2 = '/data/ch/data/SEED/seed_ExtractedFeatures/' + sub + '_2.mat'
    data2 = sio.loadmat(data_path2)
    input_train_s = np.empty([0, 310], dtype=np.float32)
    for i in range(1, 16):
        input_train1 = data2[fe + str(i)].transpose((1, 2, 0)).reshape((-1,310))
        input_train_s = np.vstack([input_train_s, input_train1])
        for j in range(0, len(input_train1), 1):
            if j + seq_length <= len(input_train1):
                output_train.append(label[i - 1] + 1)
    input_train_s = normalize(input_train_s)
    input_train = np.vstack([input_train, input_train_s])

    data_path3 = '/data/ch/data/SEED/seed_ExtractedFeatures/' + sub + '_3.mat'
    data3 = sio.loadmat(data_path3)
    for i in range(1, 16):
        input_test1 = data3[fe + str(i)].transpose((1, 2, 0)).reshape((-1,310))
        input_test = np.vstack([input_test, input_test1])
        for j in range(0, len(input_test1), 1):
            if j + seq_length <= len(input_test1):
                output_test.append(label[i - 1] + 1)
    input_test = normalize(input_test)

    output_train = np.array(output_train)
    output_test = np.array(output_test)
    # output_train = np_utils.to_categorical(output_train, num_classes=3)
    # output_test = np_utils.to_categorical(output_test, num_classes=3)
    print(input_train.shape)
    print(input_test.shape)
    print(output_train.shape)
    print(output_test.shape)
    return input_train, input_test, output_train, output_test


def train(model, device, src_dataloader, target_dataloader, criterion, num_epoch, model_path, bs, scheduler=None):
    """
    Implementation of training process.
    See more high-level details on http://sites.skoltech.ru/compvision/projects/grl/files/paper.pdf
    :param model: model which we will train
    :param device: cpu or gpu
    :param src_dataloader: dataloader with source images
    :param target_dataloader:  dataloader with target images
    :param criterion: loss function
    :param optimizer: method for update weights in NN
    :param epoch: current epoch
    :param scheduler: algorithm for changing learning rate
    """
    len_source = len(src_dataloader)
    len_target =  len(target_dataloader)
    len_dataloader = min(len_source, len_target)
    valid_losses = []

    temperature=5
    early_stopping = EarlyStopping(patience=10, verbose=True,path=model_path)

    for epoch in range(num_epoch):
        model.train()
        LR = 0.001
        #print(alpha, LR)

        optimizer = optim.Adam(model.parameters(), lr=LR)
        #optimizer = optim.RMSprop(model.parameters(), lr=LR, alpha=0.9)
        #optimizer = optim.SGD(model.parameters(), lr=LR, weight_decay=0.0005, momentum=0.9)
        #model.train()
        for batch_idx in range(len_dataloader):
            if batch_idx % len_source == 0:
                iter_source = iter(src_dataloader)
            if batch_idx % len_target == 0:
                iter_target = iter(target_dataloader)
            imgs_src, src_class = iter_source.next()

            imgs_target, _ = iter_target.next()

            '''
            if len(imgs_target) != len(imgs_src):
                imgs_src=imgs_src[:len(imgs_target)]
                src_class=src_class[:len(imgs_target)]
            '''

            imgs_src, src_class = imgs_src.to(device), src_class.to(device)
            imgs_target = imgs_target.to(device)


            # alpha=1

            model.zero_grad()

            # train on target domain

            t_class_predict, t_feature = model(imgs_target)
            ''''''
            t_class_predict_temp = t_class_predict / temperature
            target_softmax_out_temp = nn.Softmax(dim=1)(t_class_predict_temp)
            target_entropy_weight = Entropy(target_softmax_out_temp).detach()
            target_entropy_weight = 1 + torch.exp(-target_entropy_weight)
            target_entropy_weight = bs * target_entropy_weight / torch.sum(target_entropy_weight)
            cov_matrix_t = target_softmax_out_temp.mul(target_entropy_weight.view(-1, 1)).transpose(1, 0).mm(
                target_softmax_out_temp)
            # cov_matrix_t = target_softmax_out_temp.transpose(1, 0).mm(target_softmax_out_temp)
            cov_matrix_t = cov_matrix_t / torch.sum(cov_matrix_t, dim=1)
            mcc_loss = (torch.sum(cov_matrix_t) - torch.trace(cov_matrix_t)) / 3

            # train on source domain
            class_predict, s_feature = model(imgs_src)
            src_class_loss = criterion(class_predict, src_class)
            #mmd_loss = mmd_rbf_noaccelerate(s_feature, t_feature)
            # calculating loss
            #loss = src_class_loss + (src_domain_loss + t_domain_loss)
            #loss = src_class_loss + 1*mcc_loss
            #loss = src_class_loss + mmd_loss
            #loss = src_class_loss + mcc_loss + mmd_loss
            loss = src_class_loss

            # print('err_s_label: %f, err_s_domain: %f, err_t_domain: %f,loss: %f' \
            # % (src_class_loss.data.cpu().numpy(),src_domain_loss.data.cpu().numpy(),
            # t_domain_loss.data.cpu().item(),loss.data.cpu().numpy()))
            if scheduler is not None:
                scheduler.step()

            """
            Calculating gradients and update weights
            """
            loss.backward()
            optimizer.step()

            valid_losses.append(src_class_loss.item())

        valid_loss = np.average(valid_losses)
        valid_losses = []
        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        '''
        model.eval()
        accuracy = 0
        for (imgs, labels) in target_dataloader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            prediction, _ = model(imgs)
            pred_cls = prediction.data.max(1)[1]
            accuracy += pred_cls.eq(labels.data).sum().item()


        accuracy /= len(target_dataloader.dataset)
        print(f'Accuracy on SEED-test: {100 * accuracy:.2f}%')
        if accuracy>0.9:
            break
        '''

# 更新混淆矩阵
def conf_matrix(preds, labels, conf_matrix):
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1
    return conf_matrix

def plot_confusion_matrix(cm, labels_name):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]*100    # 归一化
    plt.imshow(cm, interpolation='nearest', cmap='BuGn', vmin=0, vmax=100)
    #plt.title(title)    # 图像标题
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))
    plt.xticks(num_local, labels_name, rotation=90)    # 将标签印在x轴坐标上
    plt.yticks(num_local, labels_name)    # 将标签印在y轴坐标上

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    #plt.xlabel('Predicted Label')
    plt.savefig('GRUMCC_1.png')


def plot_with_labels(lowDWeights, labels,sub_id):
    plt.cla() #clear当前活动的坐标轴
    plt.gca().get_xaxis().set_visible(False)
    plt.gca().get_yaxis().set_visible(False)
    d = lowDWeights[labels==3]
    l1=plt.plot(d[:,0], d[:,1],'r+', label='Source Negative')
    d = lowDWeights[labels == 4]
    l2=plt.plot(d[:, 0], d[:, 1],'ro', label='Source Neutral')
    d = lowDWeights[labels == 5]
    l3=plt.plot(d[:, 0], d[:, 1],'r*', label='Source Positive')
    d = lowDWeights[labels == 0]
    l4=plt.plot(d[:, 0], d[:, 1],'b+', label='Target Negative')
    d = lowDWeights[labels == 1]
    l5=plt.plot(d[:, 0], d[:, 1],'bo', label='Target Neutral')
    d = lowDWeights[labels == 2]
    l6=plt.plot(d[:, 0], d[:, 1],'b*', label='Target Positive')
    plt.legend(loc='best')
    fig_path = 'baseline/g{0}.png'.format(sub_id)
    plt.savefig(fig_path)


def test(model, device, test_loader,train_loader, c_m,sub_id):
    """
    Provide accuracy on test dataset
    :param model: Model of the NN
    :param device: cpu or gpu
    :param test_loader: loader of the test dataset
    :param max: the current max accuracy of the model
    :return: max accuracy for overall observations
    """
    model.eval()

    accuracy = 0
    feature_t = np.empty([0, 512])
    label_t = np.empty([0])

    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    for (imgs, labels) in test_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        prediction, features = model(imgs)
        pred_cls = prediction.data.max(1)[1]
        accuracy += pred_cls.eq(labels.data).sum().item()
        #c_m = conf_matrix(pred_cls, labels=labels, conf_matrix=c_m)
    '''
        feature = features.cpu().data.numpy()
        label = labels.cpu().numpy()
        feature_t = np.vstack([feature_t, feature])
        label_t = np.append(label_t, label)
    
    plot_only = 500
    feature_t = feature_t[:plot_only]
    label_t = label_t[:plot_only]

    for (imgs, labels) in train_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        prediction, features = model(imgs)

        feature = features.cpu().data.numpy()
        label = labels.cpu().numpy() + 3
        feature_t = np.vstack([feature_t, feature])
        label_t = np.append(label_t, label)
    print(feature_t.shape)
    print(label_t.shape)
    feature_t = feature_t[:2*plot_only]
    label_t = label_t[:2*plot_only]
    low_dim_embs = tsne.fit_transform(feature_t)
    plot_with_labels(low_dim_embs, label_t,sub_id)
    '''
    accuracy /= len(test_loader.dataset)

    return accuracy,c_m


if __name__ == '__main__':
    """
    Set up random seed for reproducibility
    """
    np.random.seed(1337)
    torch.manual_seed(1337)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    """
    Set up number of epochs, domain threshold, loss function, device
    """
    EPOCHS = 20
    DOMAIN_THRSH = 0.3
    BATCH_SIZE = 200
    criterion = nn.CrossEntropyLoss()
    device = 'cuda:3' if torch.cuda.is_available() else 'cpu'

    acc = []
    #c_m = np.zeros((3, 3))
    c_m = torch.zeros(3, 3)
    for sub_id in range(1, 15):
        print("processing ", sub_id)
        train_x, test_x, train_y, test_y = get_cross_sub_data1(sub_id)
        #train_x = train_x[:,:,0].reshape((-1, 62,1))
        #test_x = test_x[:, :, 0].reshape((-1, 62,1))
        #train_x = train_x[:, :, 0]
        #test_x = test_x[:, :, 0]
        '''
        index = np.array(range(0, len(train_y)))
        np.random.shuffle(index)
        train_x = train_x[index][:len(test_y)]
        train_y = train_y[index][:len(test_y)]
        cla = svm.SVC(kernel='linear')
        cla.fit(train_x, train_y)
        pred = cla.predict(test_x)
        print(pred.shape)
        accuracy = np.sum(pred == test_y) / len(pred)
        confusion = confusion_matrix(test_y, pred)
        c_m = c_m + confusion
        #c_m = c_m.astype('float') / c_m.sum(axis=1)[:, np.newaxis]
        print (c_m)
        '''
        val_x = test_x[:720]
        val_y = test_y[:720]
        dataset_val = GetLoader(val_x, val_y)
        val_dataloader = torch.utils.data.DataLoader(
            dataset=dataset_val,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=4
        )

        dataset_source = GetLoader(train_x, train_y)
        dataset_target = GetLoader(test_x, test_y)
        source_dataloader = torch.utils.data.DataLoader(
            dataset=dataset_source,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=4
        )
        target_dataloader = torch.utils.data.DataLoader(
            dataset=dataset_target,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=4
        )

        model = GRUModel().to(device)


        accuracy = 0
        model_path = 'models/best_model{0}.pt'.format(sub_id)
        train(model, device, source_dataloader, target_dataloader, criterion, EPOCHS,model_path,BATCH_SIZE)
        #train(model, device, source_dataloader, val_dataloader, criterion, EPOCHS, model_path, BATCH_SIZE)
        print('----------------------------------------')
        del model

        index = np.array(range(0, len(train_y)))
        np.random.shuffle(index)
        train_x = train_x[index][:len(test_y)]
        train_y = train_y[index][:len(test_y)]
        dataset_source = GetLoader(train_x, train_y)
        source_dataloader = torch.utils.data.DataLoader(
            dataset=dataset_source,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=4
        )

        model = GRUModel()
        print(model)
        model.load_state_dict(torch.load(model_path))
        model = model.to(device)
        accuracy,c_m = test(model, device, target_dataloader,source_dataloader,c_m,sub_id)
        del model

        acc.append(accuracy)
        print(acc)
        #print(c_m)

    acc = np.array(acc)
    print(np.mean(acc))
    print(np.std(acc))
    c_m = c_m.numpy()
    #c_m = cm.astype('float') / cm.sum(axis=1)
    #c_m = np.around(cm*100, 2)
    #print(c_m)
    #plot_confusion_matrix(c_m, ['Negative','Neutral','Positive'])