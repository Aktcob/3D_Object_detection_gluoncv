import os,sys
import numpy as np
import mxnet as mx
from mxnet import gluon
from mxnet import symbol
from mxnet.gluon import nn
from mxnet import initializer
from mxnet import ndarray as nd
from gluoncv.model_zoo.vgg import vgg16_bn

ctx = mx.gpu(0)
# model
class mxnet3dbox(nn.HybridBlock):
    def __init__(self,bins=2,**kwargs):
        super(TDbox,self).__init__(**kwargs)
        pretrained = vgg16_bn(pretrained=True, ctx=ctx)
        self.features = pretrained.features
        self.orientation = nn.HybridSequential()
        self.confidence = nn.HybridSequential()
        self.dimension = nn.HybridSequential()
        self.bins = bins
        with self.orientation.name_scope():
            self.orientation.add(nn.Dense(256,activation='relu'))     # 512*8*8 --> 256
            self.orientation.add(nn.Dropout(rate=0.5))
            self.orientation.add(nn.Dense(256,activation='relu'))     # 256 --> 256
            self.orientation.add(nn.Dropout(rate=0.5))
            self.orientation.add(nn.Dense(bins*2))                    # 256 --> bin *2 (cos, sin)

        with self.confidence.name_scope():
            self.confidence.add(nn.Dense(256,activation='relu'))     # 512*8*8 --> 256
            self.confidence.add(nn.Dropout(rate=0.5))
            self.confidence.add(nn.Dense(256,activation='relu'))     # 256 --> 256
            self.confidence.add(nn.Dropout(rate=0.5))
            self.confidence.add(nn.Dense(bins))                      # 256 --> bin

        with self.dimension.name_scope():
            self.dimension.add(nn.Dense(512,activation='relu'))     # 512*8*8 --> 512
            self.dimension.add(nn.Dropout(rate=0.5))
            self.dimension.add(nn.Dense(512,activation='relu'))     # 512 --> 512
            self.dimension.add(nn.Dropout(rate=0.5))
            self.dimension.add(nn.Dense(3))                         # 512 --> 3
        self.orientation.initialize(init=initializer.Xavier(),ctx=ctx)
        self.confidence.initialize(init=initializer.Xavier(),ctx=ctx)
        self.dimension.initialize(init=initializer.Xavier(),ctx=ctx)

    def hybrid_forward(self, F, x):
        x = self.features(x)
        orientation = self.orientation(x)
        orientation = F.reshape(orientation,shape=(-1,self.bins,2))
        length = F.norm(orientation,axis=2,keepdims=True)
        orientation = F.broadcast_div(orientation, length) 
        confidence = self.confidence(x)
        dimension = self.dimension(x)
        return orientation,confidence,dimension

if __name__ == '__main__':
    image = nd.random.normal(shape=(2,3,224,224),ctx=ctx)
    net = TDbox()
    print(net(image))
