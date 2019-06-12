import os
import shutil
import argparse
import numpy as np
from tqdm import tqdm
from lib.Dataset import *
import mxnet as mx
from mxnet import gluon, autograd, nd, initializer
from mxnet.gluon.data.vision import transforms
from mxnet.gluon.loss import *
import gluoncv
from gluoncv.loss import *
from gluoncv.utils import LRScheduler
from gluoncv.model_zoo.segbase import *
from gluoncv.model_zoo import get_model     
from gluoncv.utils.parallel import *
from gluoncv.data import get_segmentation_dataset
from gluoncv import utils as gutils
from model import *



ctx = mx.gpu(0)

def OrientationLoss(orient_batch, orientGT_batch, confGT_batch):
    batch_size,_,_ = orient_batch.shape
    indexes = nd.argmax(confGT_batch, axis=1)
    # extract just the important bin
    orientGT_batch = orientGT_batch[nd.arange(batch_size,ctx=ctx), indexes]
    orient_batch = orient_batch[nd.arange(batch_size,ctx=ctx), indexes]
    theta_diff = nd.arctan(orientGT_batch[:,1]/orientGT_batch[:,0])
    estimated_theta_diff = nd.arctan(orient_batch[:,1]/orient_batch[:,0])

    return -1 * nd.cos(theta_diff - estimated_theta_diff).mean()


def main():
    pretrain = False
    first_epoch = 0
    epochs = 100
    batch_size = 8
    alpha = 0.6
    w = 0.4

    train_path = os.path.abspath(os.path.dirname(__file__)) + '/Kitti/training'
    dataset = Dataset(train_path)
    params = {'batch_size': batch_size,
              'shuffle': True,
              'num_workers': 6}

    generator = gluon.data.DataLoader(dataset, batch_size, shuffle=True, last_batch='rollover', num_workers=6)
    model = mxnet3dbox()

    trainer = gluon.Trainer(model.collect_params(),'sgd', {'learning_rate': 0.0001, 'wd': 1e-5, 'momentum': 0.9})
    conf_loss_func = SoftmaxCrossEntropyLoss()
    dim_loss_func = L2Loss()

    total_num_batches = int(len(dataset) / batch_size)
    print total_num_batches
    for epoch in range(first_epoch, epochs):
        curr_batch = 0
        passes = 0
        sumloss = 0
        for local_batch, truth_orient, truth_conf, truth_dim in generator:
            local_batch = nd.array(local_batch,ctx=mx.gpu(0))
            truth_orient = nd.array(truth_orient,ctx=mx.gpu(0))
            truth_conf = nd.array(truth_conf,ctx=mx.gpu(0))
            truth_dim = nd.array(truth_dim,ctx=mx.gpu(0))
            with autograd.record():
                [orient, conf, dim] = model(local_batch)
                orient_loss = OrientationLoss(orient, truth_orient, truth_conf)
                dim_loss = dim_loss_func(dim, truth_dim)
                truth_conf = nd.argmax(truth_conf, axis=1)        
                conf_loss = conf_loss_func(conf, truth_conf)
                loss_theta = conf_loss + w * orient_loss
                loss = alpha * dim_loss + loss_theta
            loss.backward()
            trainer.step(batch_size)
            # print loss.mean().asnumpy()
            sumloss += loss.mean().asnumpy()
            if passes % 200 == 0:
                print("--- epoch %s | batch %s/%s --- [loss: %s]" %(epoch, curr_batch, total_num_batches, sumloss/curr_batch))
                passes = 0
            passes+=1
            curr_batch += 1
        # save after every 10 epochs
        if epoch % 10 == 0:
            name = 'epoch_%s' % epoch
            model.save_parameters(name)

if __name__=='__main__':
    main()
