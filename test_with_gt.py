import time
import os
import cv2
import mxnet as mx
from model import *
import numpy as np
from lib.Dataset import *
from library.Math import *
from library.Plotting import *
from lib import ClassAverages

ctx = mx.gpu(0)

def plot_regressed_3d_bbox(img, truth_img, cam_to_img, box_2d, dimensions, alpha, theta_ray):

    # the math! returns X, the corners used for constraint
    location, X = calc_location(dimensions, cam_to_img, box_2d, alpha, theta_ray)

    orient = alpha + theta_ray

    plot_2d_box(truth_img, box_2d)
    plot_3d_box(img, cam_to_img, orient, dimensions, location) # 3d boxes

    return location

def main():

    model = mxnet3dbox()
    path = 'epoch'
    model.load_parameters(path, ctx=ctx)

    dataset = Dataset(os.path.abspath(os.path.dirname(__file__)) + '/eval')
    # print dataset
    averages = ClassAverages.ClassAverages()

    all_images = dataset.all_objects()

    for key in sorted(all_images.keys()):
        start = time.time()
        data = all_images[key]

        truth_img = data['Image']
        img = np.copy(truth_img)
        objects = data['Objects']
        cam_to_img = data['Calib']

        for object in objects:
            label = object.label
            theta_ray = object.theta_ray
            input_img = object.img
            input_tensor = nd.expand_dims(nd.array(input_img, ctx=ctx),axis=0)
            [orient, conf, dim] = model(input_tensor)
            orient, conf, dim = nd.squeeze(orient).asnumpy(), nd.squeeze(conf).asnumpy(),nd.squeeze(dim).asnumpy()

            dim += averages.get_item(label['Class'])
            argmax = np.argmax(conf)
            orient = orient[argmax, :]
            cos = orient[0]
            sin = orient[1]
            tan = sin/cos
            # alpha = np.arctan2(sin, cos)
            alpha = np.arctan(tan)
            alpha += dataset.angle_bins[argmax]
            alpha -= np.pi
            location = plot_regressed_3d_bbox(img, truth_img, cam_to_img, label['Box_2D'], dim, alpha, theta_ray)

        numpy_vertical = np.concatenate((truth_img, img), axis=0)
        cv2.imshow('2D detection on top, 3D prediction on bottom', numpy_vertical)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
        over = time.time()
        print ('processing time is: ', over-start, 'ms') 

if __name__ == '__main__':
    main()
