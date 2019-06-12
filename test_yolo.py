import os
import cv2
import mxnet
import time
import numpy as np
from mxnet import gluon
from gluoncv import model_zoo, data, utils
from mxnet import nd, image
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms
from lib.Dataset import *
from library.Math import *
from library.Plotting import *
from lib import ClassAverages
from model import *
import argparse

transform_fn = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([.485, .456, .406], [.229, .224, .225])
])


parser = argparse.ArgumentParser()

parser.add_argument("--image-dir", default="eval/image_2/",
                    help="Relative path to the directory containing images to detect. Default \
                    is eval/image_2/")
parser.add_argument("--cal-dir", default="camera_cal/",
                    help="Relative path to the directory containing camera calibration form KITTI. \
                    Default is camera_cal/")
parser.add_argument("--model_path", default="epoch_90",
                    help="3dbbox location")

def plot_regressed_3d_bbox(img, cam_to_img, box_2d, dimensions, alpha, theta_ray, img_2d=None):
    # the math! returns X, the corners used for constraint
    location, X = calc_location(dimensions, cam_to_img, box_2d, alpha, theta_ray)
    orient = alpha + theta_ray
    if img_2d is not None:
        plot_2d_box(img_2d, box_2d)
    # print "location is ",location
    plot_3d_box(img, cam_to_img, orient, dimensions, location) # 3d boxes

    return location

def main():

    args = parser.parse_args()
    # load 3d model
    model = mxnet3dbox()
    model.load_parameters(args.model_path, ctx=ctx)

    net = model_zoo.get_model('yolo3_darknet53_voc', pretrained=True, ctx=ctx)
    # net.load_parameters(path, ctx=ctx)
    averages = ClassAverages.ClassAverages()

    angle_bins = generate_bins(2)

    image_dir = args.image_dir
    cal_dir = args.cal_dir

    img_path = os.path.abspath(os.path.dirname(__file__)) + "/" + image_dir
    # using P_rect from global calibration file
    calib_path = os.path.abspath(os.path.dirname(__file__)) + "/" + cal_dir
    calib_file = calib_path + "calib_cam_to_cam.txt"

    try:
        ids = [x.split('.')[0] for x in sorted(os.listdir(img_path))]
    except:
        print("\nError: no images in %s"%img_path)
        exit()

    for id in ids:

        start_time = time.time()

        img_file = img_path + id + ".png"
        truth_img = cv2.imread(img_file)
        h1,w1,_ = truth_img.shape
        img = np.copy(truth_img)

        nd_truth_img = nd.array(truth_img)
        imgs = [nd_truth_img]
        x, imgs = data.transforms.presets.yolo.transform_test(imgs, short=512, max_size=1024, stride=1, 
                                                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        x = nd.array(x,ctx=mxnet.gpu())
        # print imgs
        h2,w2,_ = imgs.shape
        rate = (float(h1)/float(h2) + float(w1)/float(w2))/2
        class_IDs, scores, bounding_boxs = net(x)
        class_IDs, scores, bounding_boxs = nd.squeeze(class_IDs).asnumpy(),
                                        nd.squeeze(scores).asnumpy(),nd.squeeze(bounding_boxs).asnumpy()
        length,_ = bounding_boxs.shape
        for i in range(length):
            class_ID = 'DontCare'
            if class_IDs[i]==14: class_ID = 'Pedestrian' 
            if class_IDs[i]==6: class_ID = 'Car' 
            if scores[i] < 0.7:
                continue
            bounding_box = bounding_boxs[i,:]*rate
            bounding_box = np.reshape(bounding_box,[2,2]).astype(np.int)

            if not averages.recognized_class(class_ID):  
                continue
            try:
                object = DetectedObject(img, class_ID, bounding_box, calib_file)
            except:
                continue
            theta_ray = object.theta_ray
            input_img = object.img
            proj_matrix = object.proj_matrix
            box_2d = bounding_box
            detected_class = class_ID

            input_tensor = nd.expand_dims(nd.array(input_img, ctx=ctx),axis=0)
            [orient, conf, dim] = model(input_tensor)

            orient, conf, dim = nd.squeeze(orient).asnumpy(), nd.squeeze(conf).asnumpy(),nd.squeeze(dim).asnumpy()

            dim += averages.get_item(detected_class)

            argmax = np.argmax(conf)
            orient = orient[argmax, :]
            cos = orient[0]
            sin = orient[1]
            tan = sin/cos
            alpha = np.arctan(tan)
            alpha += angle_bins[argmax]
            alpha -= np.pi
            location = plot_regressed_3d_bbox(img, proj_matrix, box_2d, dim, alpha, theta_ray)

        cv2.imshow('3D detections', img)
        if cv2.waitKey(1000) & 0xff == ord('q'):
            break
        if cv2.waitKey(0) != 32: # space bar
            exit()

        print('running time %.3f seconds'%( time.time() - start_time))

if __name__ == '__main__':
    main()
