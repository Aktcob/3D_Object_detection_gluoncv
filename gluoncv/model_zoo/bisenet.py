from __future__ import division
import mxnet as mx
from mxnet import autograd
from mxnet.gluon import nn
from mxnet.context import cpu
from mxnet.gluon.nn import HybridBlock
from .segbase import SegBaseModel
from .resnetv1b import resnet18_v1b
# pylint: disable=unused-argument,abstract-method,missing-docstring,dangerous-default-value

__all__ = ['BISENET', 'get_bisenet']


class ConvBnRelu(HybridBlock):
    def __init__(self, in_planes, out_planes, ksize, stride, pad, dilation=1,
                 groups=1, has_bn=True, norm_layer=nn.BatchNorm, bn_eps=1e-5,
                 has_relu=True, inplace=True, has_bias=False):
        super(ConvBnRelu, self).__init__()
        with self.name_scope():
            self.conv = nn.Conv2D(in_channels=in_planes, channels=out_planes, kernel_size=ksize, strides=stride,
                                padding=pad, dilation=dilation, groups=groups, use_bias=has_bias)
            self.has_bn = has_bn
            if self.has_bn:
                self.bn = norm_layer(in_channels=out_planes)
            self.has_relu = has_relu
            if self.has_relu:
                self.relu = nn.Activation('relu')

    def hybrid_forward(self, F, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)

        return x
        

class SpatialPath(HybridBlock):
    # def __init__(self):
    def __init__(self, in_planes, out_planes, norm_layer=nn.BatchNorm):
        super(SpatialPath, self).__init__()
        inner_channel = 64
        with self.name_scope():
            self.conv_7x7 = ConvBnRelu(in_planes, inner_channel, 7, 2, 3,
                                    has_bn=True, norm_layer=norm_layer,
                                    has_relu=True, has_bias=False)
            self.conv_3x3_1 = ConvBnRelu(inner_channel, inner_channel, 3, 2, 1,
                                            has_bn=True, norm_layer=norm_layer,
                                            has_relu=True, has_bias=False)
            self.conv_3x3_2 = ConvBnRelu(inner_channel, inner_channel, 3, 2, 1,
                                            has_bn=True, norm_layer=norm_layer,
                                            has_relu=True, has_bias=False)
            self.conv_1x1 = ConvBnRelu(inner_channel, out_planes, 1, 1, 0,
                                        has_bn=True, norm_layer=norm_layer,
                                        has_relu=True, has_bias=False)
        
    def hybrid_forward(self, F, x):
        x = self.conv_7x7(x)
        x = self.conv_3x3_1(x)
        x = self.conv_3x3_2(x)
        output = self.conv_1x1(x)

        return output


class ARM(HybridBlock):
    # def __init__(self, channels):
    def __init__(self, in_planes, out_planes, norm_layer=nn.BatchNorm):    
        super(ARM, self).__init__()
        with self.name_scope():
            self.conv_3x3 = ConvBnRelu(in_planes, out_planes, 3, 1, 1,
                                        has_bn=True, norm_layer=norm_layer,
                                        has_relu=True, has_bias=False)
            self.conv_1x1 = ConvBnRelu(out_planes, out_planes, 1, 1, 0,
                                        has_bn=True, norm_layer=norm_layer,
                                        has_relu=False, has_bias=False)

            self.act = nn.Activation('sigmoid')

    def hybrid_forward(self, F, x):
        x = self.conv_3x3(x)
        w = F.contrib.AdaptiveAvgPooling2D(x, output_size=1)
        w = self.conv_1x1(w)
        w = self.act(w)

        return F.broadcast_mul(x, w)



class FFM(HybridBlock):
    # def __init__(self, channels):
    def __init__(self, in_planes, out_planes,
                 reduction=1, norm_layer=nn.BatchNorm):
        super(FFM, self).__init__()
        with self.name_scope():
            self.conv_1x1 = ConvBnRelu(in_planes, out_planes, 1, 1, 0,
                                        has_bn=True, norm_layer=norm_layer,
                                        has_relu=True, has_bias=False)
                
            self.conv_channel_1 = ConvBnRelu(out_planes, out_planes // reduction, 1, 1, 0,
                                                has_bn=False, norm_layer=norm_layer,
                                                has_relu=True, has_bias=False)
            self.conv_channel_2 = ConvBnRelu(out_planes // reduction, out_planes, 1, 1, 0,
                                    has_bn=False, norm_layer=norm_layer,
                                    has_relu=False, has_bias=False)

            self.act = nn.Activation('sigmoid')


    def hybrid_forward(self, F, x1, x2):
        feature = F.concat(x1, x2, dim=1)
        feature = self.conv_1x1(feature)

        feature_se = F.contrib.AdaptiveAvgPooling2D(feature, output_size=1)
        feature_se = self.conv_channel_1(feature_se)
        feature_se = self.conv_channel_2(feature_se)
        feature_se = self.act(feature_se)

        # out = feature + F.broadcast_mul(feature, feature_se)
        out = F.broadcast_add(feature, F.broadcast_mul(feature, feature_se))

        return out


class ContextPath(HybridBlock):
    def __init__(self, height, width, aux=True, pretrained_base=True, ctx=cpu(), norm_layer=nn.BatchNorm):
        super(ContextPath, self).__init__()
        self.height = height
        self.width = width
        self.aux = aux
        conv_channel = 128

        with self.name_scope():
            self.refine_16 = ConvBnRelu(conv_channel, conv_channel, 3, 1, 1,
                                            has_bn=True, norm_layer=norm_layer,
                                            has_relu=True, has_bias=False)
            self.refine_16.initialize(ctx=ctx)
            self.refine_16.collect_params().setattr('lr_mult', 10)

            self.refine_32 = ConvBnRelu(conv_channel, conv_channel, 3, 1, 1,
                                            has_bn=True, norm_layer=norm_layer,
                                            has_relu=True, has_bias=False)
            self.refine_32.initialize(ctx=ctx)
            self.refine_32.collect_params().setattr('lr_mult', 10)

            self.global_conv = ConvBnRelu(512, conv_channel, 1, 1, 0,
                                        has_bn=True,
                                        has_relu=True, has_bias=False, norm_layer=norm_layer)
            self.global_conv.initialize(ctx=ctx)

            self.x16_arm = ARM(256, conv_channel, norm_layer)
            self.x16_arm.initialize(ctx=ctx)
            self.x16_arm.collect_params().setattr('lr_mult', 10)

            self.x32_arm = ARM(512, conv_channel, norm_layer)
            self.x32_arm.initialize(ctx=ctx)
            self.x32_arm.collect_params().setattr('lr_mult', 10)

            # with self.name_scope():
            pretrained = resnet18_v1b(pretrained=pretrained_base, dilated=False, ctx=ctx)
            self.conv1 = pretrained.conv1
            self.bn1 = pretrained.bn1
            self.relu = pretrained.relu
            self.maxpool = pretrained.maxpool
            self.layer1 = pretrained.layer1
            self.layer2 = pretrained.layer2
            self.layer3 = pretrained.layer3
            self.layer4 = pretrained.layer4

            self.global_pool = nn.GlobalAvgPool2D()
            self.global_pool.initialize(ctx=ctx)

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)

        feature_x8 = self.layer2(x)
        feature_x16 = self.layer3(feature_x8)
        feature_x32 = self.layer4(feature_x16)

        # center = self.global_context(feature_x32)
        center = self.global_pool(feature_x32)
        center = self.global_conv(center)

        feature_arm_x32 = self.x32_arm(feature_x32)
        # feature_x32 = center + feature_arm_x32
        feature_x32 = F.broadcast_add(center, feature_arm_x32)

        feature_x32 = F.contrib.BilinearResize2D(feature_x32, height=int(self.height/16), width=int(self.width/16))
        feature_x32 = self.refine_32(feature_x32)
        
        


        feature_arm_x16 = self.x16_arm(feature_x16)
        # feature_x16 = feature_x32 + feature_arm_x16
        feature_x16 = F.broadcast_add(feature_x32, feature_arm_x16)
        
        feature_x16 = F.contrib.BilinearResize2D(feature_x16, height=int(self.height/8), width=int(self.width/8))
        feature_x16 = self.refine_16(feature_x16)
        

        context_out = feature_x16

        if self.aux:
            return feature_x32, context_out
        else:
            return context_out


class BISENET(HybridBlock):
    def __init__(self, nclass=21, backbone='resnet18', aux=True, ctx=cpu(0), pretrained_base=True,
                 crop_size=480, norm_layer=nn.BatchNorm,  **kwargs):
        super(BISENET, self).__init__()
        self.height = crop_size
        self.width = crop_size      
        self.aux = aux
        print('self.crop_size', crop_size)
        with self.name_scope():
            self.spatial_path = SpatialPath(3, 128, norm_layer)
            self.spatial_path.initialize(ctx=ctx)

            self.context_path = ContextPath(self.height, self.width, aux=self.aux, ctx=ctx)

            self.ffm = FFM(256, 256, 4)
            self.ffm.initialize(ctx=ctx)
            self.ffm.collect_params().setattr('lr_mult', 10)

            self.pred_out = _BiseHead(nclass, 256, 64)
            self.pred_out.initialize(ctx=ctx)
            self.pred_out.collect_params().setattr('lr_mult', 10)

            if self.aux:
                self.aux_stage3 = _BiseHead(nclass, 128, 256)
                self.aux_stage3.initialize(ctx=ctx)
                self.aux_stage3.collect_params().setattr('lr_mult', 10)

                self.aux_stage4 =  _BiseHead(nclass, 128, 256)
                self.aux_stage4.initialize(ctx=ctx)
                self.aux_stage4.collect_params().setattr('lr_mult', 10)


    def hybrid_forward(self, F, x):
        spatial_out = self.spatial_path(x)

        if self.aux:
            feature_x32, context_out = self.context_path(x)
        else:
            context_out = self.context_path(x)

        feature = self.ffm(spatial_out,context_out)
        # print(feature.shape)

        feature = self.pred_out(feature)

        outputs = []
        bisenet_out = F.contrib.BilinearResize2D(feature, height=self.height, width=self.width)
        outputs.append(bisenet_out)

        if autograd.is_training():
            if self.aux:
                aux_stage3_out = self.aux_stage3(context_out)
                aux_stage3_out = F.contrib.BilinearResize2D(aux_stage3_out, height=self.height, width=self.width)
                outputs.append(aux_stage3_out)


                aux_stage4_out = self.aux_stage4(feature_x32)
                aux_stage4_out = F.contrib.BilinearResize2D(aux_stage4_out, height=self.height, width=self.width)
                outputs.append(aux_stage4_out)
            return tuple(outputs)

        return tuple(outputs)

    def evaluate(self, x):
        """evaluating network with inputs and targets"""
        return self.forward(x)

class _BiseHead(HybridBlock):
    def __init__(self, nclass, in_planes, out_planes, 
                 norm_layer=nn.BatchNorm):
        super(_BiseHead, self).__init__()
        with self.name_scope():
            self.conv_3x3 = ConvBnRelu(in_planes, out_planes, 3, 1, 1,
                                    has_bn=True, norm_layer=norm_layer,
                                    has_relu=True, has_bias=False)
            self.conv_1x1 = nn.Conv2D(in_channels=out_planes, channels=nclass, kernel_size=1,
                                    strides=1, padding=0)

    def hybrid_forward(self, F, x):
        x = self.conv_3x3(x)
        return self.conv_1x1(x)


def get_bisenet(dataset='pascal_voc', backbone='resnet50', pretrained=False,
            root='~/.mxnet/models', ctx=cpu(0), pretrained_base=True, **kwargs):

    acronyms = {
        'pascal_voc': 'voc',
        'pascal_aug': 'voc',
        'ade20k': 'ade',
        'coco': 'coco',
        'citys': 'citys',
    }
    from ..data import datasets
    # infer number of classes
    print ("pretrained_base", pretrained_base)
    model = BISENET(datasets[dataset].NUM_CLASS, backbone=backbone,
                   pretrained_base=pretrained_base, ctx=ctx, **kwargs)
    if pretrained:
        from .model_store import get_model_file
        model.load_parameters(get_model_file('psp_%s_%s'%(backbone, acronyms[dataset]),
                                             tag=pretrained, root=root), ctx=ctx)
    return model