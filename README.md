# dnn
Deep Neural Network Architectures

This repository contains the definitions for the following architectures, organized by task.

## Contents
- [Classification](#classification)
  - [AlexNet](#alexnet)
  - [SqueezeNet](#squeezenet)
  - [VGGNet](#vggnet)
  - [GoogLeNet](#googlenet)
  - [ResNet](#resnet)
  - [DenseNet](#densenet)
  - [DarkNet](#darknet)
  - [VoVNet](#vovnet)
  - [RepVGG](#repvgg)

## [Classification](./src/classification)

### [AlexNet](./src/classification/alexnet.h)

It contains the definition for the model that started it all.

Papers:
- [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)

### [SqueezeNet](./src/classification/squeezenet.h)

In particular, it contains SqueezeNet-{v1.0,v1.1}.

Papers:
- [SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size](https://arxiv.org/abs/1602.07360)

### [VGGNet](./src/classification/vggnet.h)

In particular, it contains VGGNet-{11,13,16,19} variants with batch normalization.

Papers:
- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)

### [GoogLeNet](./src/classification/googlenet.h)

It contains the definition of the GoogLeNet, also known as InceptionV1.

Papers:
- [Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842)

### [ResNet](./src/classification/resnet.h)

In particular, it contains ResNet-{18,34,50,101,152}-B definitions, in contrast to dlib, which contains the A variants.

Papers:
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

### [DenseNet](./src/classification/densenet.h)

In particular, it contains DenseNet-{121,169,201,264,161} definitions.

Papers:
- [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)

### [DarkNet](./src/classification/darknet.h)

In particular, it contains the backbones for DarkNet-19 (introduced in YOLOv1), DarkNet-53 (YOLOv3) and CSPDarknet-53 (YOLOv4).

Papers:
- [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640)
- [YOLOv3: An Incremental Improvement](https://arxiv.org/abs/1804.02767)
- [CSPNet: A New Backbone that can Enhance Learning Capability of CNN](https://arxiv.org/abs/1911.11929)
- [YOLOv4: Optimal Speed and Accuracy of Object Detection](https://arxiv.org/abs/2004.10934)

### [VoVNet](./src/classification/vovnet.h)

In particular, it contains implementations for VoVNetv2-{19slim,19,27slim,27,39,57,99}, which are very similar to VoVNetv1 (V2 have identiy mapping and effective Squeeze and Excitation on top of V1).

Papers:
- [An Energy and GPU-Computation Efficient Backbone Network for Real-Time Object Detection](https://arxiv.org/abs/1904.09730)
- [CenterMask: Real-Time Anchor-Free Instance Segmentation](https://arxiv.org/abs/1911.06667)

### [RepVGG](./src/classification/repvgg.h)

In particular, it contains implementations for RepVGG-{A0,A1,A2,B0,B1,B2,B3}.

Note that, at the moment, there is no way to convert from a trained RepVGG model into its inference counterpart.
I will investigate how to do that soon.

Papers:
- [RepVGG: Making VGG-style ConvNets Great Again](https://arxiv.org/abs/2101.03697)

## [Detection](./src/detection)

### [YOLOv5(./src/detection/yolov5.h)

In particular, it contains implementations for YOLOv5{n,s,m,l,x}, which match the ones in [ultralytics/yolov5](https://github.com/ultralytics/yolov5).
