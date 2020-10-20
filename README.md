# dnn
Deep Neural Network Architectures

This repository contains the definitions for the following architectures, organized by task.

## [Classification](./src/classification)

### [AlexNet](./src/classification/alexnet.h)

It contains the definition for the model that started it all.

Papers:
- [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)

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

In particular, it contains DenseNet-{121,169,201,264} definitions.

Papers:
- [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)

### [DarkNet](./src/classification/darknet.h)

In particular, it contains the backbones for DarkNet-19 (introduced in YOLOv1) and DarkNet-53 (YOLOv3).

Papers:
- [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640)
- [YOLOv3: An Incremental Improvement](https://arxiv.org/abs/1804.02767)

### [VoVNet](./src/classification/vovnet.h)
In particular, it contains implementations for VoVNetv2-{19slim,19,27slim,27,39,57,99}, which are very similar to VoVNetv1 (V2 have identiy mapping and effective Squeeze and Excitation on top of V1).

Papers:
- [An Energy and GPU-Computation Efficient Backbone Network for Real-Time Object Detection](https://arxiv.org/abs/1904.09730)
- [CenterMask: Real-Time Anchor-Free Instance Segmentation](https://arxiv.org/abs/1911.06667)
