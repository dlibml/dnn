# dnn
Deep Neural Network Architectures

This repository contains the definitions for the following architectures, organized by task

## [Classification](./src/classification)

### [ResNet](./src/classification/resnet.h)

In particular, it contains ResNet-{18,34,50,101,152}-B definitions, in contrast to dlib, which contains the A variants.

Papers:
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385).

### [DarkNet](./src/classification/darknet.h)

In particular, it contains the backbones for DarkNet-19 (introduced in YOLOv1) and DarkNet-53 (YOLOv3)

Papers:
- [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640)
- [YOLOv3: An Incremental Improvement](https://arxiv.org/abs/1804.02767)

### [VoVNet](./src/classification/vovnet.h)
In particular, it contains implementations for VoVNetv2-{19slim,19,39,57,99}, which are very similar to VoVNetv2.

Papers:
- [An Energy and GPU-Computation Efficient Backbone Network for Real-Time Object Detection](https://arxiv.org/abs/1904.09730)
- [CenterMask: Real-Time Anchor-Free Instance Segmentation](https://arxiv.org/abs/1911.06667)
