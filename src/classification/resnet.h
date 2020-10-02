#ifndef ResNet_H
#define ResNet_H

#include <dlib/dnn.h>

namespace resnet
{
    // clang-format off
    using namespace dlib;
    // ACT can be any activation layer and BN must be bn_con or affine layer
    template <template <typename> class ACT, template <typename> class BN>
    struct def
    {
        template <long num_filters, long ks, int s, int p, typename SUBNET>
        using conp = add_layer<con_<num_filters, ks, ks, s, s, p, p>, SUBNET>;

        // the resnet basic block, factor must be 1, used only for interface
        template <long num_filters, long factor, int stride, typename SUBNET>
        using basicblock = BN<conp<num_filters * factor, 3, 1, 1,
                           ACT<BN<conp<num_filters, 3, stride, 1, SUBNET>>>>>;

        // the resnet bottleneck block, factor must be 4, used only for interface
        template <long num_filters, long factor, int stride, typename SUBNET>
        using bottleneck = BN<conp<num_filters * factor, 1, 1, 0,
                           ACT<BN<conp<num_filters, 3, stride, 1,
                           ACT<BN<conp<num_filters, 1, 1, 0, SUBNET>>>>>>>>;

        // the resnet residual, where BLOCK is either basicblock or bottleneck
        template <template <long, long, int, typename> class BLOCK, long num_filters, long factor, typename SUBNET>
        using residual = add_prev1<BLOCK<num_filters, factor, 1, tag1<SUBNET>>>;

        // a resnet residual that does subsampling on both paths
        template <template <long, long, int, typename> class BLOCK, long num_filters, long factor, typename SUBNET>
        using residual_down = add_prev2<BN<conp<num_filters * factor, 1, 2, 0,
                              skip1<tag2<BLOCK<num_filters, factor, 2, tag1<SUBNET>>>>>>>;

        // residual block that takes either a residual or a residual_down
        template<
            template <template<long, long, int, typename> class, long, long, typename> class RESIDUAL,
            template <long, long, int, typename> class BLOCK,
            long num_filters,
            long factor,
            typename SUBNET>
        using residual_block = ACT<RESIDUAL<BLOCK, num_filters, factor, SUBNET>>;

        template <long num_filters, typename SUBNET>
        using resbasicblock_down = residual_block<residual_down, basicblock, num_filters, 1, SUBNET>;
        template <long num_filters, typename SUBNET>
        using resbottleneck_down = residual_block<residual_down, bottleneck, num_filters, 4, SUBNET>;

        // some definitions to allow the use of the repeat layer
        template <typename SUBNET> using resbasicblock_512 = residual_block<residual, basicblock, 512, 1, SUBNET>;
        template <typename SUBNET> using resbasicblock_256 = residual_block<residual, basicblock, 256, 1, SUBNET>;
        template <typename SUBNET> using resbasicblock_128 = residual_block<residual, basicblock, 128, 1, SUBNET>;
        template <typename SUBNET> using resbasicblock_64  = residual_block<residual, basicblock,  64, 1, SUBNET>;
        template <typename SUBNET> using resbottleneck_512 = residual_block<residual, bottleneck, 512, 4, SUBNET>;
        template <typename SUBNET> using resbottleneck_256 = residual_block<residual, bottleneck, 256, 4, SUBNET>;
        template <typename SUBNET> using resbottleneck_128 = residual_block<residual, bottleneck, 128, 4, SUBNET>;
        template <typename SUBNET> using resbottleneck_64  = residual_block<residual, bottleneck,  64, 4, SUBNET>;

        // common processing for standard resnet inputs
        template <typename INPUT> using stem = max_pool<3, 3, 2, 2, ACT<BN<conp<64, 7, 2, 3, INPUT>>>>;

        // the resnet backbone with basicblocks
        template <long nb_512, long nb_256, long nb_128, long nb_64, typename INPUT>
        using backbone_basicblock = repeat<nb_512, resbasicblock_512, resbasicblock_down<512,
                                    repeat<nb_256, resbasicblock_256, resbasicblock_down<256,
                                    repeat<nb_128, resbasicblock_128, resbasicblock_down<128,
                                    repeat<nb_64,  resbasicblock_64, stem<INPUT>>>>>>>>;

        // the resnet backbone with bottlenecks
        template <long nb_512, long nb_256, long nb_128, long nb_64, typename INPUT>
        using backbone_bottleneck = repeat<nb_512, resbottleneck_512, resbottleneck_down<512,
                                    repeat<nb_256, resbottleneck_256, resbottleneck_down<256,
                                    repeat<nb_128, resbottleneck_128, resbottleneck_down<128,
                                    repeat<nb_64,  resbottleneck_64, stem<INPUT>>>>>>>>;

        // the backbones for the classic architectures
        template <typename INPUT> using backbone_18  = backbone_basicblock<1, 1, 1, 2, INPUT>;
        template <typename INPUT> using backbone_34  = backbone_basicblock<2, 5, 3, 3, INPUT>;
        template <typename INPUT> using backbone_50  = backbone_bottleneck<2, 5, 3, 3, INPUT>;
        template <typename INPUT> using backbone_101 = backbone_bottleneck<2, 22, 3, 3, INPUT>;
        template <typename INPUT> using backbone_152 = backbone_bottleneck<2, 35, 7, 3, INPUT>;
    };

    template <typename SUBNET>
    using classification_head = loss_multiclass_log<fc<1000, avg_pool_everything<SUBNET>>>;

    // the typical classification models
    using train_18  = classification_head<def<relu, bn_con>::backbone_18<input_rgb_image>>;
    using infer_18  = classification_head<def<relu, affine>::backbone_18<input_rgb_image>>;
    using train_34  = classification_head<def<relu, bn_con>::backbone_34<input_rgb_image>>;
    using infer_34  = classification_head<def<relu, affine>::backbone_34<input_rgb_image>>;
    using train_50  = classification_head<def<relu, bn_con>::backbone_50<input_rgb_image>>;
    using infer_50  = classification_head<def<relu, affine>::backbone_50<input_rgb_image>>;
    using train_101 = classification_head<def<relu, bn_con>::backbone_101<input_rgb_image>>;
    using infer_101 = classification_head<def<relu, affine>::backbone_101<input_rgb_image>>;
    using train_152 = classification_head<def<relu, bn_con>::backbone_152<input_rgb_image>>;
    using infer_152 = classification_head<def<relu, affine>::backbone_152<input_rgb_image>>;
    // clang-format on
}  // namespace resnet

#endif  // ResNet_H
