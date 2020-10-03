#ifndef VGGNet_H
#define VGGNet_H

#include <dlib/dnn.h>

namespace vggnet
{
    // clang-format off
    using namespace dlib;

    template <template <typename> class ACT, template <typename> class BN, template <typename> class DO>
    struct def
    {
        template <long num_filters, long ks, int s, int p, typename SUBNET>
        using conp = add_layer<con_<num_filters, ks, ks, s, s, p, p>, SUBNET>;

        // the main vgg building block
        template <long num_filters, typename SUBNET>
        using block = ACT<BN<conp<num_filters, 3, 1, 1, SUBNET>>>;

        // some definitions to allow the use of the repeat layer
        template <typename SUBNET> using block_512 = block<512, SUBNET>;
        template <typename SUBNET> using block_256 = block<256, SUBNET>;
        template <typename SUBNET> using block_128 = block<128, SUBNET>;
        template <typename SUBNET> using block_64 = block<64, SUBNET>;

        // the vgg backbones
        template <long nb_512, long nb_256, long nb_128, long nb_64, typename INPUT>
        using backbone = max_pool<2, 2, 2, 2, repeat<nb_512, block_512,
                         max_pool<2, 2, 2, 2, repeat<nb_512, block_512,
                         max_pool<2, 2, 2, 2, repeat<nb_256, block_256,
                         max_pool<2, 2, 2, 2, repeat<nb_128, block_128,
                         max_pool<2, 2, 2, 2, repeat<nb_64, block_64, tag1<INPUT>>>>>>>>>>>;

        // the final fully connected layers
        template <typename SUBNET>
        using final_fc = fc<1000, DO<relu<fc<4096, DO<relu<fc<4096, SUBNET>>>>>>>;

        template<typename INPUT> using backbone_11 = final_fc<backbone<2, 2, 1, 1, INPUT>>;
        template<typename INPUT> using backbone_13 = final_fc<backbone<2, 2, 2, 2, INPUT>>;
        template<typename INPUT> using backbone_16 = final_fc<backbone<3, 3, 2, 2, INPUT>>;
        template<typename INPUT> using backbone_19 = final_fc<backbone<4, 4, 2, 2, INPUT>>;
    };

    using train_11 = loss_multiclass_log<def<relu, bn_con, dropout>::backbone_11<input_rgb_image>>;
    using infer_11 = loss_multiclass_log<def<relu, affine, multiply>::backbone_11<input_rgb_image>>;
    using train_13 = loss_multiclass_log<def<relu, bn_con, dropout>::backbone_13<input_rgb_image>>;
    using infer_13 = loss_multiclass_log<def<relu, affine, multiply>::backbone_13<input_rgb_image>>;
    using train_16 = loss_multiclass_log<def<relu, bn_con, dropout>::backbone_16<input_rgb_image>>;
    using infer_16 = loss_multiclass_log<def<relu, affine, multiply>::backbone_16<input_rgb_image>>;
    using train_19 = loss_multiclass_log<def<relu, bn_con, dropout>::backbone_19<input_rgb_image>>;
    using infer_19 = loss_multiclass_log<def<relu, affine, multiply>::backbone_19<input_rgb_image>>;

    // clang-format on
}  // namespace vggnet

#endif  // VGGNet_H
