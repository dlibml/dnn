#ifndef DenseNet_H
#define DenseNet_H

#include <dlib/dnn.h>

namespace densenet
{
    // clang-format off
    using namespace dlib;
    // ACT can be any activation layer and BN must be bn_con or affine layer
    template <template <typename> class ACT, template <typename> class BN>
    struct def
    {
        template <long num_filters, long ks, int s, typename SUBNET>
        using conp = add_layer<con_<num_filters, ks, ks, s, s, ks/2, ks/2>, SUBNET>;

        template <typename INPUT>
        using stem = add_layer<max_pool_<3, 3, 2, 2, 1, 1>, ACT<BN<conp<64, 7, 2, INPUT>>>>;

        template <long num_filters, typename SUBNET>
        using transition = avg_pool<2, 2, 2, 2, con<num_filters, 1, 1, 1, 1, ACT<BN<SUBNET>>>>;

        template <long num_filters, long growth_rate, typename SUBNET>
        using dense_layer = concat2<tag1, tag2,
                            tag2<conp<growth_rate, 3, 1,
                            ACT<BN<conp<4 * growth_rate, 1, 1,
                            ACT<BN<tag1<SUBNET>>>>>>>>>;

        template <typename SUBNET> using dense_layer_128 = dense_layer<128, 32, SUBNET>;
        template <typename SUBNET> using dense_layer_256 = dense_layer<256, 32, SUBNET>;
        template <typename SUBNET> using dense_layer_512 = dense_layer<512, 32, SUBNET>;
        template <typename SUBNET> using dense_layer_1024 = dense_layer<1024, 32, SUBNET>;

        template <size_t nb_1024, size_t nb_512, size_t nb_256, size_t nb_128, typename INPUT>
        using backbone = ACT<BN<
                         repeat<nb_1024, dense_layer_1024, transition<512,
                         repeat<nb_512, dense_layer_512, transition<256,
                         repeat<nb_256, dense_layer_256, transition<128,
                         repeat<nb_128, dense_layer_128, stem<INPUT>>>>>>>>>>;

         template <typename INPUT> using backbone_121 = backbone<16, 24, 12, 6, INPUT>;
         template <typename INPUT> using backbone_169 = backbone<32, 32, 12, 6, INPUT>;
         template <typename INPUT> using backbone_201 = backbone<32, 48, 12, 6, INPUT>;
         template <typename INPUT> using backbone_264 = backbone<48, 64, 12, 6, INPUT>;
    };

    template <typename SUBNET>
    using classification_head = loss_multiclass_log<fc<1000, avg_pool_everything<SUBNET>>>;

    using train_121 = classification_head<def<relu, bn_con>::backbone_121<input_rgb_image>>;
    using infer_121 = classification_head<def<relu, affine>::backbone_121<input_rgb_image>>;
    using train_169 = classification_head<def<relu, bn_con>::backbone_169<input_rgb_image>>;
    using infer_169 = classification_head<def<relu, affine>::backbone_169<input_rgb_image>>;
    using train_201 = classification_head<def<relu, bn_con>::backbone_201<input_rgb_image>>;
    using infer_201 = classification_head<def<relu, affine>::backbone_201<input_rgb_image>>;
    using train_264 = classification_head<def<relu, bn_con>::backbone_264<input_rgb_image>>;
    using infer_264 = classification_head<def<relu, affine>::backbone_264<input_rgb_image>>;

    // clang-format on
}  // namespace densenet

#endif  // DenseNet_H
