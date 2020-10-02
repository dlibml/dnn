#ifndef DarkNet_H
#define DarkNet_H

#include <dlib/dnn.h>

namespace darknet
{
    // clang-format off
    using namespace dlib;

    template <template <typename> class ACT, template <typename> class BN>
    struct def
    {
        template <long nf, long ks, int s, int p, typename SUBNET>
        using convolutional = ACT<BN<add_layer<con_<nf, ks, ks, s, s, p, p>, SUBNET>>>;

        template <long num_filters, typename SUBNET>
        using residual = add_prev1<
                         convolutional<num_filters, 3, 1, 1,
                         convolutional<num_filters / 2, 1, 1, 0,
                         tag1<SUBNET>>>>;

        template <long num_filters, typename SUBNET>
        using block3 = convolutional<num_filters, 3, 1, 1,
                       convolutional<num_filters / 2, 1, 1, 0,
                       convolutional<num_filters, 3, 1, 1,
                       SUBNET>>>;

        template <long num_filters, typename SUBNET>
        using block5 = convolutional<num_filters, 3, 1, 1,
                       convolutional<num_filters / 2, 1, 1, 0,
                       convolutional<num_filters, 3, 1, 1,
                       convolutional<num_filters / 2, 1, 1, 0,
                       convolutional<num_filters, 3, 1, 1,
                       SUBNET>>>>>;

        template <typename SUBNET> using residual_128 = residual<128, SUBNET>;
        template <typename SUBNET> using residual_256 = residual<256, SUBNET>;
        template <typename SUBNET> using residual_512 = residual<512, SUBNET>;
        template <typename SUBNET> using residual_1024 = residual<1024, SUBNET>;

        template <typename INPUT>
        using backbone19 = block5<1024,
                           max_pool<2, 2, 2, 2, block5<512,
                           max_pool<2, 2, 2, 2, block3<256,
                           max_pool<2, 2, 2, 2, block3<128,
                           max_pool<2, 2, 2, 2, convolutional<64, 3, 1, 1,
                           max_pool<2, 2, 2, 2, convolutional<32, 3, 1, 1,
                           INPUT>>>>>>>>>>>;

        template <typename INPUT>
        using backbone53 = repeat<4, residual_1024, convolutional<1024, 3, 2, 1,
                           repeat<8, residual_512, convolutional<512, 3, 2, 1,
                           repeat<8, residual_256, convolutional<256, 3, 2, 1,
                           repeat<2, residual_128, convolutional<128, 3, 2, 1,
                           residual<64, convolutional<64, 3, 2, 1, convolutional<32, 3, 1, 1,
                           INPUT>>>>>>>>>>>;
    };

    template <typename SUBNET>
    using classification_head = loss_multiclass_log<fc<1000, avg_pool_everything<SUBNET>>>;

    using train_19 = classification_head<def<leaky_relu, bn_con>::backbone19<input_rgb_image>>;
    using infer_19 = classification_head<def<leaky_relu, affine>::backbone19<input_rgb_image>>;

    using train_53 = classification_head<def<leaky_relu, bn_con>::backbone53<input_rgb_image>>;
    using infer_53 = classification_head<def<leaky_relu, affine>::backbone53<input_rgb_image>>;
    // clang-format on

}  // namespace darknet

#endif  // DarkNet_H
