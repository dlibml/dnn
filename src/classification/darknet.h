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
        template <long nf, long ks, int s, typename SUBNET>
        using conblock = ACT<BN<add_layer<con_<nf, ks, ks, s, s, ks/2, ks/2>, SUBNET>>>;

        template <long nf1, long nf2, typename SUBNET>
        using residual = add_prev1<
                         conblock<nf1, 3, 1,
                         conblock<nf2, 1, 1,
                         tag1<SUBNET>>>>;

        template <long nf, typename SUBNET> using resv3 = residual<nf, nf / 2, SUBNET>;
        template <long nf, typename SUBNET> using resv4 = residual<nf, nf, SUBNET>;

        template <long num_filters, typename SUBNET>
        using block3 = conblock<num_filters, 3, 1,
                       conblock<num_filters / 2, 1, 1,
                       conblock<num_filters, 3, 1,
                       SUBNET>>>;

        template <long num_filters, typename SUBNET>
        using block5 = conblock<num_filters, 3, 1,
                       conblock<num_filters / 2, 1, 1,
                       conblock<num_filters, 3, 1,
                       conblock<num_filters / 2, 1, 1,
                       conblock<num_filters, 3, 1,
                       SUBNET>>>>>;

        template <long nf, long factor, size_t N, template <typename> class RES, typename SUBNET>
        using cspblock = conblock<nf * factor, 1, 1,
                         concat2<tag1, tag2,
                         tag1<conblock<nf, 1, 1,
                         repeat<N, RES,
                         conblock<nf, 1, 1,
                         skip1<
                         tag2<conblock<nf, 1, 1,
                         tag1<conblock<nf * factor, 3, 2,
                         SUBNET>>>>>>>>>>>;

        template <typename SUBNET> using resv3_64= resv3<64, SUBNET>;
        template <typename SUBNET> using resv3_128 = resv3<128, SUBNET>;
        template <typename SUBNET> using resv3_256 = resv3<256, SUBNET>;
        template <typename SUBNET> using resv3_512 = resv3<512, SUBNET>;
        template <typename SUBNET> using resv3_1024 = resv3<1024, SUBNET>;
        template <typename SUBNET> using resv4_64= resv4<64, SUBNET>;
        template <typename SUBNET> using resv4_128 = resv4<128, SUBNET>;
        template <typename SUBNET> using resv4_256 = resv4<256, SUBNET>;
        template <typename SUBNET> using resv4_512 = resv4<512, SUBNET>;

        template <typename INPUT>
        using backbone19 = block5<1024,
                           max_pool<2, 2, 2, 2, block5<512,
                           max_pool<2, 2, 2, 2, block3<256,
                           max_pool<2, 2, 2, 2, block3<128,
                           max_pool<2, 2, 2, 2, conblock<64, 3, 1,
                           max_pool<2, 2, 2, 2, conblock<32, 3, 1,
                           INPUT>>>>>>>>>>>;

        template <typename INPUT>
        using backbone53 = repeat<4, resv3_1024, conblock<1024, 3, 2,
                           repeat<8, resv3_512, conblock<512, 3, 2,
                           repeat<8, resv3_256, conblock<256, 3, 2,
                           repeat<2, resv3_128, conblock<128, 3, 2,
                           resv3<64, conblock<64, 3, 2, conblock<32, 3, 1,
                           INPUT>>>>>>>>>>>;

        template <typename INPUT>
        using backbone53csp = cspblock<512, 2, 4, resv4_512,
                              cspblock<256, 2, 8, resv4_256,
                              cspblock<128, 2, 8, resv4_128,
                              cspblock<64, 2, 2, resv4_64,
                              cspblock<64, 1, 1, resv3_64,
                              conblock<32, 3, 1,
                              INPUT>>>>>>;
    };

    template <typename SUBNET>
    using classification_head = loss_multiclass_log<fc<1000, avg_pool_everything<SUBNET>>>;

    using train_19 = classification_head<def<leaky_relu, bn_con>::backbone19<input_rgb_image>>;
    using infer_19 = classification_head<def<leaky_relu, affine>::backbone19<input_rgb_image>>;
    using train_53 = classification_head<def<leaky_relu, bn_con>::backbone53<input_rgb_image>>;
    using infer_53 = classification_head<def<leaky_relu, affine>::backbone53<input_rgb_image>>;
    using train_53csp = classification_head<def<mish, bn_con>::backbone53csp<input_rgb_image>>;
    using infer_53csp = classification_head<def<mish, affine>::backbone53csp<input_rgb_image>>;
    // clang-format on

}  // namespace darknet

#endif  // DarkNet_H
