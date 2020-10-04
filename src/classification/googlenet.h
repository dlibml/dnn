#ifndef GoogLeNet_H
#define GoogLeNet_H

#include <dlib/dnn.h>

namespace googlenet
{
    // clang-format off
    using namespace dlib;

    template <template <typename> class ACT, template <typename> class BN, template <typename> class DO>
    struct def
    {
        template <long num_filters, long ks, int s, int p, typename SUBNET>
        using con_block = ACT<BN<add_layer<con_<num_filters, ks, ks, s, s, p, p>, SUBNET>>>;

        template <long ks, int s, int p, typename SUBNET>
        using maxpool = add_layer<max_pool_<ks, ks, s, s, p, p>, SUBNET>;

        template <typename INPUT>
        using stem = maxpool<3, 2, 1, con_block<192, 3, 1, 1,
                     con_block<64, 1, 1, 0,
                     maxpool<3, 2, 1, con_block<64, 7, 2, 3,
                     INPUT>>>>>;

        template <long nf_1, long nf_3o, long nf_3i, long nf_5o, long nf_5i, long nf_pool, typename SUBNET>
        using inception_block = concat4<itag1, itag2, itag3, itag4,
                                itag1<con_block<nf_1, 1, 1, 0, iskip<
                                itag2<con_block<nf_3o, 3, 1, 1, con_block<nf_3i, 1, 1, 0, iskip<
                                itag3<con_block<nf_5o, 3, 1, 1, con_block<nf_5i, 1, 1, 0, iskip<
                                itag4<con_block<nf_pool, 3, 1, 1, maxpool<3, 1, 1, itag0<
                                SUBNET>>>>>>>>>>>>>>>>;

        template <typename INPUT>
        using backbone = inception_block<384, 384, 192, 128, 48, 128,
                         inception_block<256, 320, 160, 128, 32, 128,
                         maxpool<3, 2, 1,
                         inception_block<256, 320, 160, 128, 32, 128,
                         inception_block<112, 288, 144, 64, 32, 64,
                         inception_block<128, 256, 128, 64, 24, 64,
                         inception_block<160, 224, 112, 64, 24, 64,
                         inception_block<192, 208, 96, 48, 16, 64,
                         maxpool<3, 2, 1,
                         inception_block<128, 192, 128, 96, 32, 64,
                         inception_block<64, 128, 96, 32, 16, 32,
                         stem<INPUT>>>>>>>>>>>>;

         using net_type = loss_multiclass_log<fc<1000, DO<avg_pool_everything<backbone<input_rgb_image>>>>>;
    };
    using train = def<relu, bn_con, dropout>::net_type;
    using infer = def<relu, affine, multiply>::net_type;
    // clang-format on
}  // namespace googlenet

#endif  // GoogLeNet_H
