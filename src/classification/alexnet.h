#ifndef AlexNet_H
#define AlexNet_H

#include <dlib/dnn.h>

namespace alexnet
{
    // clang-format off
    using namespace dlib;
    template <template <typename> class ACT, template <typename> class DO>
    struct def
    {
        template <long num_filters, long ks, int s, int p, typename SUBNET>
        using conp = add_layer<con_<num_filters, ks, ks, s, s, p, p>, SUBNET>;

        template <typename INPUT>
        using backbone = fc<1000, ACT<fc<4096, DO<ACT<fc<4096, DO<
                         max_pool<3, 3, 2, 2, ACT<conp<256, 3, 1, 1,
                         ACT<conp<384, 3, 1, 1, ACT<conp<384, 3, 1, 1,
                         max_pool<3, 3, 2, 2, ACT<conp<256, 5, 1, 2,
                         max_pool<3, 3, 2, 2, ACT<conp<96, 11, 4, 2,
                         INPUT>>>>>>>>>>>>>>>>>>>>;
    };

    using train = loss_multiclass_log<def<relu, dropout>::backbone<input_rgb_image>>;
    using infer = loss_multiclass_log<def<relu, multiply>::backbone<input_rgb_image>>;
    // clang-format on
}  // namespace alexnet

#endif  // AlexNet_H
