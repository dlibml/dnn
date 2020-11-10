#ifndef SqueezeNet_H
#define SqueezeNet_H

#include <dlib/dnn.h>

namespace squeezenet
{
    // clang-format off
    using namespace dlib;
    // ACT can be any activation
    template <template <typename> class ACT>
    struct def
    {
        template <long num_filters, long ks, int s, typename SUBNET>
        using conp = add_layer<con_<num_filters, ks, ks, s, s, ks/2, ks/2>, SUBNET>;

        template <typename SUBNET>
        using max_pool3 = add_layer<max_pool_<3, 3, 2, 2, 1, 1>, SUBNET>;

        template <typename INPUT>
        using stem = max_pool3<ACT<conp<64, 7, 2, INPUT>>>;

        template <long nf3x3, long nf1x1, long nf, typename SUBNET>
        using fire_module = concat2<tag2, tag3,
                            tag3<ACT<conp<nf3x3, 3, 1,
                            skip1<
                            tag2<ACT<conp<nf1x1, 1, 1,
                            tag1<ACT<conp<nf, 1, 1,
                            SUBNET>>>>>>>>>>>;

         template <typename INPUT>
         using backbone_1_0 = fire_module<256, 256, 64,
                              max_pool3<
                              fire_module<256, 256, 64,
                              fire_module<192, 192, 48,
                              fire_module<192, 192, 48,
                              fire_module<128, 128, 32,
                              max_pool3<
                              fire_module<128, 128, 32,
                              fire_module<64, 64, 16,
                              fire_module<64, 64, 16,
                              stem<INPUT>>>>>>>>>>>;

          template <typename INPUT>
          using backbone_1_1 = fire_module<256, 256, 64,
                               fire_module<256, 256, 64,
                               fire_module<192, 192, 48,
                               fire_module<192, 192, 48,
                               max_pool3<
                               fire_module<128, 128, 32,
                               fire_module<128, 128, 32,
                               max_pool3<
                               fire_module<64, 64, 16,
                               fire_module<64, 64, 16,
                               stem<INPUT>>>>>>>>>>>;
    };

    // DO must be dropout for train mode and multiply for infer
    template <template <typename> class ACT, template <typename> class DO, typename SUBNET>
    using classification_head = loss_multiclass_log<avg_pool_everything<ACT<con<1000, 1, 1, 1, 1, DO<SUBNET>>>>>;

    using train_v1_0 = classification_head<relu, dropout, def<relu>::backbone_1_0<input_rgb_image>>;
    using infer_v1_0 = classification_head<relu, multiply, def<relu>::backbone_1_0<input_rgb_image>>;
    using train_v1_1 = classification_head<relu, dropout, def<relu>::backbone_1_1<input_rgb_image>>;
    using infer_v1_1 = classification_head<relu, multiply, def<relu>::backbone_1_1<input_rgb_image>>;

    // clang-format on
}  // namespace squeezenet

#endif  // SqueezeNet_H
