#ifndef VoVNet_H
#define VoVNet_H

#include <dlib/dnn.h>

namespace vovnet
{
    // clang-format off
    using namespace dlib;

    // vov_tag0 is used for identity mapping
    template <typename SUBNET> using vov_tag0 = add_tag_layer<5050, SUBNET>;
    template <typename SUBNET> using vov_tag1 = add_tag_layer<5051, SUBNET>;
    template <typename SUBNET> using vov_tag2 = add_tag_layer<5052, SUBNET>;
    template <typename SUBNET> using vov_tag3 = add_tag_layer<5053, SUBNET>;
    template <typename SUBNET> using vov_tag4 = add_tag_layer<5054, SUBNET>;
    template <typename SUBNET> using vov_tag5 = add_tag_layer<5055, SUBNET>;
    template <typename SUBNET> using id_mapping = add_prev<vov_tag0, SUBNET>;

    template <template <typename> class ACT, template <typename> class BN>
    struct def
    {

        // The concatenate layer with custom number of outputs for OSA Module with 3 layers
        template <long num_filters, typename SUBNET>
        using concatenate3 = ACT<BN<con<num_filters, 1, 1, 1, 1,
        add_layer<concat_<vov_tag0, vov_tag1, vov_tag2, vov_tag3>, SUBNET>>>>;

        // The concatenate layer with custom number of outputs for OSA Module with 5 layers
        template <long num_filters, typename SUBNET>
        using concatenate5 = ACT<BN<con<num_filters, 1, 1, 1, 1,
        add_layer<concat_<vov_tag0, vov_tag1, vov_tag2, vov_tag3, vov_tag4, vov_tag5>, SUBNET>>>>;

        // 1-padded 3x3 convolution with custom number of filters, kernel size and stride
        template <long num_filters, int s, typename SUBNET>
        using con3 = ACT<BN<add_layer<con_<num_filters, 3, 3, s, s, 1, 1>, SUBNET>>>;

        // Max-pooling with 3x3 kernel size 2-stride and 1-padding
        template <typename SUBNET> using maxpool = add_layer<max_pool_<3, 3, 2, 2, 1, 1>, SUBNET>;

        // Stem block
        template <typename INPUT>
        using stem = con3<128, 2, con3<64, 1, con3<64, 2, INPUT>>>;

        // The VoVNet effective Squeeze and Excitation Module
        template <long num_filters, typename SUBNET>
        using ese_module = scale_prev2<skip1<
                           tag2<sig<con<num_filters, 1, 1, 1, 1,
                           avg_pool_everything<
                           tag1<SUBNET>>>>>>>;

        // The VoVNet One-Shot Aggregation Module with 3 inner layers
        template <long num_filters_out, long num_filters_in, typename SUBNET>
        using osa_module3 = ese_module<num_filters_out,
                            concatenate3<num_filters_out,
                            vov_tag3<con3<num_filters_in, 1,
                            vov_tag2<con3<num_filters_in, 1,
                            vov_tag1<con3<num_filters_in, 1,
                            vov_tag0<SUBNET>>>>>>>>>;

        // The VoVNet One-Shot Aggregation Module with 5 inner layers
        template <long num_filters_out, long num_filters_in, typename SUBNET>
        using osa_module5 = ese_module<num_filters_out,
                            concatenate5<num_filters_out,
                            vov_tag5<con3<num_filters_in, 1,
                            vov_tag4<con3<num_filters_in, 1,
                            vov_tag3<con3<num_filters_in, 1,
                            vov_tag2<con3<num_filters_in, 1,
                            vov_tag1<con3<num_filters_in, 1,
                            vov_tag0<SUBNET>>>>>>>>>>>>>;

        // some definitions to allow the use of the repeat layer
        template <typename SUBNET> using osa_module5_id_512 = id_mapping<osa_module5<512, 160, SUBNET>>;
        template <typename SUBNET> using osa_module5_id_768 = id_mapping<osa_module5<768, 192, SUBNET>>;
        template <typename SUBNET> using osa_module5_id_1024 = id_mapping<osa_module5<1024, 224, SUBNET>>;

        template <typename INPUT>
        using backbone_19_slim = osa_module3<512, 112,
                                 maxpool<osa_module3<384, 96,
                                 maxpool<osa_module3<256, 80,
                                 maxpool<osa_module3<112, 64,
                                 stem<INPUT>>>>>>>>;

        template <typename INPUT>
        using backbone_19 = osa_module3<1024, 224,
                            maxpool<osa_module3<768, 192,
                            maxpool<osa_module3<512, 160,
                            maxpool<osa_module3<256, 128,
                            stem<INPUT>>>>>>>>;

        template <typename INPUT>
        using backbone_39 = osa_module5_id_1024<osa_module5<1024, 224,
                            maxpool<osa_module5_id_768<osa_module5<768, 192,
                            maxpool<osa_module5<512, 160,
                            maxpool<osa_module5<256, 128,
                            stem<INPUT>>>>>>>>>>;

        template <typename INPUT>
        using backbone_57 = repeat<2, osa_module5_id_1024, osa_module5<1024, 224,
                            maxpool<repeat<3, osa_module5_id_768, osa_module5<768, 192,
                            maxpool<osa_module5<512, 160,
                            maxpool<osa_module5<256, 128,
                            stem<INPUT>>>>>>>>>>;

        template <typename INPUT>
        using backbone_99 = repeat<2, osa_module5_id_1024, osa_module5<1014, 224,
                            maxpool<repeat<8, osa_module5_id_768, osa_module5<768, 192,
                            maxpool<repeat<2, osa_module5_id_512, osa_module5<512, 160,
                            maxpool<osa_module5<256, 128,
                            stem<INPUT>>>>>>>>>>>;
    };

    template <long num_filters, typename SUBNET> using classification_head =
    loss_multiclass_log<fc<num_filters, avg_pool_everything<SUBNET>>>;

    using train_19_slim = classification_head<1000, def<relu, bn_con>::backbone_19_slim<input_rgb_image>>;
    using train_19 = classification_head<1000, def<relu, bn_con>::backbone_19<input_rgb_image>>;
    using train_39 = classification_head<1000, def<relu, bn_con>::backbone_39<input_rgb_image>>;
    using train_57 = classification_head<1000, def<relu, bn_con>::backbone_57<input_rgb_image>>;
    using train_99 = classification_head<1000, def<relu, bn_con>::backbone_99<input_rgb_image>>;
    using infer_19_slim = classification_head<1000, def<relu, affine>::backbone_19_slim<input_rgb_image>>;
    using infer_19 = classification_head<1000, def<relu, affine>::backbone_19<input_rgb_image>>;
    using infer_39 = classification_head<1000, def<relu, affine>::backbone_39<input_rgb_image>>;
    using infer_57 = classification_head<1000, def<relu, affine>::backbone_57<input_rgb_image>>;
    using infer_99 = classification_head<1000, def<relu, affine>::backbone_99<input_rgb_image>>;
    // clang-format on
}  // namespace vovnet
#endif  // VoVNet_H
