#ifndef ResNet_H
#define ResNet_H

#include <dlib/dnn.h>

namespace resnet
{
    // clang-format off
    using namespace dlib;
    template <template <typename> class BN = bn_con, template <typename> class ACT = relu>
    struct def
    {
        template <long N, int K, int S, typename SUBNET>
        using conv = add_layer<con_<N, K, K, S, S, K / 2, K / 2>, SUBNET>;

        template <typename INPUT>
        using stem = add_layer<max_pool_<3, 3, 2, 2, 1, 1>, ACT<BN<conv<64, 7, 2, INPUT>>>>;

        template <long N, int S, typename SUBNET>
        using basicblock = BN<conv<N, 3, 1, ACT<BN<conv<N, 3, S, SUBNET>>>>>;

        template<long N, int S, typename SUBNET>
        using bottleneck = BN<conv<4 * N, 1, 1, ACT<BN<conv<N, 3, S, ACT<BN<conv<N, 1, 1, SUBNET>>>>>>>>;

        template <template <long, int, typename> class BLOCK, long N, typename SUBNET>
        using residual = ACT<add_prev1<BLOCK<N, 1, tag1<SUBNET>>>>;

        template <template <long, int, typename> class BLOCK, long N, long F, long S, typename SUBNET>
        using transition = ACT<add_prev2<BN<conv<N * F, 1, S, skip1<tag2<BLOCK<N, S, tag1<SUBNET>>>>>>>>;

        template <typename SUBNET> using resbasicblock_512 = residual<basicblock, 512, SUBNET>;
        template <typename SUBNET> using resbasicblock_256 = residual<basicblock, 256, SUBNET>;
        template <typename SUBNET> using resbasicblock_128 = residual<basicblock, 128, SUBNET>;
        template <typename SUBNET> using resbasicblock_64  = residual<basicblock, 64, SUBNET>;
        template <typename SUBNET> using resbottleneck_512 = residual<bottleneck, 512, SUBNET>;
        template <typename SUBNET> using resbottleneck_256 = residual<bottleneck, 256, SUBNET>;
        template <typename SUBNET> using resbottleneck_128 = residual<bottleneck, 128, SUBNET>;
        template <typename SUBNET> using resbottleneck_64  = residual<bottleneck, 64, SUBNET>;

        template <long N512, long N256, long N128, long N64, typename INPUT>
        using backbone_basicblock = repeat<N512, resbasicblock_512, transition<basicblock, 512, 1, 2,
                                    repeat<N256, resbasicblock_256, transition<basicblock, 256, 1, 2,
                                    repeat<N128, resbasicblock_128, transition<basicblock, 128, 1, 2,
                                    repeat<N64, resbasicblock_64, transition<basicblock, 64, 1, 1,
                                    stem<INPUT>>>>>>>>>;

        template <long N512, long N256, long N128, long N64, typename INPUT>
        using backbone_bottleneck = repeat<N512, resbottleneck_512, transition<bottleneck, 512, 4, 2,
                                    repeat<N256, resbottleneck_256, transition<bottleneck, 256, 4, 2,
                                    repeat<N128, resbottleneck_128, transition<bottleneck, 128, 4, 2,
                                    repeat<N64, resbottleneck_64, transition<bottleneck, 64, 4, 1,
                                    stem<INPUT>>>>>>>>>;

        // the backbones for the classic architectures
        template <typename INPUT> using backbone_18  = backbone_basicblock<1, 1, 1, 1, INPUT>;
        template <typename INPUT> using backbone_34  = backbone_basicblock<2, 5, 3, 2, INPUT>;
        template <typename INPUT> using backbone_50  = backbone_bottleneck<2, 5, 3, 2, INPUT>;
        template <typename INPUT> using backbone_101 = backbone_bottleneck<2, 22, 3, 2, INPUT>;
        template <typename INPUT> using backbone_152 = backbone_bottleneck<2, 35, 7, 2, INPUT>;
    };
    // clang-format on

    template <typename SUBNET>
    using classification_head = loss_multiclass_log<fc<1000, avg_pool_everything<SUBNET>>>;

    using train_18  = classification_head<def<bn_con, relu>::backbone_18<input_rgb_image>>;
    using infer_18  = classification_head<def<affine, relu>::backbone_18<input_rgb_image>>;
    using train_34  = classification_head<def<bn_con, relu>::backbone_34<input_rgb_image>>;
    using infer_34  = classification_head<def<affine, relu>::backbone_34<input_rgb_image>>;
    using train_50  = classification_head<def<bn_con, relu>::backbone_50<input_rgb_image>>;
    using infer_50  = classification_head<def<affine, relu>::backbone_50<input_rgb_image>>;
    using train_101 = classification_head<def<bn_con, relu>::backbone_101<input_rgb_image>>;
    using infer_101 = classification_head<def<affine, relu>::backbone_101<input_rgb_image>>;
    using train_152 = classification_head<def<bn_con, relu>::backbone_152<input_rgb_image>>;
    using infer_152 = classification_head<def<affine, relu>::backbone_152<input_rgb_image>>;
};  // namespace resnet

#endif  // ResNet_H
