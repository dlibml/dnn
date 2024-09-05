#ifndef ResNet_H
#define ResNet_H

#include <dlib/dnn.h>

namespace resnet
{
    // clang-format off
    using namespace dlib;
    template <template <typename> class BN = bn_con, template <typename> class ACT = relu, long k = 64>
    struct def
    {
        template <long N, int K, int S, typename SUBNET>
        using conv = add_layer<con_<N, K, K, S, S, K / 2, K / 2>, SUBNET>;

        template <typename INPUT>
        using stem = add_layer<max_pool_<3, 3, 2, 2, 1, 1>, ACT<BN<conv<k, 7, 2, INPUT>>>>;

        template <long N, int S, typename SUBNET>
        using basicblock = BN<conv<N, 3, 1, ACT<BN<conv<N, 3, S, SUBNET>>>>>;

        template<long N, int S, typename SUBNET>
        using bottleneck = BN<conv<4 * N, 1, 1, ACT<BN<conv<N, 3, S, ACT<BN<conv<N, 1, 1, SUBNET>>>>>>>>;

        template <template <long, int, typename> class BLOCK, long N, typename SUBNET>
        using residual = ACT<add_prev1<BLOCK<N, 1, tag1<SUBNET>>>>;

        template <template <long, int, typename> class BLOCK, long N, long F, long S, typename SUBNET>
        using transition = ACT<add_prev2<BN<conv<N * F, 1, S, skip1<tag2<BLOCK<N, S, tag1<SUBNET>>>>>>>>;

        template <typename SUBNET> using resbasicblock_8k = residual<basicblock, 8 * k, SUBNET>;
        template <typename SUBNET> using resbasicblock_4k = residual<basicblock, 4 * k, SUBNET>;
        template <typename SUBNET> using resbasicblock_2k = residual<basicblock, 2 * k, SUBNET>;
        template <typename SUBNET> using resbasicblock_1k = residual<basicblock, 1 * k, SUBNET>;
        template <typename SUBNET> using resbottleneck_8k = residual<bottleneck, 8 * k, SUBNET>;
        template <typename SUBNET> using resbottleneck_4k = residual<bottleneck, 4 * k, SUBNET>;
        template <typename SUBNET> using resbottleneck_2k = residual<bottleneck, 2 * k, SUBNET>;
        template <typename SUBNET> using resbottleneck_1k = residual<bottleneck, 1 * k, SUBNET>;

        template <long N8k, long N4k, long N2k, long N1k, typename INPUT>
        using backbone_basicblock = repeat<N8k, resbasicblock_8k, transition<basicblock, 8 * k, 1, 2,
                                    repeat<N4k, resbasicblock_4k, transition<basicblock, 4 * k, 1, 2,
                                    repeat<N2k, resbasicblock_2k, transition<basicblock, 2 * k, 1, 2,
                                    repeat<N1k, resbasicblock_1k, transition<basicblock, 1 * k, 1, 1,
                                    stem<INPUT>>>>>>>>>;

        template <long N8k, long N4k, long N2k, long N1k, typename INPUT>
        using backbone_bottleneck = repeat<N8k, resbottleneck_8k, transition<bottleneck, 8 * k, 4, 2,
                                    repeat<N4k, resbottleneck_4k, transition<bottleneck, 4 * k, 4, 2,
                                    repeat<N2k, resbottleneck_2k, transition<bottleneck, 2 * k, 4, 2,
                                    repeat<N1k, resbottleneck_1k, transition<bottleneck, 1 * k, 4, 1,
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
