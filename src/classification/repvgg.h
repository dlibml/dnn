#ifndef RepVGG_H
#define RepVGG_H

#include <dlib/dnn.h>

namespace repvgg
{
    // clang-format off
    using namespace dlib;
    // ACT can be any activation layer, BN must be bn_con or affine
    // a_n, a_d: a parameter numerator and denominator, respectively
    // b_n, b_d: b parameter numerator and denominator, respectively
    template <template <typename> class ACT, long a_n, long a_d, long b_n, long b_d>
    struct def
    {
        static const long filters_0 = std::min<long>(64, 64 * a_n / a_d);
        static const long filters_1 = 64 * a_n / a_d;
        static const long filters_2 = 128 * a_n / a_d;
        static const long filters_3 = 256 * a_n / a_d;
        static const long filters_4 = 512 * b_n / b_d;

        // padded convolution
        template <long num_filters, long ks, int s, typename SUBNET>
        using pcon = add_layer<con_<num_filters, ks, ks, s, s, ks/2, ks/2>, SUBNET>;

        // batch norm + padded convolution
        template <long num_filters, long ks, int s, typename SUBNET>
        using bcon = bn_con<pcon<num_filters, ks, s, SUBNET>>;

        // RepVGG block: 3x3 & 1x1 convolutions
        template <long num_filters, int s, typename SUBNET>
        using repvggblock = add_prev1<bcon<num_filters, 1, s, skip2<tag1<bcon<num_filters, 3, s, tag2<SUBNET>>>>>>;

        // RepVGG block + identity (with batch norm): tag5 is the input of the RepVGG block defined above
        template <long num_filters, typename SUBNET>
        using repvggblock_id = add_prev1<bn_con<skip2<repvggblock<num_filters, 1, tag1<SUBNET>>>>>;

        template <typename SUBNET> using repvggblock_id_1 = ACT<repvggblock_id<filters_1, SUBNET>>;
        template <typename SUBNET> using repvggblock_id_2 = ACT<repvggblock_id<filters_2, SUBNET>>;
        template <typename SUBNET> using repvggblock_id_3 = ACT<repvggblock_id<filters_3, SUBNET>>;

        template <typename SUBNET> using iblock_1 = ACT<pcon<filters_1, 3, 1, SUBNET>>;
        template <typename SUBNET> using iblock_2 = ACT<pcon<filters_2, 3, 1, SUBNET>>;
        template <typename SUBNET> using iblock_3 = ACT<pcon<filters_3, 3, 1, SUBNET>>;

        template <size_t nb_3, size_t nb_2, size_t nb_1, typename INPUT>
        using tbackbone = ACT<repvggblock<filters_4, 2,
                          repeat<nb_3, repvggblock_id_3, ACT<repvggblock<filters_3, 2,
                          repeat<nb_2, repvggblock_id_2, ACT<repvggblock<filters_2, 2,
                          repeat<nb_1, repvggblock_id_1, ACT<repvggblock<filters_1, 2,
                          ACT<repvggblock<filters_0, 2, INPUT>>>>>>>>>>>>>;

        template <size_t nb_3, size_t nb_2, size_t nb_1, typename INPUT>
        using ibackbone = ACT<pcon<filters_4, 3, 2,
                          repeat<nb_3, iblock_3, ACT<pcon<filters_3, 3, 2,
                          repeat<nb_2, iblock_2, ACT<pcon<filters_2, 3, 2,
                          repeat<nb_1, iblock_1, ACT<pcon<filters_1, 3, 2,
                          ACT<pcon<filters_0, 3, 2, INPUT>>>>>>>>>>>>>;

    };

    template <long num_filters, typename SUBNET>
    using classification_head = loss_multiclass_log<fc<num_filters, avg_pool_everything<SUBNET>>>;

    using train_a0 = classification_head<1000, def<relu, 3, 4, 5, 2>::tbackbone<13, 3, 1, input_rgb_image>>;
    using infer_a0 = classification_head<1000, def<relu, 3, 4, 5, 2>::ibackbone<13, 3, 1, input_rgb_image>>;
    using train_a1 = classification_head<1000, def<relu, 1, 1, 5, 2>::tbackbone<13, 3, 1, input_rgb_image>>;
    using infer_a1 = classification_head<1000, def<relu, 1, 1, 5, 2>::ibackbone<13, 3, 1, input_rgb_image>>;
    using train_a2 = classification_head<1000, def<relu, 3, 2, 11, 4>::tbackbone<13, 3, 1, input_rgb_image>>;
    using infer_a2 = classification_head<1000, def<relu, 3, 2, 11, 4>::ibackbone<13, 3, 1, input_rgb_image>>;
    using train_b0 = classification_head<1000, def<relu, 1, 1, 5, 2>::tbackbone<15, 5, 3, input_rgb_image>>;
    using infer_b0 = classification_head<1000, def<relu, 1, 1, 5, 2>::ibackbone<15, 5, 3, input_rgb_image>>;
    using train_b1 = classification_head<1000, def<relu, 2, 1, 4, 1>::tbackbone<15, 5, 3, input_rgb_image>>;
    using infer_b1 = classification_head<1000, def<relu, 2, 1, 4, 1>::ibackbone<15, 5, 3, input_rgb_image>>;
    using train_b2 = classification_head<1000, def<relu, 5, 2, 5, 1>::tbackbone<15, 5, 3, input_rgb_image>>;
    using infer_b2 = classification_head<1000, def<relu, 5, 2, 5, 1>::ibackbone<15, 5, 3, input_rgb_image>>;
    using train_b3 = classification_head<1000, def<relu, 5, 2, 5, 1>::tbackbone<15, 5, 3, input_rgb_image>>;
    using infer_b3 = classification_head<1000, def<relu, 3, 1, 5, 1>::ibackbone<15, 5, 3, input_rgb_image>>;
    // clang-format on
}  // namespace repvgg

#endif  // RepVGG_H
