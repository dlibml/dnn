#ifndef RepVGG_H
#define RepVGG_H

#include <dlib/dnn.h>

namespace repvgg
{
    // clang-format off
    using namespace dlib;
    // ACT can be any activation layer, BN must be bn_con or affine layer and k is the growth rate
    template <template <typename> class ACT, long a_n, long a_d, long b_n, long b_d>
    struct def
    {
        // padded convolution
        template <long num_filters, long ks, int s, typename SUBNET>
        using pcon = add_layer<con_<num_filters, ks, ks, s, s, ks/2, ks/2>, SUBNET>;

        // bathnorm + padded convolution
        template <long num_filters, long ks, int s, typename SUBNET>
        using bcon = bn_con<pcon<num_filters, ks, s, SUBNET>>;

        // train block
        template <long num_filters, typename SUBNET>
        using tblock = ACT<add_prev2<bcon<num_filters, 1, 1, skip1<tag2<add_prev1<bcon<num_filters, 3, 1, tag1<SUBNET>>>>>>>>;

        // transition block for training
        template <long num_filters, typename SUBNET>
        using trans = ACT<add_prev2<bcon<num_filters, 1, 2, skip1<tag2<bcon<num_filters, 3, 2, tag1<SUBNET>>>>>>>;

        static const long filters_0 = std::min<long>(64, 64 * a_n / a_d);
        static const long filters_1 = 64 * a_n / a_d;
        static const long filters_2 = 128 * a_n / a_d;
        static const long filters_3 = 256 * a_n / a_d;
        static const long filters_4 = 512 * b_n / b_d;

        template <typename SUBNET>
        using tstem = trans<filters_0, SUBNET>;

        template <typename SUBNET>
        using istem = pcon<filters_0, 3, 2, SUBNET>;

        template <typename SUBNET> using tblock_1 = tblock<filters_1, SUBNET>;
        template <typename SUBNET> using tblock_2 = tblock<filters_2, SUBNET>;
        template <typename SUBNET> using tblock_3 = tblock<filters_3, SUBNET>;

        template <typename SUBNET> using iblock_1 = pcon<filters_1, 3, 1, SUBNET>;
        template <typename SUBNET> using iblock_2 = pcon<filters_2, 3, 1, SUBNET>;
        template <typename SUBNET> using iblock_3 = pcon<filters_3, 3, 1, SUBNET>;
        template <typename SUBNET> using iblock_4 = pcon<filters_4, 3, 1, SUBNET>;

        template <size_t nb_3, size_t nb_2, size_t nb_1, typename INPUT>
        using tbackbone = trans<filters_4,
                          repeat<nb_3 - 1, tblock_3, trans<filters_3,
                          repeat<nb_2 - 1, tblock_2, trans<filters_2,
                          repeat<nb_1 - 1, tblock_1, trans<filters_1,
                          tstem<INPUT>>>>>>>>;

        template <size_t nb_3, size_t nb_2, size_t nb_1, typename INPUT>
        using ibackbone = pcon<filters_4, 3, 2,
                          repeat<nb_3 - 1, iblock_3, pcon<filters_3, 3, 2,
                          repeat<nb_2 - 1, iblock_2, pcon<filters_2, 3, 2,
                          repeat<nb_1 - 1, iblock_1, pcon<filters_1, 3, 2,
                          istem<INPUT>>>>>>>>;
    };

    template <long num_filters, typename SUBNET>
    using classification_head = loss_multiclass_log<fc<num_filters, avg_pool_everything<SUBNET>>>;

    using train_a0 = classification_head<1000, def<relu, 3, 4, 5, 2>::tbackbone<14, 4, 2, input_rgb_image>>;
    using infer_a0 = classification_head<1000, def<relu, 3, 4, 5, 2>::ibackbone<14, 4, 2, input_rgb_image>>;
    using train_a1 = classification_head<1000, def<relu, 1, 1, 5, 2>::tbackbone<14, 4, 2, input_rgb_image>>;
    using infer_a1 = classification_head<1000, def<relu, 1, 1, 5, 2>::ibackbone<14, 4, 2, input_rgb_image>>;
    using train_a2 = classification_head<1000, def<relu, 3, 2, 11, 4>::tbackbone<14, 4, 2, input_rgb_image>>;
    using infer_a2 = classification_head<1000, def<relu, 3, 2, 11, 4>::ibackbone<14, 4, 2, input_rgb_image>>;
    using train_b0 = classification_head<1000, def<relu, 1, 1, 5, 2>::tbackbone<16, 6, 4, input_rgb_image>>;
    using infer_b0 = classification_head<1000, def<relu, 1, 1, 5, 2>::ibackbone<16, 6, 4, input_rgb_image>>;
    using train_b1 = classification_head<1000, def<relu, 2, 1, 4, 1>::tbackbone<16, 6, 4, input_rgb_image>>;
    using infer_b1 = classification_head<1000, def<relu, 2, 1, 4, 1>::ibackbone<16, 6, 4, input_rgb_image>>;
    using train_b2 = classification_head<1000, def<relu, 5, 2, 5, 1>::tbackbone<16, 6, 4, input_rgb_image>>;
    using infer_b2 = classification_head<1000, def<relu, 5, 2, 5, 1>::ibackbone<16, 6, 4, input_rgb_image>>;
    using train_b3 = classification_head<1000, def<relu, 5, 2, 5, 1>::tbackbone<16, 6, 4, input_rgb_image>>;
    using infer_b3 = classification_head<1000, def<relu, 3, 1, 5, 1>::ibackbone<16, 6, 4, input_rgb_image>>;
    // clang-format on
}  // namespace repvgg

#endif  // RepVGG_H
