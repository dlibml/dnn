#ifndef yolov5_h_INCLUDED
#define yolov5_h_INCLUDED

#include <dlib/dnn.h>

namespace yolov5
{
    using namespace dlib;
    template <typename SUBNET> using ytag3 = add_tag_layer<4003, SUBNET>;
    template <typename SUBNET> using ytag4 = add_tag_layer<4004, SUBNET>;
    template <typename SUBNET> using ytag5 = add_tag_layer<4005, SUBNET>;
    template <typename SUBNET> using ptag3 = add_tag_layer<7003, SUBNET>;
    template <typename SUBNET> using ptag4 = add_tag_layer<7004, SUBNET>;
    template <typename SUBNET> using ptag5 = add_tag_layer<7005, SUBNET>;

    template <
        template <typename> class ACT,
        template <typename> class BN,
        long depth_num = 1,
        long depth_den = 1,
        long width_num = 1,
        long width_den = 1
    >
    struct def
    {
        const static long nf = 64 * width_num / width_den;

        template <long NF, int KS, int S, typename SUBNET>
        using conv = ACT<BN<add_layer<con_<NF, KS, KS, S, S, (KS-1)/2, (KS-1)/2>, SUBNET>>>;

        template <long NF, typename SUBNET>
        using bottleneck = conv<NF, 3, 1, conv<NF, 1, 1, SUBNET>>;

        template <long NF, typename SUBNET>
        using resbottleneck = add_prev10<bottleneck<NF, tag10<SUBNET>>>;

        template <long NF, typename SUBNET>
        using sppf = conv<NF, 1, 1,
                     concat4<tag1, tag2, tag3, tag4,
                tag4<max_pool<5, 5, 1, 1,
                tag3<max_pool<5, 5, 1, 1,
                tag2<max_pool<5, 5, 1, 1,
                tag1<conv<NF/2, 1, 1, SUBNET>>>>>>>>>>;

        template <typename SUBNET> using bottleneck_x2 = bottleneck<2 * nf, SUBNET>;
        template <typename SUBNET> using bottleneck_x4 = bottleneck<4 * nf, SUBNET>;
        template <typename SUBNET> using bottleneck_x8 = bottleneck<8 * nf, SUBNET>;
        template <typename SUBNET> using resbottleneck_x1 = resbottleneck<nf, SUBNET>;
        template <typename SUBNET> using resbottleneck_x2 = resbottleneck<2 * nf, SUBNET>;
        template <typename SUBNET> using resbottleneck_x4 = resbottleneck<4 * nf, SUBNET>;
        template <typename SUBNET> using resbottleneck_x8 = resbottleneck<8 * nf, SUBNET>;

        // CSP Bottleneck with 3 convolutions
        template <long NF, size_t N, template <typename> class BLOCK, typename SUBNET>
        using c3 = conv<NF, 1, 1,
                   concat2<tag8, tag9,
              tag9<conv<NF/2, 1, 1, skip7<
              tag8<repeat<N * depth_num / depth_den, BLOCK, conv<NF/2, 1, 1,
              tag7<SUBNET>>>>>>>>>;

        template <typename INPUT>
        using backbone = sppf<16 * nf,
                   ptag5<c3<16 * nf, 3, resbottleneck_x8,
                         conv<16 * nf, 3, 2,
                   ptag4<c3<8 * nf, 9, resbottleneck_x4,
                         conv<8 * nf, 3, 2,
                   ptag3<c3<4 * nf, 6, resbottleneck_x2,
                         conv<4 * nf, 3, 2,
                         c3<2 * nf, 3, resbottleneck_x1,
                         conv<2 * nf, 3, 2,
                         conv<nf, 6, 2,
                         INPUT>>>>>>>>>>>>>;

        template <template <typename> class YTAG, typename SUBNET>
        using yolo = YTAG<sig<con<1, 1, 1, 1, 1, SUBNET>>>;

        template <typename SUBNET>
        using head = yolo<ytag5,
                     c3<16 * nf, 3, bottleneck_x8,
                     concat2<tag1, tag5,
                tag1<conv<8 * nf, 3, 2, skip2<
                     yolo<ytag4,
                tag2<c3<8 * nf, 3, bottleneck_x4,
                     concat2<tag1, tag4,
                tag1<conv<4 * nf, 3, 2, skip2<
                     yolo<ytag3,
                tag2<c3<4 * nf, 3, bottleneck_x2,
                     concat2<tag1, ptag3,
                tag1<upsample<2,
                tag4<conv<4 * nf, 1, 1,
                     c3<8 * nf, 3, bottleneck_x4,
                     concat2<tag1, ptag4,
                tag1<upsample<2,
                tag5<conv<8 * nf, 1, 1,
                     SUBNET>>>>>>>>>>>>>>>>>>>>>>>>>>>;

        using net_type = loss_yolo<ytag3, ytag4, ytag5, head<backbone<input_rgb_image>>>;
    };

    using train_type_n = def<leaky_relu, bn_con, 1, 3, 1, 4>::net_type;
    using infer_type_n = def<leaky_relu, affine, 1, 3, 1, 4>::net_type;
    using train_type_s = def<leaky_relu, bn_con, 1, 3, 1, 2>::net_type;
    using infer_type_s = def<leaky_relu, affine, 1, 3, 1, 2>::net_type;
    using train_type_m = def<leaky_relu, bn_con, 2, 3, 3, 4>::net_type;
    using infer_type_m = def<leaky_relu, affine, 2, 3, 3, 4>::net_type;
    using train_type_l = def<leaky_relu, bn_con, 1, 1, 1, 1>::net_type;
    using infer_type_l = def<leaky_relu, affine, 1, 1, 1, 1>::net_type;
    using train_type_x = def<leaky_relu, bn_con, 4, 3, 5, 4>::net_type;
    using infer_type_x = def<leaky_relu, affine, 4, 3, 5, 4>::net_type;
}

#endif // yolov5_h_INCLUDED
