#ifndef yolov7_h_INCLUDED
#define yolov7_h_INCLUDED

#include <dlib/dnn.h>

namespace yolov7
{
    using namespace dlib;
    template <typename SUBNET> using ytag3 = add_tag_layer<4003, SUBNET>;
    template <typename SUBNET> using ytag4 = add_tag_layer<4004, SUBNET>;
    template <typename SUBNET> using ytag5 = add_tag_layer<4005, SUBNET>;
    template <typename SUBNET> using ptag3 = add_tag_layer<7003, SUBNET>;
    template <typename SUBNET> using ptag4 = add_tag_layer<7004, SUBNET>;
    template <typename SUBNET> using ptag5 = add_tag_layer<7005, SUBNET>;
    template <typename SUBNET> using ntag4 = add_tag_layer<5004, SUBNET>;

    template <template <typename> class ACT, template <typename> class BN>
    struct def
    {

        template <long NF, int KS, int S, typename SUBNET>
        using conv = ACT<BN<add_layer<con_<NF, KS, KS, S, S, (KS-1)/2, (KS-1)/2>, SUBNET>>>;

        template <long NF, typename SUBNET>
        using transition = concat2<itag2, itag1,
                     itag2<conv<NF, 3, 2,
                           conv<NF, 1, 1, iskip<
                     itag1<conv<NF, 1, 1,
                           max_pool<2, 2, 2, 2,
                     itag0<SUBNET>>>>>>>>>;

        template <long NF, typename SUBNET>
        using e_elan = conv<NF * 4, 1, 1,
                       concat4<itag4, itag3, itag2, itag1,
                 itag4<conv<NF, 3, 1,
                       conv<NF, 3, 1,
                 itag3<conv<NF, 3, 1,
                       conv<NF, 3, 1,
                 itag2<conv<NF, 1, 1, iskip<
                 itag1<conv<NF, 1, 1,
                 itag0<SUBNET>>>>>>>>>>>>>>;

        template <long NF, template<typename> class TAG, typename SUBNET>
        using transition2 = concat3<itag2, itag1, TAG,
                      itag2<conv<NF, 3, 2,
                            conv<NF, 1, 1, iskip<
                      itag1<conv<NF, 1, 1,
                            max_pool<2, 2, 2, 2,
                      itag0<SUBNET>>>>>>>>>;

        template <long NF, typename SUBNET>
        using e_elan2 = conv<NF, 1, 1,
                   add_layer<concat_<tag1, tag2, tag3, tag4, tag5, tag6>,
                   tag1<conv<NF / 2, 3, 1,
                   tag2<conv<NF / 2, 3, 1,
                   tag3<conv<NF / 2, 3, 1,
                   tag4<conv<NF / 2, 3, 1,
                   tag5<conv<NF, 1, 1, iskip<
                   tag6<conv<NF, 1, 1,
                  itag0<SUBNET>>>>>>>>>>>>>>>>;

        template <typename INPUT>
        using backbone = ptag5<e_elan<256, transition<512,
                         ptag4<e_elan<256, transition<256,
                         ptag3<e_elan<128, transition<128,
                               e_elan<64, conv<128, 3, 2,
                               conv<64, 3, 1, conv<64, 3, 2,
                               conv<32, 3, 1, INPUT>>>>>>>>>>>>>>;

        template <long NF, typename SUBNET>
        using sppcspc = conv<NF, 1, 1,
                        concat2<tag1, tag2,
                   tag2<conv<NF, 1, 1, iskip<
                   tag1<conv<NF, 3, 1, conv<NF, 1, 1,
                        concat4<itag1, itag2, itag3, itag4,
                  itag4<max_pool<5, 5, 1, 1,
                  itag3<max_pool<5, 5, 1, 1,
                  itag2<max_pool<5, 5, 1, 1,
                  itag1<conv<NF, 1, 1, conv<NF, 3, 1, conv<NF, 1, 1,
                  itag0<SUBNET>>>>>>>>>>>>>>>>>>>>;

        template <template <typename> class YTAG, typename SUBNET>
        using yolo = YTAG<sig<con<255, 1, 1, 1, 1, SUBNET>>>;

        template <typename SUBNET>
        using head = yolo<ytag5,
                     e_elan2<512,
                     transition2<256, tag8, skip9<
                     yolo<ytag4,
                tag9<e_elan2<256,
                     transition2<128, tag7, skip9<
                     yolo<ytag3,
                tag9<e_elan2<128,
                     concat2<tag2, tag1,
                tag2<conv<128, 1, 1, add_skip_layer<ptag3,
                tag1<upsample<2,
                     conv<128, 1, 1,
                tag7<e_elan2<256,
                     concat2<tag2, tag1,
                tag2<conv<256, 1, 1, add_skip_layer<ptag4,
                tag1<upsample<2,
                     conv<256, 1, 1,
               tag8<sppcspc<512,
               SUBNET>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>;

        using net_type = loss_yolo<ytag3, ytag4, ytag5, head<backbone<input_rgb_image>>>;
    };

    using train_type = def<silu, bn_con>::net_type;
    using infer_type = def<silu, affine>::net_type;
}

#endif // yolov7_h_INCLUDED
