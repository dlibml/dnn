#include "benchmark.h"
#include "classification/alexnet.h"
#include "classification/darknet.h"
#include "classification/densenet.h"
#include "classification/googlenet.h"
#include "classification/resnet.h"
#include "classification/squeezenet.h"
#include "classification/vggnet.h"
#include "classification/vovnet.h"

#include <dlib/cmd_line_parser.h>

#define DNN_BENCH_ALEXNET 1
#define DNN_BENCH_VGGNET 1
#define DNN_BENCH_GOOGLENET 1
#define DNN_BENCH_RESNET 1
#define DNN_BENCH_DARKNET 1
#define DNN_BENCH_DENSENET 1
#define DNN_BENCH_VOVNET 1
#define DNN_BENCH_SQUEEZENET 1

int main(const int argc, const char** argv)
try
{

    dlib::command_line_parser parser;
    parser.add_option("batch-size", "set the batch size (default: 1)", 1);
    parser.add_option("image-size", "set the image size (default: 224)", 1);
    parser.add_option("num-iters", "set the number of iterations (default: 100)", 1);
    parser.add_option("no-cuda-blocking", "disable cuda synchronization");
    parser.set_group_name("Help Options");
    parser.add_option("h", "alias for --help");
    parser.add_option("help", "display this message and exit");
    parser.parse(argc, argv);

    if (parser.option("h") or parser.option("help"))
    {
        parser.print_options();
        return EXIT_SUCCESS;
    }

    const std::string cuda_blocking = parser.option("no-cuda-blocking") ? "0" : "1";
    const size_t batch_size = dlib::get_option(parser, "batch-size", 1);
    const size_t image_size = dlib::get_option(parser, "image-size", 224);
    const int num_iters = dlib::get_option(parser, "num-iters", 100);
    setenv("CUDA_LAUNCH_BLOCKING", cuda_blocking.c_str(), 1);
    std::cout << std::fixed << std::setprecision(3);

#if DNN_BENCH_ALEXNET
    {
        alexnet::train tnet;
        dlib::disable_duplicative_bias(tnet);
        alexnet::infer net(tnet);
        benchmark("alexnet  ", net, batch_size, image_size, num_iters);
    }
#endif

#if DNN_BENCH_SQUEEZENET
    {
        squeezenet::train_v1_0 tnet;
        dlib::disable_duplicative_bias(tnet);
        squeezenet::infer_v1_0 net(tnet);
        benchmark("sqznet1.0", net, batch_size, image_size, num_iters);
    }
    {
        squeezenet::train_v1_1 tnet;
        dlib::disable_duplicative_bias(tnet);
        squeezenet::infer_v1_1 net(tnet);
        benchmark("sqznet1.1", net, batch_size, image_size, num_iters);
    }
#endif

#if DNN_BENCH_VGGNET
    {
        vggnet::train_11 tnet;
        dlib::disable_duplicative_bias(tnet);
        vggnet::infer_11 net(tnet);
        benchmark("vggnet11 ", net, batch_size, image_size, num_iters);
    }
    {
        vggnet::train_13 tnet;
        dlib::disable_duplicative_bias(tnet);
        vggnet::infer_13 net(tnet);
        benchmark("vggnet13 ", net, batch_size, image_size, num_iters);
    }
    {
        vggnet::train_16 tnet;
        dlib::disable_duplicative_bias(tnet);
        vggnet::infer_16 net(tnet);
        benchmark("vggnet16 ", net, batch_size, image_size, num_iters);
    }
    {
        vggnet::train_19 tnet;
        dlib::disable_duplicative_bias(tnet);
        vggnet::infer_19 net(tnet);
        benchmark("vggnet19 ", net, batch_size, image_size, num_iters);
    }
#endif

#if DNN_BENCH_GOOGLENET
    {
        googlenet::train tnet;
        dlib::disable_duplicative_bias(tnet);
        googlenet::infer net(tnet);
        benchmark("googlenet", net, batch_size, image_size, num_iters);
    }
#endif

#if DNN_BENCH_RESNET
    {
        resnet::train_18 tnet;
        dlib::disable_duplicative_bias(tnet);
        resnet::infer_18 net(tnet);
        benchmark("resnet18 ", net, batch_size, image_size, num_iters);
    }
    {
        resnet::train_34 tnet;
        dlib::disable_duplicative_bias(tnet);
        resnet::infer_34 net(tnet);
        benchmark("resnet34 ", net, batch_size, image_size, num_iters);
    }
    {
        resnet::train_50 tnet;
        dlib::disable_duplicative_bias(tnet);
        resnet::infer_50 net(tnet);
        benchmark("resnet50 ", net, batch_size, image_size, num_iters);
    }
    {
        resnet::train_101 tnet;
        dlib::disable_duplicative_bias(tnet);
        resnet::infer_101 net(tnet);
        benchmark("resnet101", net, batch_size, image_size, num_iters);
    }
    {
        resnet::train_152 tnet;
        dlib::disable_duplicative_bias(tnet);
        resnet::infer_152 net(tnet);
        benchmark("resnet152", net, batch_size, image_size, num_iters);
    }
#endif

#if DNN_BENCH_DARKNET
    {
        darknet::train_19 tnet;
        dlib::disable_duplicative_bias(tnet);
        darknet::infer_19 net(tnet);
        benchmark("darknet19", net, batch_size, image_size, num_iters);
    }
    {
        darknet::train_53 tnet;
        dlib::disable_duplicative_bias(tnet);
        darknet::infer_53 net(tnet);
        benchmark("darknet53", net, batch_size, image_size, num_iters);
    }
    {
        darknet::train_53csp tnet;
        dlib::disable_duplicative_bias(tnet);
        darknet::infer_53csp net(tnet);
        benchmark("darknet53csp", net, batch_size, image_size, num_iters);
    }
#endif

#if DNN_BENCH_DENSENET
    {
        densenet::train_121 tnet;
        dlib::disable_duplicative_bias(tnet);
        densenet::infer_121 net(tnet);
        benchmark("densenet121", net, batch_size, image_size, num_iters);
    }
    {
        densenet::train_169 tnet;
        dlib::disable_duplicative_bias(tnet);
        densenet::infer_169 net(tnet);
        benchmark("densenet169", net, batch_size, image_size, num_iters);
    }
    {
        densenet::train_201 tnet;
        dlib::disable_duplicative_bias(tnet);
        densenet::infer_201 net(tnet);
        benchmark("densenet201", net, batch_size, image_size, num_iters);
    }
    {
        densenet::train_264 tnet;
        dlib::disable_duplicative_bias(tnet);
        densenet::infer_264 net(tnet);
        benchmark("densenet264", net, batch_size, image_size, num_iters);
    }
#endif

#if DNN_BENCH_VOVNET
    {
        vovnet::train_19_slim tnet;
        dlib::disable_duplicative_bias(tnet);
        vovnet::infer_19_slim net(tnet);
        benchmark("vovnet19s", net, batch_size, image_size, num_iters);
    }
    {
        vovnet::train_19 tnet;
        dlib::disable_duplicative_bias(tnet);
        vovnet::infer_19 net(tnet);
        benchmark("vovnet19 ", net, batch_size, image_size, num_iters);
    }
    {
        vovnet::train_27_slim tnet;
        dlib::disable_duplicative_bias(tnet);
        vovnet::infer_27_slim net(tnet);
        benchmark("vovnet27s", net, batch_size, image_size, num_iters);
    }
    {
        vovnet::train_27 tnet;
        dlib::disable_duplicative_bias(tnet);
        vovnet::infer_27 net(tnet);
        benchmark("vovnet27 ", net, batch_size, image_size, num_iters);
    }
    {
        vovnet::train_39 tnet;
        dlib::disable_duplicative_bias(tnet);
        vovnet::infer_39 net(tnet);
        benchmark("vovnet39 ", net, batch_size, image_size, num_iters);
    }
    {
        vovnet::train_57 tnet;
        dlib::disable_duplicative_bias(tnet);
        vovnet::infer_57 net(tnet);
        benchmark("vovnet57 ", net, batch_size, image_size, num_iters);
    }
    {
        vovnet::train_99 tnet;
        dlib::disable_duplicative_bias(tnet);
        vovnet::infer_99 net(tnet);
        benchmark("vovnet99 ", net, batch_size, image_size, num_iters);
    }
#endif

    return EXIT_SUCCESS;
}
catch (const std::exception& e)
{
    std::cout << e.what() << '\n';
    return EXIT_FAILURE;
}
