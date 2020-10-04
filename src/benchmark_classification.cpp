#include "benchmark.h"
#include "classification/alexnet.h"
#include "classification/darknet.h"
#include "classification/googlenet.h"
#include "classification/resnet.h"
#include "classification/vggnet.h"
#include "classification/vovnet.h"

#include <dlib/cmd_line_parser.h>

#define DNN_BENCH_ALEXNET 1
#define DNN_BENCH_VGGNET 1
#define DNN_BENCH_GOOGLENET 1
#define DNN_BENCH_RESNET 1
#define DNN_BENCH_DARKNET 1
#define DNN_BENCH_VOVNET 1

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
        alexnet::train net;
        benchmark("alexnet  ", net, batch_size, image_size, num_iters);
    }
#endif

#if DNN_BENCH_VGGNET
    {
        vggnet::train_11 net;
        benchmark("vggnet11 ", net, batch_size, image_size, num_iters);
    }
    {
        vggnet::train_13 net;
        benchmark("vggnet13 ", net, batch_size, image_size, num_iters);
    }
    {
        vggnet::train_16 net;
        benchmark("vggnet16 ", net, batch_size, image_size, num_iters);
    }
    {
        vggnet::train_19 net;
        benchmark("vggnet19 ", net, batch_size, image_size, num_iters);
    }
#endif

#if DNN_BENCH_GOOGLENET
    {
        googlenet::train net;
        benchmark("googlenet", net, batch_size, image_size, num_iters);
    }
#endif

#if DNN_BENCH_RESNET
    {
        resnet::train_18 net;
        benchmark("resnet18 ", net, batch_size, image_size, num_iters);
    }
    {
        resnet::train_34 net;
        benchmark("resnet34 ", net, batch_size, image_size, num_iters);
    }
    {
        resnet::train_50 net;
        benchmark("resnet50 ", net, batch_size, image_size, num_iters);
    }
    {
        resnet::train_101 net;
        benchmark("resnet101", net, batch_size, image_size, num_iters);
    }
    {
        resnet::train_152 net;
        benchmark("resnet152", net, batch_size, image_size, num_iters);
    }
#endif

#if DNN_BENCH_DARKNET
    {
        darknet::train_19 net;
        benchmark("darknet19", net, batch_size, image_size, num_iters);
    }
    {
        darknet::train_53 net;
        benchmark("darknet53", net, batch_size, image_size, num_iters);
    }
#endif

#if DNN_BENCH_VOVNET
    {
        vovnet::train_19_slim net;
        benchmark("vovnet19s", net, batch_size, image_size, num_iters);
    }
    {
        vovnet::train_19 net;
        benchmark("vovnet19 ", net, batch_size, image_size, num_iters);
    }
    {
        vovnet::train_39 net;
        benchmark("vovnet39 ", net, batch_size, image_size, num_iters);
    }
    {
        vovnet::train_57 net;
        benchmark("vovnet57 ", net, batch_size, image_size, num_iters);
    }
    {
        vovnet::train_99 net;
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
