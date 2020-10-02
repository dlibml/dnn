#include "classification/benchmark.h"
#include "classification/darknet.h"
#include "classification/resnet.h"
#include "classification/vovnet.h"

int main(/*const int argc, const char** argv*/)
try
{
    setenv("CUDA_LAUNCH_BLOCKING", "1", 1);
    std::cout << std::fixed << std::setprecision(3);

    {
        resnet::train_18 net;
        measure("resnet18 ", net);
    }

    {
        resnet::train_34 net;
        measure("resnet34 ", net);
    }

    {
        resnet::train_50 net;
        measure("resnet50 ", net);
    }

    {
        resnet::train_101 net;
        measure("resnet101", net);
    }

    {
        resnet::train_152 net;
        measure("resnet152", net);
    }

    {
        darknet::train_19 net;
        measure("darknet19", net);
    }

    {
        darknet::train_53 net;
        measure("darknet53", net);
    }

    {
        vovnet::train_19_slim net;
        measure("vovnet19s", net);
    }

    {
        vovnet::train_19 net;
        measure("vovnet19 ", net);
    }

    {
        vovnet::train_39 net;
        measure("vovnet39 ", net);
    }

    {
        vovnet::train_57 net;
        measure("vovnet57 ", net);
    }

    {
        vovnet::train_99 net;
        measure("vovnet99 ", net);
    }

    return EXIT_SUCCESS;
}
catch (const std::exception& e)
{
    std::cout << e.what() << '\n';
    return EXIT_FAILURE;
}
