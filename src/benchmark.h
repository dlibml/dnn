#include <dlib/dnn.h>

class visitor_con_disable_bias
{
    public:
    visitor_con_disable_bias() = default;
    // ignore other layers
    template <typename T> void operator()(size_t, T&) {}
    template <long nf, long nr, long nc, int sy, int sx, int py, int px, typename SUBNET>
    void operator()(size_t, dlib::add_layer<dlib::con_<nf, nr, nc, sy, sx, py, px>, SUBNET>& l)
    {
        l.layer_details().disable_bias();
    }
};

class visitor_count_convolutions
{
    public:
    visitor_count_convolutions(size_t& num_convolutions) : num_convolutions(num_convolutions) {}
    // ignore other layers
    template <typename T> void operator()(size_t, T&) {}
    template <long nf, long nr, long nc, int sy, int sx, int py, int px, typename SUBNET>
    void operator()(size_t, dlib::add_layer<dlib::con_<nf, nr, nc, sy, sx, py, px>, SUBNET>&)
    {
        ++num_convolutions;
    }

    private:
    size_t& num_convolutions;
};

template <typename net_type> auto benchmark(
    const std::string& name,
    net_type& net,
    const size_t batch_size = 1,
    const size_t image_size = 224,
    const int iterations = 100)
{
    using fms = std::chrono::duration<float, std::milli>;
    dlib::resizable_tensor x;
    dlib::matrix<dlib::rgb_pixel> image(image_size, image_size);
    assign_all_pixels(image, dlib::rgb_pixel(0, 0, 0));
    std::vector<dlib::matrix<dlib::rgb_pixel>> batch(batch_size, image);
    dlib::running_stats<double> rs;
    net.to_tensor(batch.begin(), batch.end(), x);
    // warmup for 10 iterations
    for (int i = 0; i < 10; ++i)
    {
        net.forward(x);
    }
    // std::cout << net << '\n';
    for (int i = 0; i < iterations; ++i)
    {
        const auto t0 = std::chrono::steady_clock::now();
        net.forward(x);
        const auto& t = net.subnet().get_output();
        t.host();
        const auto t1 = std::chrono::steady_clock::now();
        rs.add(std::chrono::duration_cast<fms>(t1 - t0).count());
    }
    std::cout << name << " inference: " << rs.mean() << " ms";
    std::cout << " (" << 1.0 / rs.mean() * 1000.0 * batch_size << " fps)";
    std::cout << " #params: " << count_parameters(net);
    std::ostringstream sout;
    serialize(net, sout);
    std::cout << " (memory usage: " << sout.str().size() / 1024.0 / 1024.0 << " MiB)";
    size_t num_convolutions = 0;
    dlib::visit_layers(net, visitor_count_convolutions(num_convolutions));
    std::cout << " #num convolutions: " << num_convolutions << ' ';
    std::cout << " #num layers: " << net_type::num_computational_layers << '\n';
    std::cin.get();
}
