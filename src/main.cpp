#include <dlib/cmd_line_parser.h>

int main(const int argc, const char** argv)
try
{
    dlib::command_line_parser parser;
    parser.set_group_name("Help Options");
    parser.add_option("h", "alias for --help");
    parser.add_option("help", "display this message and exit");
    parser.parse(argc, argv);

    if (parser.option("h") or parser.option("help"))
    {
        parser.print_options();
        return EXIT_SUCCESS;
    }

    return EXIT_SUCCESS;
}
catch (const std::exception& e)
{
    std::cout << e.what() << '\n';
    return EXIT_FAILURE;
}
