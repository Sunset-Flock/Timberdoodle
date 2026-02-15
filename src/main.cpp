#include "application.hpp"

#if 0
#include "tex_compression/test.hpp"
int main(int argc, char const * const * argv)
{
    test_main();
}
#else
int main(int argc, char const * const * argv)
{

    Application app = Application();
    if (argc > 1)
    {
        // try loading first argument as scene
        app.load_scene(argv[1]);
    }

    return app.run();
}
#endif