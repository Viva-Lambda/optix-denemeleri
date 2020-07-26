// see License
#include <cuda_runtime.h>
#include <iostream>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <stdexcept>

#define GDT_TERMINAL_RED "\033[1;31m"
#define GDT_TERMINAL_GREEN "\033[1;32m"
#define GDT_TERMINAL_YELLOW "\033[1;33m"
#define GDT_TERMINAL_BLUE "\033[1;34m"
#define GDT_TERMINAL_RESET "\033[0m"
#define GDT_TERMINAL_DEFAULT GDT_TERMINAL_RESET
#define GDT_TERMINAL_BOLD "\033[1;1m"

int OPTIX_CHECK(OptixResult call) {
  OptixResult res = call;
  if (res != OPTIX_SUCCESS) {
    std::cerr << "Optix call" << call << " failed with code" << res << " (line "
              << __LINE__ << std::endl;
    return 2;
  }
  return 0;
}

/*! \namespace osc - Optix Siggraph Course */
namespace osc {

/*! helper function that initializes optix and checks for errors */
void initOptix() {
  // -------------------------------------------------------
  // check for available optix7 capable devices
  // -------------------------------------------------------
  cudaFree(0);
  int numDevices;
  cudaGetDeviceCount(&numDevices);
  if (numDevices == 0)
    throw std::runtime_error("#osc: no CUDA capable devices found!");
  std::cout << "#osc: found " << numDevices << " CUDA devices" << std::endl;

  // -------------------------------------------------------
  // initialize optix
  // -------------------------------------------------------
  OPTIX_CHECK(optixInit());
}

/*! main entry point to this example - initially optix, print hello
  world, then exit */
extern "C" int main(int ac, char **av) {
  try {
    std::cout << "#osc: initializing optix..." << std::endl;

    initOptix();

    std::cout << GDT_TERMINAL_GREEN
              << "#osc: successfully initialized optix... yay!"
              << GDT_TERMINAL_DEFAULT << std::endl;

    // for this simple hello-world example, don't do anything else
    // ...
    std::cout << "#osc: done. clean exit." << std::endl;

  } catch (std::runtime_error &e) {
    std::cout << GDT_TERMINAL_RED << "FATAL ERROR: " << e.what()
              << GDT_TERMINAL_DEFAULT << std::endl;
    exit(1);
  }
  return 0;
}
}
