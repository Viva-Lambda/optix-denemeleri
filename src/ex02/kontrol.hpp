// kontrol fonksiyonlari
#ifndef KONTROL_HPP
#define KONTROL_HPP

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>

#include <sstream>
#include <stdexcept>

bool OPTIX_KONTROL(OptixResult sonuc) {
  if (sonuc != OPTIX_SUCCESS) {
    std::stringstream uyari;
    uyari << "Optix ifadesi su kodla" << sonuc << "(satir " << __LINE__ << " )"
          << " basarisiz olmustur." << std::endl;
    throw std::runtime_error(uyari.str());
  }
  return true;
}

bool CUDA_KONTROL(cudaError_t sonuc) {
  if (sonuc != cudaSuccess) {
    std::stringstream uyari;
    uyari << "CUDA Hatasi: " << cudaGetErrorName(sonuc) << " " << __LINE__
          << "::" __FILE__ << " (" << cudaGetErrorString(sonuc) << ")";
    throw std::runtime_error(uyari.str());
  }
  return true;
}

bool CUDA_SENKRON_KONTROL() {
  cudaDeviceSynchronize();
  cudaError_t hata = cudaGetLastError();
  if (hata != cudaSuccess) {
    std::cerr << "Senkron Hatasi: " << __FILE__ << "::" << __LINE__
              << "::" << cudaGetErrorString(hata);
    exit(2);
  }
}

#endif
