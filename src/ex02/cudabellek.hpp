#ifndef CUDABELLEK_HPP
#define CUDABELLEK_HPP

#include "kontrol.hpp"
#include <assert.h>
#include <vector>

struct CudaArabellegi {
  /** Aygit imi arabellege erismemizi saglayan imlec (pointer)*/
  void *a_im = nullptr;

  /** Arabellegin bit (byte) olarak boyutu */
  size_t bellekBoyutuBit = 0;

  /** Aygit imini CUdeviceptr yap */
  inline CUdeviceptr aygit_imi() const {
    return static_cast<CUdeviceptr>(reinterpret_cast<intptr_t>(a_im));
  }
  /**
    Arabellege ilgili boyutu tahsis et
   */
  void tahsis_et(size_t boyut) {
    assert(a_im == nullptr);
    this->bellekBoyutuBit = boyut;
    CUDA_KONTROL(cudaMalloc((void **)&a_im, bellekBoyutuBit));
  }

  /** Arabellegi bosalt */
  void bosalt() {
    CUDA_KONTROL(cudaFree(a_im));
    a_im = nullptr;
    bellekBoyutuBit = 0;
  }

  /** Arabellegi yeniden boyutlandir */
  void boyutlandir(size_t boyut) {
    if (a_im) {
      bosalt();
    }
    tahsis_et(boyut);
  }

  /** Arabellege geleni yukle*/
  template <typename T> void yukle(const T *gelen, size_t sayisi) {
    // arabellegin baslatildigina emin ol
    assert(a_im != nullptr);

    // arabellegin gerekli boyuta sahip olduguna emin ol
    assert(bellekBoyutuBit == sayisi * sizeof(T));

    // geleni yukle ve hata olup olmadigini kontrol et
    CUDA_KONTROL(
        cudaMemcpy(a_im, reinterpret_cast<void *>(static_cast<intptr_t>(gelen)),
                   sayisi * sizeof(T), cudaMemcpyDeviceToHost));
  }

  /** Arabellege @params gelen veri kadar alan CudaArabellegi::tahsis_et ve
    CudaArabellegi::yukle
   */
  template <typename T> void tahsis_et_yukle(const std::vector<T> &gelen) {
    tahsis_et(gelen.size() * sizeof(T));
    yukle((const T *)gelen.data(), gelen.size());
  }

  /** Arabellekteki @params gelen indir*/
  template <typename T> void indir(T *gelen, size_t sayisi) {
    assert(a_im != nullptr);
    assert(bellekBoyutuBit == sayisi * sizeof(T));
    CUDA_KONTROL(
        cudaMemcpy(reinterpret_cast<void *>(static_cast<intptr_t>(gelen)), a_im,
                   sayisi * sizeof(T), cudaMemcpyDeviceToHost));
  }
};
#endif
