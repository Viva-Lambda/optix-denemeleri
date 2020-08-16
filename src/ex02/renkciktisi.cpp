// see licence
#include <glm/glm.hpp>
#include <string>
//
#include "cizer.hpp"

//
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <custom/stb_image_write.h>

extern "C" int main(int ac, char **argv) {

  try {
    OrnekCizer ornek;
    const glm::ivec2 cerceveBoyutu(1200, 1024);
    ornek.boyutlandir(cerceveBoyutu);
    ornek.ciz();

    std::vector<uint32_t> pikseller(cerceveBoyutu.x, cerceveBoyutu.y);
    ornek.pikseliIndir(pikseller);

    const std::string resimAdi = "optixResim.png";
    stbi_write_png(resimAdi.c_str(), cerceveBoyutu.x, cerceveBoyutu.y, 4,
                   pikseller.data(), cerceveBoyutu.x * sizeof(uint32_t));

    std::cout << "Resim cizildi ve '" << resimAdi << "' dosyasina kaydedildi."
              << std::endl;
  } catch (std::runtime_error &hata) {
    std::cout << "HATA :: " << hata.what() << std::endl;
    exit(1);
  }
  return 0;
}
