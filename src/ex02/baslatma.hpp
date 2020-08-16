#ifndef BASLATMA_HPP
#define BASLATMA_HPP
#include <glm/glm.hpp>

struct BaslatmaParametreleri {
  int CerceveNo{0};
  uint32_t *renkArabellegi;
  glm::ivec2 cerceveBoyutu;
};

#endif
