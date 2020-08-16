# cmake file
include_directories({OptiX_INCLUDE})

cuda_compile_and_embed(koyulan_ptx_kodu aygitProgrami.cu)

add_executable(renkciktisi.out
    ${koyulan_ptx_kodu}
  "kontrol.hpp"
  "cudabellek.hpp"
  "cizer.hpp"
  "renkciktisi.cpp"
  # Compiled cuda ptx programs
)
target_link_libraries(renkciktisi.out
  ${optix_LIBRARY}
  ${CUDA_LIBRARIES}
  ${CUDA_CUDA_LIBRARY}
)

