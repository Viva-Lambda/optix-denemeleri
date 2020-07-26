include_directories(.)

cuda_compile_and_embed(
  drawColor_ptx_c drawColor.cu
)

add_executable(hello.out
  "hello.cpp"
  # Compiled cuda ptx programs
)

target_link_libraries(hello.out
  ${optix_LIBRARY}
  ${CUDA_LIBRARIES}
  ${CUDA_CUDA_LIBRARY}
)

