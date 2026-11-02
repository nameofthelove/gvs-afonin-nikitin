# Библиотека заголовочная: кёрнел и функция запуска — шаблоны с параметрами фильтра времени
# компиляции, экземпляры порождаются в единицах трансляции CUDA потребителя.
add_library(hsys_nn INTERFACE)
add_library(hsys::nn ALIAS hsys_nn)

target_include_directories(hsys_nn INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Цель torch уже переносит на потребителя каталоги заголовков и TORCH_CXX_FLAGS.
target_link_libraries(hsys_nn INTERFACE
    "${TORCH_LIBRARIES}"
    CUDA::cudart
)

# Флаги, с которыми torch.utils.cpp_extension компилирует расширения: c10::Half и
# c10::BFloat16 определяют собственные операторы и преобразования, встроенные операторы
# cuda_fp16.h / cuda_bf16.h с ними конфликтуют.
set(HSYS_TORCH_NVCC_FLAGS
    -D__CUDA_NO_HALF_OPERATORS__
    -D__CUDA_NO_HALF_CONVERSIONS__
    -D__CUDA_NO_BFLOAT16_CONVERSIONS__
    -D__CUDA_NO_HALF2_OPERATORS__
    --expt-relaxed-constexpr
)
target_compile_options(hsys_nn INTERFACE
    "$<$<COMPILE_LANGUAGE:CUDA>:${HSYS_TORCH_NVCC_FLAGS}>"
)

if(MSVC)
  # Исходники в UTF-8 (комментарии на русском); остальные предупреждения — из заголовков
  # LibTorch, их подавляет и cpp_extension.
  set(HSYS_MSVC_FLAGS /utf-8 /EHsc /wd4819 /wd4251 /wd4244 /wd4267 /wd4275 /wd4018 /wd4190
                      /wd4624 /wd4067 /wd4068)
  string(REPLACE ";" "," HSYS_MSVC_FLAGS_CUDA "${HSYS_MSVC_FLAGS}")
  target_compile_options(hsys_nn INTERFACE
      "$<$<COMPILE_LANGUAGE:CXX>:${HSYS_MSVC_FLAGS}>"
      "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${HSYS_MSVC_FLAGS_CUDA}>"
  )
  # Компоновщик MSVC отбрасывает torch_cuda.dll, если на неё нет прямых ссылок; тогда
  # не регистрируются CUDA-реализации операторов и torch::cuda::is_available() == false.
  target_link_options(hsys_nn INTERFACE "/INCLUDE:?warp_size@cuda@at@@YAHXZ")
endif()
