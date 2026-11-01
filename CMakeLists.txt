# ═══════════════════════════════════════════════════════════════════════════
#  compute.neuralnet.cpp — 计算引擎（引擎 LGPL + 抽象层 MIT）
#
#  目标：引擎实现编进独立 .so（libnn-cpu.so / libnn-gpu.so），以抽象层（MIT）
#  的 neuralnet::ComputeEngine C++ 接口直出。消费方只 include 抽象层头 + 链接 .so。
#
#  目录：
#    engine/       LGPL —— 真实引擎实现（CpuEngine/GpuEngine + 原语 + DSL + shader）
#    abstraction/  MIT  —— 公共 C++ 抽象接口（compute_engine.hpp）
#    tests/        消费方验证
# ═══════════════════════════════════════════════════════════════════════════
cmake_minimum_required(VERSION 3.30)
project(compute_neuralnet_cpp VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 与源引擎一致的严格旗标（-fno-exceptions 是 ABI 前提：消费方与 .so 必须同款）
add_compile_options(-fno-exceptions -Wall -Wextra -Wpedantic)
find_package(Threads REQUIRED)

# std::stacktrace 运行时库探测（仅 tools/scan_exprs 用；GCC/Clang+libstdc++ 需
# stdc++exp 或 stdc++_libbacktrace，见源仓库 CMakeLists.txt §247）
include(CheckCXXSourceCompiles)
set(NN_STACKTRACE_LIB "")
foreach(_lib stdc++exp stdc++_libbacktrace)
    if(NN_STACKTRACE_LIB)
        break()
    endif()
    set(CMAKE_REQUIRED_LIBRARIES ${_lib})
    check_cxx_source_compiles("
        #include <stacktrace>
        int main() { return std::stacktrace::current().empty() ? 1 : 0; }
    " NN_HAS_STACKTRACE_${_lib})
    unset(CMAKE_REQUIRED_LIBRARIES)
    if(NN_HAS_STACKTRACE_${_lib})
        set(NN_STACKTRACE_LIB ${_lib})
    endif()
endforeach()
if(NN_STACKTRACE_LIB)
    message(STATUS "std::stacktrace 运行时库: ${NN_STACKTRACE_LIB}")
endif()

set(ABST_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/abstraction/include)
set(ENGINE_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/engine/include)
set(GENERATED_INCLUDE ${CMAKE_CURRENT_BINARY_DIR}/generated)

# ── libnn-cpu.so（LGPL）：CPU 引擎 + 原语，C++ 直出 ───────────────────────
add_library(nn_cpu SHARED engine/src/cpu_engine_so.cpp)
target_include_directories(nn_cpu PRIVATE ${ABST_INCLUDE} ${ABST_INCLUDE}/neuralnet.cpp ${ENGINE_INCLUDE} ${ENGINE_INCLUDE}/neuralnet.cpp)
target_link_libraries(nn_cpu PRIVATE Threads::Threads)
set_target_properties(nn_cpu PROPERTIES
    OUTPUT_NAME nn-cpu
    CXX_VISIBILITY_PRESET default
    BUILD_RPATH ${CMAKE_CURRENT_BINARY_DIR})
install(TARGETS nn_cpu LIBRARY DESTINATION lib)

# ═══════════════════════════════════════════════════════════════════════════
#  libnn-gpu.so（LGPL）：GPU 引擎 + Vulkan backend + 原语 shader
#  构建期 glslc 管线生成 *_spv.hpp（NN_SPV_GETTER_LIST 用 __has_include 探测）。
# ═══════════════════════════════════════════════════════════════════════════
option(NN_ENABLE_GPU "构建 GPU 引擎 libnn-gpu.so" ON)
if(NN_ENABLE_GPU)
    find_package(Vulkan QUIET)
    find_program(GLSLC glslc HINTS /home/ethan/vulkan/1.4.357.1/x86_64/bin ${Vulkan_GLSLC_EXECUTABLE})
    if(Vulkan_FOUND AND GLSLC)
        set(SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/engine/shaders)

        # glslc → .spv → embed_spirv.cmake → C++ 头文件（同源仓库 nn_embed_shader）
        # 用 macro（非 function）：宏在调用方作用域展开，list(APPEND ...) 直接累积
        # 到调用方的 GPU_GENERATED_HPP，确保 gpu_generated 依赖全部生成头。
        set(GPU_GENERATED_HPP "")
        macro(nn_embed_shader SHADER_NAME SRC_FILE)
            set(SPV ${CMAKE_CURRENT_BINARY_DIR}/${SHADER_NAME}.spv)
            set(HPP ${GENERATED_INCLUDE}/${SHADER_NAME}_spv.hpp)
            set(FUNC "nn_${SHADER_NAME}_spirv_bytecode")
            add_custom_command(
                OUTPUT ${SPV}
                COMMAND ${GLSLC} -I ${SHADER_DIR} ${ARGN} -o ${SPV} ${SRC_FILE}
                DEPENDS ${SRC_FILE} ${SHADER_DIR}/nn_f16_rne.glsl
                COMMENT "[GPU] Compiling ${SHADER_NAME}.comp → ${SHADER_NAME}.spv")
            add_custom_command(
                OUTPUT ${HPP}
                COMMAND ${CMAKE_COMMAND} -E make_directory ${GENERATED_INCLUDE}
                COMMAND ${CMAKE_COMMAND}
                    -DSPV_INPUT=${SPV} -DHPP_OUTPUT=${HPP} -DFUNC_NAME=${FUNC}
                    -P ${CMAKE_CURRENT_SOURCE_DIR}/engine/cmake/embed_spirv.cmake
                DEPENDS ${SPV} ${CMAKE_CURRENT_SOURCE_DIR}/engine/cmake/embed_spirv.cmake
                COMMENT "[GPU] Embedding ${SHADER_NAME} SPIR-V into C++ header")
            list(APPEND GPU_GENERATED_HPP ${HPP})
        endmacro()

        # 1. 普通（手写）纯原语 shader
        set(GPU_SHADERS
            matmul matmul_tiled batched_matmul rearrange_3d elementwise_v2 transpose
            gather scatter_add im2col col2im group_reduce scan_prefix_outer
            scan_suffix_outer scan_prefix_outer_gen scan_suffix_outer_gen
            outer_col trit_pack matmul_q cast)
        foreach(SHADER_NAME ${GPU_SHADERS})
            nn_embed_shader(${SHADER_NAME} ${SHADER_DIR}/${SHADER_NAME}.comp)
        endforeach()

        # 1b. 小 N GEMV / reduce：subgroup 归约 → SPIR-V ≥1.3 → vulkan1.2
        nn_embed_shader(matmul_gemv ${SHADER_DIR}/matmul_gemv.comp --target-env=vulkan1.2)
        nn_embed_shader(matmul_q_gemv ${SHADER_DIR}/matmul_q_gemv.comp --target-env=vulkan1.2)
        nn_embed_shader(reduce ${SHADER_DIR}/reduce.comp --target-env=vulkan1.2)

        # 1d. 精度/网格变体（-DNN_SHADER_F16 / -DNN_MM_FINE / -DNN_SCAN_DK_MAX）
        nn_embed_shader(batched_matmul_f16 ${SHADER_DIR}/batched_matmul.comp -DNN_SHADER_F16=1)
        nn_embed_shader(matmul_tiled_f16 ${SHADER_DIR}/matmul_tiled.comp -DNN_SHADER_F16=1)
        nn_embed_shader(matmul_tiled_fine ${SHADER_DIR}/matmul_tiled.comp -DNN_MM_FINE=1)
        nn_embed_shader(matmul_gemv_f16 ${SHADER_DIR}/matmul_gemv.comp --target-env=vulkan1.2 -DNN_SHADER_F16=1)
        nn_embed_shader(scan_prefix_outer_f16 ${SHADER_DIR}/scan_prefix_outer.comp -DNN_SHADER_F16=1)
        nn_embed_shader(scan_suffix_outer_f16 ${SHADER_DIR}/scan_suffix_outer.comp -DNN_SHADER_F16=1)
        nn_embed_shader(scan_prefix_outer_d32 ${SHADER_DIR}/scan_prefix_outer.comp -DNN_SCAN_DK_MAX=32)
        nn_embed_shader(scan_prefix_outer_d32_f16 ${SHADER_DIR}/scan_prefix_outer.comp -DNN_SCAN_DK_MAX=32 -DNN_SHADER_F16=1)
        nn_embed_shader(scan_prefix_outer_gen_f16 ${SHADER_DIR}/scan_prefix_outer_gen.comp -DNN_SHADER_F16=1)
        nn_embed_shader(scan_suffix_outer_gen_f16 ${SHADER_DIR}/scan_suffix_outer_gen.comp -DNN_SHADER_F16=1)
        nn_embed_shader(outer_col_f16 ${SHADER_DIR}/outer_col.comp -DNN_SHADER_F16=1)
        nn_embed_shader(transpose_f16 ${SHADER_DIR}/transpose.comp -DNN_SHADER_F16=1)
        nn_embed_shader(rearrange_3d_f16 ${SHADER_DIR}/rearrange_3d.comp -DNN_SHADER_F16=1)
        nn_embed_shader(gather_f16 ${SHADER_DIR}/gather.comp -DNN_SHADER_F16=1)
        nn_embed_shader(elementwise_v2_f16 ${SHADER_DIR}/elementwise_v2.comp -DNN_SHADER_F16=1)
        nn_embed_shader(reduce_f16 ${SHADER_DIR}/reduce.comp --target-env=vulkan1.2 -DNN_SHADER_F16=1)
        nn_embed_shader(scatter_add_f16 ${SHADER_DIR}/scatter_add.comp -DNN_SHADER_F16=1)
        nn_embed_shader(group_reduce_f16 ${SHADER_DIR}/group_reduce.comp -DNN_SHADER_F16=1)
        nn_embed_shader(im2col_f16 ${SHADER_DIR}/im2col.comp -DNN_SHADER_F16=1)
        nn_embed_shader(col2im_f16 ${SHADER_DIR}/col2im.comp -DNN_SHADER_F16=1)
        nn_embed_shader(matmul_q_f16 ${SHADER_DIR}/matmul_q.comp -DNN_SHADER_F16=1)
        nn_embed_shader(matmul_q_gemv_f16 ${SHADER_DIR}/matmul_q_gemv.comp --target-env=vulkan1.2 -DNN_SHADER_F16=1)

        # 2. AOT 融合 shader：scan_exprs（单步 = 收集 + 生成）→ fused_registry.hpp
        set(NN_FUSED_REGISTRY_HPP ${GENERATED_INCLUDE}/fused_registry.hpp)
        add_executable(scan_exprs engine/tools/scan_exprs.cpp)
        target_include_directories(scan_exprs PRIVATE ${ABST_INCLUDE} ${ABST_INCLUDE}/neuralnet.cpp ${ENGINE_INCLUDE} ${ENGINE_INCLUDE}/neuralnet.cpp)
        target_link_libraries(scan_exprs PRIVATE Threads::Threads Vulkan::Vulkan ${NN_STACKTRACE_LIB})
        add_custom_command(
            OUTPUT ${NN_FUSED_REGISTRY_HPP}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${GENERATED_INCLUDE}
            COMMAND scan_exprs ${GENERATED_INCLUDE} ${GLSLC}
            DEPENDS scan_exprs ${CMAKE_CURRENT_SOURCE_DIR}/engine/tools/scan_exprs.cpp
                    ${CMAKE_CURRENT_SOURCE_DIR}/engine/tools/fused_generate.hpp
            COMMENT "[GPU] Scan + synthesize fused shaders (scan_exprs) → fused_registry.hpp")
        list(APPEND GPU_GENERATED_HPP ${NN_FUSED_REGISTRY_HPP})

        # 3. libnn-gpu.so：依赖全部生成头
        add_library(nn_gpu SHARED engine/src/gpu_engine_so.cpp)
        target_include_directories(nn_gpu PRIVATE ${ABST_INCLUDE} ${ABST_INCLUDE}/neuralnet.cpp ${ENGINE_INCLUDE} ${ENGINE_INCLUDE}/neuralnet.cpp ${GENERATED_INCLUDE})
        target_link_libraries(nn_gpu PRIVATE Threads::Threads Vulkan::Vulkan)
        target_compile_definitions(nn_gpu PRIVATE NN_HAS_VULKAN=1)
        add_custom_target(gpu_generated DEPENDS ${GPU_GENERATED_HPP})
        add_dependencies(nn_gpu gpu_generated)
        set_target_properties(nn_gpu PROPERTIES
            OUTPUT_NAME nn-gpu
            CXX_VISIBILITY_PRESET default
            BUILD_RPATH ${CMAKE_CURRENT_BINARY_DIR})
        install(TARGETS nn_gpu LIBRARY DESTINATION lib)
        message(STATUS "libnn-gpu.so 启用（Vulkan + glslc + ${GPU_SHADERS} + fused 融合）")
    else()
        message(STATUS "Vulkan/glslc 不可用，跳过 libnn-gpu.so")
    endif()
endif()

# ── libnn-kernelgen.so（LGPL）：运行期 JIT 算子生成器（GPU 引擎运行期 dlopen）──
# 引擎 .nnk 缓存 miss → dlopen 本 .so → nnk_generate_spirv 现编 SPIR-V。
# 内嵌 GlslEmitter（IR→GLSL）+ shaderc（GLSL→SPIR-V），与构建期 glslc 同源。
find_path(SHADERC_INCLUDE_DIR shaderc/shaderc.h
    HINTS ${Vulkan_INCLUDE_DIRS} /home/ethan/vulkan/1.4.357.1/x86_64/include)
find_library(SHADERC_LIBRARY NAMES shaderc_shared
    HINTS ${Vulkan_LIBRARY_DIR} /home/ethan/vulkan/1.4.357.1/x86_64/lib)
if(SHADERC_INCLUDE_DIR AND SHADERC_LIBRARY)
    add_library(nn_kernelgen SHARED engine/src/kernelgen.cpp)
    target_include_directories(nn_kernelgen PRIVATE ${ABST_INCLUDE} ${ABST_INCLUDE}/neuralnet.cpp ${ENGINE_INCLUDE} ${ENGINE_INCLUDE}/neuralnet.cpp ${SHADERC_INCLUDE_DIR})
    target_link_libraries(nn_kernelgen PRIVATE Threads::Threads ${SHADERC_LIBRARY})
    set_target_properties(nn_kernelgen PROPERTIES
        OUTPUT_NAME nn-kernelgen
        CXX_VISIBILITY_PRESET default
        BUILD_RPATH ${CMAKE_CURRENT_BINARY_DIR})
    install(TARGETS nn_kernelgen LIBRARY DESTINATION lib)
    message(STATUS "libnn-kernelgen.so 启用（JIT 算子生成器，shaderc）")
else()
    message(STATUS "shaderc 未找到，libnn-kernelgen.so 跳过（GPU JIT 不可用，原语不受影响）")
endif()

# ── 消费方验证：只 include 抽象层（MIT）+ 链接 .so ────────────────────────
add_executable(consumer_test tests/consumer_test.cpp)
target_include_directories(consumer_test PRIVATE ${ABST_INCLUDE})
target_link_libraries(consumer_test PRIVATE nn_cpu)
set_target_properties(consumer_test PROPERTIES
    BUILD_RPATH ${CMAKE_CURRENT_BINARY_DIR})
enable_testing()
add_test(NAME consumer_test COMMAND consumer_test)

if(TARGET nn_gpu)
    add_executable(gpu_consumer_test tests/gpu_consumer_test.cpp)
    target_include_directories(gpu_consumer_test PRIVATE ${ABST_INCLUDE} ${Vulkan_INCLUDE_DIRS})
    # 跨界 ABI 契约：Tensor 布局随 NN_HAS_VULKAN 条件编译 → 消费方必须与 .so
    # 同宏编译（GpuBuffer 析构经 .so 符号，消费方无需完整后端类型）。
    target_compile_definitions(gpu_consumer_test PRIVATE NN_HAS_VULKAN=1)
    target_link_libraries(gpu_consumer_test PRIVATE nn_gpu)
    set_target_properties(gpu_consumer_test PROPERTIES
        BUILD_RPATH ${CMAKE_CURRENT_BINARY_DIR})
    add_test(NAME gpu_consumer_test COMMAND gpu_consumer_test)
endif()

if(TARGET nn_kernelgen)
    add_executable(kernelgen_smoke tests/kernelgen_smoke.cpp)
    target_include_directories(kernelgen_smoke PRIVATE ${ABST_INCLUDE} ${ABST_INCLUDE}/neuralnet.cpp ${ENGINE_INCLUDE})
    target_link_libraries(kernelgen_smoke PRIVATE nn_kernelgen)
    set_target_properties(kernelgen_smoke PROPERTIES
        BUILD_RPATH ${CMAKE_CURRENT_BINARY_DIR})
    add_test(NAME kernelgen_smoke COMMAND kernelgen_smoke)
endif()
