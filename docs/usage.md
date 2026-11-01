# 消费方接入指南（docs/usage.md）

面向把 `compute.neuralnet.cpp` 用作计算引擎的消费方（应用 / 上层库）。引擎本体的
设计文档见主仓库 `neuralnet.cpp/docs/internal/34`（边界清单 §10 + 实施记录 §10.4）。

## 1. 接入

```cmake
# 消费方 CMakeLists.txt
add_subdirectory(<path/to/compute.neuralnet.cpp>)
target_link_libraries(my_app PRIVATE nn_cpu)          # CPU 引擎（libnn-cpu.so）
# 或：target_link_libraries(my_app PRIVATE nn_gpu)     # GPU 引擎（libnn-gpu.so，需 Vulkan）
target_include_directories(my_app PRIVATE
    <path/to/compute.neuralnet.cpp>/abstraction/include)
```

- 只链需要的引擎；GPU 消费方**无需 Vulkan SDK、无需 `NN_HAS_VULKAN` 宏**
  （Tensor 布局与 `NN_HAS_VULKAN` 无关，GpuBuffer PIMPL——Vulkan 句柄藏在引擎侧）。
- 引擎自带的测试默认关闭（`NN_BUILD_TESTS=OFF`），嵌入不拉入本仓测试。

## 2. 最小示例（CPU）

```cpp
#include <neuralnet.cpp/compute_engine.hpp>   // MIT 抽象层（唯一需要的头）

auto e = nn::create_cpu_engine();             // 失败返回 nullptr + nn::last_error()
if (!e) return 1;

nn::Matrix a(2, 3), b(3, 2);                  // 宿主 I/O 载体（MIT 值类型）
// ... 填 a/b ...
auto ta = e->from_matrix(a);                  // Result<Tensor>（std::expected）
auto tb = e->from_matrix(b);
e->begin_batch();                              // GPU 录制窗口（CPU no-op）
auto tc = e->matmul(*ta, *tb, false, false);
e->end_batch();
auto c  = e->to_matrix(*tc);                  // 下载回宿主
```

GPU 引擎把 `create_cpu_engine()` 换成 `create_gpu_engine(device)`（`device` = 设备名
子串，空指针/空串 = 自动选卡，等价 `NN_VULKAN_DEVICE` 环境变量）；初始化失败返回
`nullptr`（`nn::last_error()` 给原因），消费方可回退 CPU。

## 3. 跨界 ABI 契约（C++ 直出前提）

1. 消费方与 `.so` **同编译器 / 同标准库 / 同 `-fno-exceptions`** 编译；
2. **Tensor 布局与 `NN_HAS_VULKAN` 无关**（2026-11 PIMPL 重构）→ 消费方无需同宏
   编译、无需 Vulkan SDK，CPU/GPU 引擎共用同一跨界布局；
3. GPU 值类型析构经 .so 符号（PIMPL 式），消费方不需要完整后端类型；
4. `std::shared_ptr` / `std::expected` / `std::string` 跨界由同标准库保证。

## 4. 引擎接口面

`nn::ComputeEngine` 是完整接口（52 个虚方法，NVI：public 非虚入口 → protected
`*_impl`）：矩阵级（matmul/转置/add/scale）、归约级（row/col/grouped_reduce）、
数据操作（slice/gather/scatter/im2col/col2im/cast）、扫描级（scan_prefix/suffix_outer）、
表达式（`eval_expr`/`eval_expr_reduce`/`eval_expr_into`，AOT 融合）、批次/内存池、
IR-C 录制（`begin_expr`/`end_expr`）、activation offload。能力**信息**只经
`describe()` 的 `EngineInfo` 披露（禁止用于选路——选路是引擎内部的事）。

层/模型/DSL 模板面（`dsl::compute` 等）在**主仓库** `neuralnet.cpp`（MIT 消费侧）；
本仓库的 `engine/include` 提供其副本供 `.so` 编译与测试使用。

## 5. LGPL 可替换验证

引擎是 LGPL：`.so` 可被替换（消费方动态链接，不内联 LGPL 代码）。替换验证口径
（本仓库 `bench/lgpl_verify.sh`，CPU 侧已验）：

- **符号级**：消费方二进制 0 个引擎实现符号（`nm -C` 无 `CpuEngine` 等）；
- **行为级**：patch `libnn-cpu.so` 的原语（如 matmul 内 scale×2）→ 同一消费方二进制
  的输出 hash 变化 → 复原 `.so` → hash 复原。正解观测点 = `layer_bench --op matmul
  --hash`（显式走引擎虚方法；训练 loss 对比不可用——DSL 模板路径在消费方 TU 求值，
  不经 .so）。

## 6. 测试 / 验证

```bash
cmake -B build -G Ninja -DNN_BUILD_TESTS=ON     # 默认 Release
cmake --build build
NN_VULKAN_DEVICE=850M ctest --test-dir build -j1   # 10/10（GPU 权威设备）
```

- 引擎级测试清单 / 覆盖 / 基线见 `README.md`「引擎级测试」节。
- ⚠ 默认 Release 必须带 `-march=native`（`scan_bitwise_test` 逐位锚的 FMA 前提）；
  GPU 不可用或口径不满足的测试返回 77 = ctest SKIP（`SKIP_RETURN_CODE` 已配）。
- `expr_fuse_test` 的录制段链由 `scan_exprs` 的 test_cover_pass 构建期登记进
  `fused_registry.hpp`（闭合世界两端一致）。
