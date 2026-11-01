# 计算引擎（compute.neuralnet.cpp）

从 `neuralnet.cpp` 分离出的**计算引擎独立仓库**，目标是彻底分离引擎（LGPL）与
抽象层（MIT），引擎实现编进可替换的 `.so`，以 C++ 接口直出。

## 双许可证（彻底分离的落点）

| 目录 | 许可证 | 内容 |
|------|--------|------|
| `engine/` | **LGPLv3**（`LICENSE.LGPLv3`） | 真实引擎实现：`CpuEngine` / `GpuEngine` + 原语 + DSL + shader，编译进 `.so` |
| `abstraction/` | **MIT**（`LICENSE.MIT`） | 公共 C++ 抽象接口（`compute_engine.hpp`）：只有接口声明 + 值类型，零引擎实现代码 |

**为何 MIT 抽象层不被 LGPL 传染**：抽象层只**声明**接口（纯虚方法 + 轻量值类型），
不包含任何引擎实现代码；消费方动态链接引擎 `.so`（LGPL），不把 LGPL 代码内联进
自身二进制，且可替换 `.so` → 满足 LGPL 的"可替换库二进制"承诺。因此抽象层与
消费方都以 MIT 自由使用。

## 架构

```
消费方（cli/model/layer）          只 include 抽象层头 + 链接引擎 .so
        │
抽象层（MIT，abstraction/include）  nn::ComputeEngine 完整接口（52 虚方法）
                                    + Tensor/Matrix/ExprSpec/Precision 值类型
        ├──────────────┐
libnn-cpu.so (LGPL)   libnn-gpu.so (LGPL)
CpuEngine + CPU 原语    GpuEngine + Vulkan backend + shader + kernelgen
        └── 引擎实现全编进对应 .so，C++ 直出 ──┘
```

- **引擎直出 C++（完整接口）**：`nn::create_cpu_engine()` / `create_gpu_engine(device)`
  返回 `shared_ptr<ComputeEngine>`；Tensor 是**真实值类型**（布局在 MIT 抽象层头），
  `from_matrix` / `matmul` 等返回 `Result<Tensor>`（`std::expected`）直接跨界，
  无句柄映射、无适配器层。
- **许可边界**：抽象层 = 引擎接口声明 + 跨界值类型（MIT，无引擎实现）；引擎具体
  实现（CpuEngine/GpuEngine 类体、原语、Vulkan、shader）编进 .so（LGPL，可替换）。
  Layer/Model/DSL 模板面在主仓库（MIT）。
- **跨界 ABI 契约（C++ 直出的前提）**：
  1. 消费方与 .so **同编译器 / 同标准库 / 同 `-fno-exceptions`** 编译；
  2. **Tensor 布局与 `NN_HAS_VULKAN` 无关**（2026-11 重构）：GpuBuffer 用 PIMPL
     （Vulkan 句柄藏在引擎侧 `Impl`），抽象层头零 Vulkan 依赖 → 消费方
     **无需同宏编译、无需 Vulkan SDK**，CPU/GPU 引擎共用同一跨界布局；
  3. GPU 值类型析构经 .so 符号（PIMPL 式），消费方不需要完整后端类型；
  4. `std::shared_ptr` / `std::expected` / `std::string` 跨界由同标准库保证。

## 构建与测试

```bash
cmake -B build -G Ninja -DNN_ENABLE_GPU=OFF   # CPU-only
cmake --build build
./build/consumer_test                          # CPU 消费方（只 include 抽象层 + 链 libnn-cpu.so）

# GPU（默认 ON，需 Vulkan + glslc）：
cmake -B build -G Ninja
cmake --build build
./build/gpu_consumer_test                      # GPU 消费方（无需宏/无需 Vulkan SDK）
LD_LIBRARY_PATH=<Vulkan SDK>/lib ./build/kernelgen_smoke   # JIT 生成器 C ABI
```

## 当前状态

- **libnn-cpu.so ✅**：真实 `CpuEngine` 编进 `.so`，完整接口 C++ 直出；`consumer_test`
  跑通 matmul / 归约 / 原地操作 / transpose；LGPL 边界 nm 干净。
- **libnn-gpu.so ✅**：真实 `GpuEngine` + Vulkan backend + glslc 管线（44 份 `*_spv.hpp`）
  + `scan_exprs` AOT 融合（158 条表达式 → `fused_registry.hpp`）；真实 850M matmul PASS。
- **libnn-kernelgen.so ✅**：运行期 JIT 算子生成器（shaderc），`nnk_generate_spirv`
  现编 SPIR-V 冒烟 PASS；GPU 引擎运行期 dlopen（懒加载，`NN_KERNELGEN_PATH` 可覆盖）。

## 待办（后续）

1. **主仓库 cli 改链 .so**：主仓库已 submodule 挂载本仓库（`engine/`），下一步把
   cli 从 header-only 引擎改为 include 抽象层头 + 链 .so（mnist_train 先，再全量）。
2. **双引擎各自 `.so` 替换验证**：对 `libnn-cpu.so` / `libnn-gpu.so` 分别做
   `bench/lgpl_verify.sh` 式替换验证（改一个原语实现，同一消费方二进制行为随新
   `.so` 变化）。
3. **eval_expr 消费方测试**：ExprSpec 载荷生产方是引擎 DSL，消费方手编 spec 的
   跨界用例待引擎侧工具支持后补充。
