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
- **主仓库接入 ✅**（T9，2026-11-01）：主仓库 `neuralnet.cpp` 以 submodule 挂载本仓库
  （`engine/`），全 cli 改链双引擎 .so（只 include 抽象层头 + 工厂 + 链接）；LGPL
  可替换验证 = `bench/lgpl_verify.sh`（符号级 + patched `libnn-cpu.so` → `layer_bench
  --op matmul --hash` 原 hash → patched hash → 复原）。边界清单/实施记录：
  `neuralnet.cpp/docs/internal/34` §10。

## 引擎级测试（tests/，消费方形态）

引擎回归从主仓库迁入本仓 `tests/`，**只 include MIT 抽象层接口头 + 引擎侧 LGPL 头，
引擎经工厂创建（真实实现编在 .so），GPU 不可用 → 返回 77 = ctest SKIP**：

```bash
cmake -B build -G Ninja -DNN_BUILD_TESTS=ON      # 默认 Release（-march=native，见下）
cmake --build build
NN_VULKAN_DEVICE=850M ctest --test-dir build -j1   # 10/10 全绿（GPU 测试用权威设备）
```

| 测试 | 链接 | 覆盖 |
|------|------|------|
| `consumer_test` / `gpu_consumer_test` / `kernelgen_smoke` | .so | 消费方最小闭环 / GPU 跨界 ABI / JIT 生成器 C ABI |
| `gpu_test` | nn_cpu + nn_gpu | 引擎原语 CPU↔GPU 对拍（matmul/转置/归约/扫描）+ 性能 |
| `gpu_f16_test` | nn_cpu + nn_gpu | f16 存储变体对拍（reduce/grouped/scatter/im2col/col2im/扫描族） |
| `scan_bitwise_test` | nn_cpu + nn_gpu | 扫描原语 **逐位**回归（61 用例，dk 16/31/32/33/64 × causal/state/bnd × f32/f16） |
| `gpt_offload_test` / `rapt_offload_test` | nn_gpu | GPT/RAPT activation offload 一致性（vs 全存基线） |
| `expr_graph_test` | nn_cpu | IR-C 图融合分析（链/边界/上限/确定性/多输出） |
| `expr_fuse_test` | nn_cpu + nn_gpu | begin_expr/end_expr 融合 GPU 端到端（3 节点链/matmul 链/Adam 金刚石/批量 vs 逐个逐字节一致） |

**验证基线（2026-11-01，本机 850M）**：ctest 10/10 全绿；`scan_bitwise_test` 61/61
逐位一致（Linux Release + 850M 权威口径；Windows/Debug 按设计 77 = SKIP）。

**⚠ 两条编译口径**：
1. **默认 Release 必须带 `-march=native`**：CPU 参考需要 FMA 收缩，否则与 GPU kernel
   （glslc 默认 FMA）累加路径不同 → `scan_bitwise_test` 位型漂移（裸 `-O3` 实测 57/61
   失败，加 `-march=native` 后 61/61）。CMakeLists 已内置（`NN_ENABLE_NATIVE=OFF` 可关，
   但逐位锚测试将 SKIP）。
2. **`expr_fuse_test` 的闭合世界登记**：其录制段链（`tests/expr_fuse_shapes.hpp` 单一
   事实源）由 `scan_exprs` 的 **test_cover_pass** 在构建期登记进 `fused_registry.hpp`
   ——否则 GPU `end_expr` 按 IR-C 闭合世界硬报错。链定义改动时两端自动同步（共用
   同一组链函数）。

## 待办（后续）

1. **GPU .so 的 lgpl_verify 扩展**：当前只验 `libnn-cpu.so`（patched matmul → hash 变化）；
   GPU 同法 patch 原语 + GPU 观测。
2. **mem_probe 探针迁回**（可选）：主仓库探针目标随 T9 删除；如需 GPU 池账本/显存
   统计探针，可在本仓以 `NN_ENABLE_PROBES` 门控形态重建。
3. **eval_expr 消费方手编 spec 用例**：ExprSpec 载荷生产方是引擎 DSL，消费方手编 spec
   的跨界用例待引擎侧工具支持后补充（`expr_graph_test` 已覆盖图分析侧）。
