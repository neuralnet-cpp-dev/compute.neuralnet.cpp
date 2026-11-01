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
抽象层（MIT，header-only C++）      neuralnet::ComputeEngine 抽象接口 + Tensor 值类型
        ├──────────────┐
libnn-cpu.so (LGPL)   libnn-gpu.so (LGPL)
CpuEngine + CPU 原语    GpuEngine + Vulkan backend + shader + kernelgen
        └── 引擎实现全编进对应 .so，C++ 直出 ──┘
```

- **引擎直出 C++**（区别于源仓库的 C ABI `nn_api.h`）：消费方 `create_cpu_engine()` /
  `create_gpu_engine(device)` 得到 `std::unique_ptr<neuralnet::ComputeEngine>`，直接调
  用虚方法，不经过 C 句柄/序列化层。
- **跨 `.so` 边界协议**：`Tensor` 是轻量值类型（不透明句柄 + 缓存维度），真实存储由
  引擎托管（引擎析构连带回收）；`Tensor` 拷贝 = 别名；错误经 `last_error()`。
  STL 仅 `std::unique_ptr` / `std::string` 跨界 → 要求消费方与 `.so` 同编译器 /
  同标准库 / 同 `-fno-exceptions`。
- **边界验证**：`nm` 消费方二进制零引擎实现符号（引擎全在 `.so`），只引用工厂符号。

## 构建与测试

```bash
cmake -B build -G Ninja -DNN_ENABLE_GPU=OFF   # CPU-only（本轮已交付）
cmake --build build
./build/consumer_test                          # 消费方验证（只 include 抽象层 + 链 libnn-cpu.so）
```

`NN_ENABLE_GPU=ON`（默认）时尝试构建 `libnn-gpu.so`，需 Vulkan + glslc（见下）。

## 当前状态

- **libnn-cpu.so ✅ 已交付并验证**：真实 `CpuEngine` 编进 `.so`，经 MIT 抽象层
  C++ 直出；`consumer_test` 跑通 matmul / 行列归约 / transpose / clone /
  add_inplace / scale_inplace；`nm` 验证 LGPL 边界干净。
- **libnn-gpu.so ✅ 已交付并验证**：真实 `GpuEngine` + Vulkan backend + 原语 shader
  管线 + AOT 融合（`scan_exprs` → `fused_registry.hpp`）编进 `.so`；构建期 glslc
  管线生成 44 份 `*_spv.hpp`；`gpu_consumer_test` 在真实 GPU（GTX 850M）跑通
  matmul；`nm` 验证 LGPL 边界干净。

## 待办（后续）

1. **kernelgen（`.nnk` JIT）**：GPU 引擎在表达式 JIT miss 时运行期 dlopen
   `libnn-kernelgen.so`（懒加载，非链接依赖）。当前 GPU `.so` 未迁 kernelgen，
   eval_expr 的 JIT 回退路径会因加载失败而报错；原语（matmul/reduce）不受影响。
   后续迁 `src/kernelgen/kernelgen.cpp`（依赖 shaderc）+ 生成器 `.so`。
2. **双引擎各自 `.so` 替换验证**：对 `libnn-cpu.so` / `libnn-gpu.so` 分别做
   `bench/lgpl_verify.sh` 式替换验证（改一个原语实现，同一消费方二进制行为随新
   `.so` 变化）。
3. **双轨测试**：ctest 覆盖 CPU/GPU 双引擎路径的更多原语（当前仅 matmul +
   归约 + 数据操作冒烟）。
