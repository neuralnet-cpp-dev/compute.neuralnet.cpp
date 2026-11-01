# 完整适配方案：抽象层升级为 nn::ComputeEngine 完整接口面（T7）

> 状态：**已实施**（T9 于 2026-11-01 落地；本文件是当时的实施蓝图，落地差异见
> 末尾"实施差异"节。实施记录与验证基线见主仓库 `neuralnet.cpp/docs/internal/34`
> §10.4 与 `docs/history/history.md`「引擎 .so 化（T9）」条）。
> 目标：新仓库抽象层（MIT）从"最小原语面"升级为**主仓库 layer/model/dsl 能直接
> 使用的完整引擎接口面**；主仓库 cli 改链 .so；引擎实现全部收敛到 .so（LGPL）。

## 1. 目标架构（与 docs/internal/34 §10 一致，C++ 直出版）

```
主仓库 cli / layer / model / dsl（模板面，header）    include 抽象层头（MIT）
        │
抽象层（MIT，abstraction/include/neuralnet.cpp）        nn::ComputeEngine 接口（52 虚方法）
                                                        + Tensor/Matrix/ExprSpec/Precision 值类型
        ├──────────────┐
libnn-cpu.so (LGPL)   libnn-gpu.so (LGPL)              CpuEngine/GpuEngine 具体实现 + 原语 + shader
```

- **许可边界**：抽象层 = 引擎**接口声明 + 跨界值类型**（MIT，无引擎实现）；
  引擎具体实现（CpuEngine/GpuEngine 类体、原语、Vulkan、shader）编进 .so（LGPL）。
  主仓库的 Layer/Loss/Optimizer/Model/DSL 模板面**留在主仓库**（LGPL，主仓库内部
  使用不涉及传染；外部消费方用模板面接触 LGPL 是主仓库既有事实，本次不改）。
- **跨界协议（C++ 直出）**：Tensor 是真实值类型（MatrixT / shared_ptr<GpuBuffer>
  布局在 MIT 头），.so 与消费方按同一布局编译（同编译器/旗标/`-fno-exceptions`）→
  ABI 一致。`std::expected`（Result）同理由两侧同标准库保证。
- **工厂**：抽象层头提供 `nn::CpuEngine::create()` / `nn::GpuEngine::create()` 的
  **转发声明**（具体类体在 .so），或自由函数工厂；消费方不链引擎实现头。

## 2. 头拆分清单

### 2.1 从 engine/include 复制到 abstraction/include（MIT，接口+值类型）

| 头 | 内容 | 说明 |
|----|------|------|
| `compute_engine.hpp` | ComputeEngine 接口（纯虚 `*_impl` + NVI 转发体）+ EngineInfo | 接口本尊 |
| `compute_tensor.hpp` | Tensor 值类型（MatrixT/GpuBuffer 布局） | **跨界值类型，布局即 ABI** |
| `compute_gpu_storage.hpp` | GpuBuffer 定义 | Tensor 析构需要其布局 |
| `precision.hpp` | Precision / PrecisionProfile | 值类型 |
| `core_config.hpp` / `core_errors.hpp` / `core_assert.hpp` / `core_env.hpp` / `core_domain.hpp` | 常量 / Result / 断言 / env / 域 | 基础面 |
| `algebra_matrix.hpp` / `algebra_span.hpp` / `algebra_ops.hpp` | Matrix/Scalar/运算符 | 值类型 |
| `expr_spec.hpp` | ExprSpec/ExprInstr/MatmulSpec/FoldSpec | 表达式序列化载荷 |
| `expr_opt.hpp` | canonicalize/validate | 纯函数（消费方 DSL 需要） |
| `kernel_cache.hpp` | encode/decode_expr_spec + .nnk 缓存 | 边界载荷编解码 |
| `core_parallel.hpp` | 并行循环模板 | 模板面 |

### 2.2 留在 engine/include（LGPL，编进 .so）

`compute_cpu_engine.hpp` / `compute_gpu_engine.hpp` / `backend/*` /
`expr_dsl.hpp`（CPU 求值）/ `expr_graph.hpp` / `expr_emitter.hpp` /
`expr_glsl_gen.hpp` / `expr_registry.hpp` / `expr_op_table.hpp` /
`expr_fold.hpp` / `ternary_pack.hpp`。

### 2.3 主仓库（LGPL，保持现状）

Layer/Loss/Optimizer/Model/DSL 模板头、cli 源、训练循环、分词器/数据集。
主仓库**删除**自己的引擎实现头（compute_cpu_engine.hpp / compute_gpu_engine.hpp /
backend/ 等），include 路径改指 submodule 抽象层头 + 链 .so。

## 3. .so 改造

- `cpu_engine_so.cpp` / `gpu_engine_so.cpp` 的适配器从"最小原语面"升级为**完整
  52 虚方法 NVI 面**（Tensor 句柄映射删除，改为直接持有真实 nn::Tensor 值类型）；
- 导出 `nn::CpuEngine::create()` 等 C++ 符号（用户裁定 C++ 直出，不做 C ABI）；
- `eval_expr` 走完整 ExprSpec（含 matmul 段/fold/批量），JIT miss 接 kernelgen。

## 4. 分步

1. **抽象层头拆分**（2.1 列表）：复制 + 从 engine/include 移除（避免双份漂移），
   `.so` 适配器升级为完整接口面，内部冒烟（模拟 layer 调 matmul/reduce/eval_expr）。
2. **主仓库 include 改指 + cli 改链**：`mnist_train` 先改链 .so 跑通训练/推理；
   再全 cli（text/tokenizer/dataset）批量迁移。
3. **双轨收敛**：header-only 引擎面收敛到抽象层转发，ctest 双轨全绿；
   对 libnn-cpu/libnn-gpu 分别做 `.so` 替换验证（lgpl_verify.sh 式）。

## 5. 关键风险

- **Tensor 跨界布局**（最大）：MatrixT/GpuBuffer 布局跨 .so，任何布局变更 =
  ABI 变更，两侧必须同步重编（同编译器/旗标保证）。
- **模板面与 .so 的精度/能力交互**：Layer 精度槽校验在 init 期调引擎
  `describe()` 能力——能力信息须在完整接口面里可见。
- **双轨维护**：拆分后 engine/include 与 abstraction/include 不再有重复接口头，
  只保留实现侧依赖（单向：LGPL 头 include MIT 头）。

## 6. 实施差异（T9 落地后，2026-11-01）

- **工厂形态**：§1 提到"`nn::CpuEngine::create()` 转发声明或自由函数工厂"→ 实际为
  **自由函数工厂** `nn::create_cpu_engine()` / `nn::create_gpu_engine(device)` /
  `nn::last_error()`（声明在抽象层 `compute_engine.hpp`，实现在 `engine/src/*_engine_so.cpp`）。
  抽象层**没有** `CpuEngine`/`GpuEngine` 具体类（消费方统一用 `nn::ComputeEngine&` 接口）。
- **2.1 头拆分实际落点**：`expr_graph.hpp`（图 IR + 融合分析）与 §2.2 的预期相反，也在
  **抽象层**（`begin_expr/end_expr` 的 ExprGraph 类型跨界）；`compute_gpu_storage.hpp`
  未单独成头（GpuBuffer 定义并入 `compute_tensor.hpp`，PIMPL 化后布局与 `NN_HAS_VULKAN`
  无关）；`kernel_cache.hpp` 留在主仓库（.nnk 缓存属消费侧工具）；`core_parallel.hpp`
  在抽象层（模板面）。**原则**：凡"消费方 TU 需要构造/读取的类型"都进抽象层，
  凡"引擎实现体"都留 engine/include——`expr_spec/expr_opt/expr_graph/precision/core_*/algebra_*`
  是跨界唯一版本（.so 与消费方同份编译），无双份漂移。
- **DSL 头归属**：`expr_dsl.hpp`（CPU 模板求值）在 engine/include（LGPL）——消费方
  layer/model 必须 include 它才能写 `dsl::compute`（模板在消费方 TU 实例化、数值与
  .so 一致）；这是 LGPL 头的消费方 include（动态链接语境），设计如此。
- **引擎级测试**：主仓库引擎级测试迁入 `tests/`（消费方形态，见 README"引擎级测试"节），
  共 10 个 ctest；`scan_exprs` 增 **test_cover_pass**（登记 `tests/expr_fuse_shapes.hpp`
  链，闭合世界两端一致）。
- **未做**：§2.1 的 `kernel_cache.hpp` 迁移（留主仓库）；`core_parallel.hpp` 已在抽象层。
