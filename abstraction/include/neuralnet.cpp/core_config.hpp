#pragma once

#include <cstddef>

#include "precision.hpp"

// NN_VECTORIZE_PRAGMA：循环向量化提示的编译器分支实现，统一定义在 core_config.hpp
#if defined(__clang__)
#define NN_VECTORIZE_PRAGMA _Pragma("clang loop vectorize(assume_safety)")
#elif defined(__GNUC__) || defined(__GNUG__)
#define NN_VECTORIZE_PRAGMA _Pragma("GCC ivdep")
#elif defined(_MSC_VER)
    // MSVC 的 #pragma loop(ivdep) / __pragma(loop(ivdep)) 在 lambda/复合语句内部
    // 会导致 C2059/C3925 编译错误（解析器将 { 误判为循环体）。
    // MSVC 在 /O2 下默认启用自动向量化，此处留空即可。
#define NN_VECTORIZE_PRAGMA
#elif defined(__ICC) || defined(__INTEL_COMPILER)
#define NN_VECTORIZE_PRAGMA _Pragma("vector always")
#else
#define NN_VECTORIZE_PRAGMA
#endif

// ── 缓存分块大小 ─────────────────────────────────────────────────────────────
// 64×64×8 = 32 KB，安全装入大多数 CPU 的 L1 缓存
// 矩阵乘法、转置等所有分块操作共用此值
// 修改时需同步评估 b_block 栈占用（BLOCK_SIZE² × 8 字节）
namespace nn
{
// ── 标量类型（float 加速 / double 精度） ─────────────────────────
using Scalar = float;

inline constexpr std::size_t BLOCK_SIZE = 64;
static_assert(BLOCK_SIZE * BLOCK_SIZE * sizeof(Scalar) <= 65536,
              "BLOCK_SIZE too large: b_block would exceed 64KB stack budget");
// ── 数值常量 ─────────────────────────────────────────────────────────────
// 使用 constexpr 避免运行时计算
inline constexpr Scalar EPSILON = 1e-6f;

// exstl::PARALLEL_THRESHOLD 已迁 <exstl/parallel.hpp>（exstl::PARALLEL_THRESHOLD）。

// ── 诊断探针编译门（默认关；CMake `-DNN_ENABLE_PROBES=ON` 才编入）────────
// "探针" = 会在**热路径**采样/打印/下载/扫描的诊断代码：层内逐中间量扫描
//（`nn_dbg_scan` / `NN_F16_DEBUG`）、精度变体与边界 cast 归因
//（`NN_PREC_TRACE` / `note_temp_`）、训练循环的非有限值追踪（`NN_NAN_TRACE`）、
// 显存阶段采样（`NN_MEM_STATS`），以及本仓探针可执行目标（`f16_cpu_probe` /
// `gpu_stability_probe`；显存分项探针 `mem_probe` 已随引擎 .so 化迁新仓库）。
// 关闭（默认）时它们**不参与编译**：正常训练没有额外分支、没有字符串、
// 没有 host 下载路径，相关环境变量一律无效——探针只在排障时按需打开，
// 不允许"设了环境变量就悄悄拖慢正常训练"。
// 新增探针一律走这个门；初始化期读一次的**行为开关**（`NN_VULKAN_DEVICE`
// 正确性检查、`NN_VULKAN_*` 设备/特性逃生阀、`NN_POOL_*` 池 A/B）不在此列。
// 门经 `nn_core` 的 INTERFACE 定义传播 → 所有 TU（含嵌入消费方）取值一致。
#if defined(NN_ENABLE_PROBES) && NN_ENABLE_PROBES
#define NN_PROBES_ENABLED 1
#else
#define NN_PROBES_ENABLED 0
#endif

} // namespace nn
