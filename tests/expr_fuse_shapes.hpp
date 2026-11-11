#pragma once

// ───────────────────────────────────────────────────────────────────────────
//  expr_fuse_shapes.hpp — expr_fuse_test 的链定义（单一事实源）
//
//  每条录制链的每一步一个小函数：运行期用例（expr_fuse_test.cpp）与构建期
//  自登记（expr_fuse_shapes.cpp 的 NN_EXPR_SCAN 静态初始化 dry-run）**调用
//  同一组函数**——融合结构由表达式类型决定，共用函数即保证两端逐表达式
//  一致（不会漂移）。改链 = 改这里，两端自动同步。
// ───────────────────────────────────────────────────────────────────────────

#include <array>
#include <cstddef>
#include <cstdint>

// 链定义模板只依赖 DSL 表达式构造（消费方 TU 求值）与 Tensor/Scalar 值类型：
// 接口头走 MIT 抽象层（compute_engine.hpp），DSL 头走引擎侧 LGPL 头
// （expr_dsl.hpp）——与主仓库消费方形态一致（引擎 .so 化 T9）。
#include <neuralnet.cpp/compute_engine.hpp>
#include <neuralnet.cpp/expr_dsl.hpp>
#include <neuralnet.cpp/expr_fold.hpp> // P10-③ fold 头链（make_fold_rowsum）

namespace nn::test_fuse
{

// 链 1（3 节点）：t1 = x*2；t2 = t1+3；t3 = t2*row_broadcast(g)
template <typename Eng>
[[nodiscard]] auto chain3_a(Eng& eng, const Tensor& x, std::size_t R, std::size_t C)
{ return dsl::compute(eng, dsl::leaf(x) * Scalar{2}, R, C); }

template <typename Eng>
[[nodiscard]] auto chain3_b(Eng& eng, const Tensor& t, std::size_t R, std::size_t C)
{ return dsl::compute(eng, dsl::leaf(t) + Scalar{3}, R, C); }

template <typename Eng>
[[nodiscard]] auto chain3_c(Eng& eng, const Tensor& t, const Tensor& g, std::size_t R,
                            std::size_t C)
{ return dsl::compute(eng, dsl::leaf(t) * dsl::row_broadcast(g), R, C); }

// 链 2（独立分支 B）：u = w+1（分支 A 复用 chain3_a）
template <typename Eng>
[[nodiscard]] auto indep_b(Eng& eng, const Tensor& w, std::size_t R, std::size_t C)
{ return dsl::compute(eng, dsl::leaf(w) + Scalar{1}, R, C); }

// 链 3（S5 matmul）：t1 = matmul(a0,b0)+bias；t2 = max(t1, 0)（relu）
template <typename Eng>
[[nodiscard]] auto mm_a(Eng& eng, const Tensor& a0, const Tensor& b0, const Tensor& bias,
                        std::size_t M, std::size_t N)
{ return dsl::compute(eng, dsl::matmul(a0, b0, false, false, 1) + dsl::leaf(bias), M, N); }

template <typename Eng>
[[nodiscard]] auto mm_b(Eng& eng, const Tensor& t, std::size_t M, std::size_t N)
{ return dsl::compute(eng, dsl::max(dsl::leaf(t), Scalar{0}), M, N); }

// 单表达式（未录制回归 / vec4 路径）：out = x*row_broadcast(g)*2
template <typename Eng>
[[nodiscard]] auto one_shot_rb2(Eng& eng, const Tensor& x, const Tensor& g, std::size_t R,
                                std::size_t C)
{ return dsl::compute(eng, dsl::leaf(x) * dsl::row_broadcast(g) * Scalar{2}, R, C); }

// ── 链 4（Adam 金刚石，P2 多输出）────────────────────────────────────────
//   K1: m = 0.9·m0 + 0.1·g        （状态写穿 → m 张量）
//   K2: v = 0.99·v0 + 0.01·g²     （状态写穿 → v 张量）
//   K3: d = 2·m − 3·v             （纯中间量，依赖 K1/K2 —— 金刚石合流）
//   K4: p += d（compute_into 目标传递 → p 张量）
// 目标：fuse 成**单 kernel 4 输出**（tail=p，extras=[m, v, d]）。
template <typename Eng>
[[nodiscard]] auto adam_a(Eng& eng, const Tensor& m0, const Tensor& g, std::size_t R, std::size_t C)
{
    // 超参走 RParam（同真实 adam_update_ 的 lr/β/ε 形态）——拼接器的
    // RParam 槽偏移曾漏过（GPU 锚位偏移真凶），本链即其回归锁
    return dsl::compute(eng, dsl::leaf(m0) * dsl::rparam(0.9f) + dsl::leaf(g) * dsl::rparam(0.1f),
                        R, C);
}

template <typename Eng>
[[nodiscard]] auto adam_b(Eng& eng, const Tensor& v0, const Tensor& g, std::size_t R, std::size_t C)
{
    return dsl::compute(
        eng, dsl::leaf(v0) * dsl::rparam(0.99f) + dsl::leaf(g) * dsl::leaf(g) * dsl::rparam(0.01f),
        R, C);
}

template <typename Eng>
[[nodiscard]] auto adam_c(Eng& eng, const Tensor& m, const Tensor& v, std::size_t R, std::size_t C)
{
    return dsl::compute(eng, dsl::leaf(m) * dsl::rparam(2.0f) - dsl::leaf(v) * dsl::rparam(3.0f), R,
                        C);
}

template <typename Eng> [[nodiscard]] auto adam_d_into(Eng& eng, Tensor& p, const Tensor& d)
{ return dsl::compute_into(eng, dsl::leaf(p) + dsl::leaf(d), p); }

// ── 链 5（P10-② 归约头 + 广播应用）──────────────────────────────────────
//   r = row_sum(x)（归约向量 (R,1)）；y = x − row_broadcast(r)·0.25
//   录制图里是两个节点：归约边界 + 唯一消费者的广播应用。期望融合成单
//   kernel（reduce shader 的 vector_out=0 广播路径，尾链读 s_red）。
template <typename Eng>
[[nodiscard]] auto rsum_a(Eng& eng, const Tensor& x, std::size_t R, std::size_t C)
{ return dsl::compute_reduce(eng, dsl::row_reduce_sum(dsl::leaf(x)), R, C); }

template <typename Eng>
[[nodiscard]] auto rsum_b(Eng& eng, const Tensor& x, const Tensor& r, std::size_t R,
                          std::size_t C)
{ return dsl::compute(eng, dsl::leaf(x) - dsl::row_broadcast(r) * Scalar{0.25f}, R, C); }

// ── 链 6（P10-③ fold 头 + 逐元素尾链）──────────────────────────────────────
//   s = fold_rowsum(x)（fold 输出网格 (R,1)）；y = s * row_broadcast(g)
// 录制图 = fold 节点 + 唯一消费者的逐元素节点 → 期望融合为**单 kernel**
// （fold spec 顶层尾链 y = Σ · g；域规则见 validate_expr_spec 的 fold 段）。
template <typename Eng>
[[nodiscard]] auto fold_sum_a(Eng& eng, const Tensor& x, std::size_t R, std::uint32_t K)
{
    const std::array<Tensor, 1> ins{x};
    return eng.eval_expr(nn::expr::make_fold_rowsum(K), ins, R, 1);
}

template <typename Eng>
[[nodiscard]] auto fold_tail_b(Eng& eng, const Tensor& s, const Tensor& g, std::size_t R)
{ return dsl::compute(eng, dsl::leaf(s) * dsl::row_broadcast(g), R, 1); }

// ── 链 7（P10-③ v2）：双域 fold（mm 段 + vecacc，输出网格 (R, veclen)）────
//   s = fold_mm_chain(A,B,V)；y = s * 2（逐元素消费）→ 录制期拼成单 kernel
//   （fold v2 尾链，GLSL 'T' 上下文 / dd 步进循环内发射）。
// make_fold_mm_chain 原定义在 expr_graph_test（#T5 Ash 分档用例），P10-③ 起
// 两处测试共用 → 迁至本共享 fixture（结构说明见其注释）。
inline ExprSpec make_fold_mm_chain(std::uint32_t fold_k, std::uint32_t mm_k,
                                   std::uint32_t veclen = 8)
{
    const auto ins = [](ExprOp op, std::uint8_t dst, ExprOperand a, ExprOperand b = {})
    {
        ExprInstr i;
        i.op = static_cast<uint8_t>(op);
        i.dst = dst;
        i.a = a;
        i.b = b;
        return i;
    };
    ExprSpec s;
    FoldSpec f;
    f.k = fold_k;
    f.num_state = 1;
    f.inits = {Scalar{0}};
    f.vec_state_len = veclen;
    f.matmul = MatmulSpec{0, 1, 0, 0, mm_k, 1}; // A=输入0 B=输入1 batch=1
    f.vecacc = VecAccSpec{0, 4, 2, 0, 0};       // weight=r4 V=输入2 无 scale
    f.body = {
        ins(ExprOp::Add, 1, nn::expr::matmul_op(), nn::expr::cst(0)),
        ins(ExprOp::Exp, 2, nn::expr::reg(1)),
        ins(ExprOp::RowSum, 3, nn::expr::reg(2)),
        ins(ExprOp::Add, 0, nn::expr::reg(0), nn::expr::reduce(3)),
        ins(ExprOp::Exp, 4, nn::expr::reg(2)),
    };
    f.finalize = {ins(ExprOp::Add, 5, nn::expr::vec_state(0), nn::expr::cst(0))};
    s.fold = std::move(f);
    s.consts = {Scalar{0}};
    s.num_regs = 6;
    s.views = {nn::expr::linear(), nn::expr::linear(), nn::expr::linear()};
    return s;
}

template <typename Eng>
[[nodiscard]] auto fold_mm_a(Eng& eng, const Tensor& a0, const Tensor& b0, const Tensor& v,
                             std::size_t R, std::uint32_t fold_k, std::uint32_t mm_k,
                             std::uint32_t veclen = 8)
{
    const std::array<Tensor, 3> ins{a0, b0, v};
    return eng.eval_expr(make_fold_mm_chain(fold_k, mm_k, veclen), ins, R, veclen);
}

template <typename Eng>
[[nodiscard]] auto fold_mm_tail_b(Eng& eng, const Tensor& s, std::size_t R, std::uint32_t veclen)
{ return dsl::compute(eng, dsl::leaf(s) * Scalar{2}, R, veclen); }

} // namespace nn::test_fuse
