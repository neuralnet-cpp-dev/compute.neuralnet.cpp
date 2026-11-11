// ───────────────────────────────────────────────────────────────────────────
//  expr_graph_test.cpp — 图 IR + 融合分析（IR-C）单元验证（纯 CPU、无 GPU）
//
//  Tensor/Matrix 访问走引擎 API：create_tensor/write/to_matrix——铁律 #11。
//  验证内容（对应 docs/internal/03-ir-optimization.md 的验证策略）：
//   1. 链融合正确性：A→B 融合成单 kernel 后求值 == 逐节点求值
//   2. 多节点链融合（3 节点）
//   3. 融合边界：非链（无消费者 / 多消费者 / 非 Linear 引用）不融合
//   4. 归约头 + 广播应用融合（P10-②）；轴不匹配时仍为融合边界
//   5. 上限压力：拼接后超 EXPR_MAX_* 则保守放弃融合
//   6. 确定性：同一图多次融合结果完全一致（含 key）
//   7. 不变量：融合 kernel 通过 validate；views/输入顺序确定
//   8. 独立分支：不消费前序输出的相邻表达式各自独立成 kernel
//   9. 录制图 add_node 依赖识别（virtual_tag）
//  10. matmul 参与图融合（S5 两种形态）
//  11. IR-D emitter 抽象
//  12. #T5：fold Ash 容量分档（#P9）+ 链尾死存储消除（#P3）——GLSL 文本断言
//      + CPU 执行语义数值对拍
//  13. P10-③：fold 头 + 尾链融合 / fold 作链成员拒拼 / v2 双域 fold 尾链
//      （reg 与 VecState 两读法：validate + GLSL 文本 + CPU 数值）
//
//  构建/运行：cmake -B build -DNN_ENABLE_TESTS=ON && cmake --build build
//            ctest --test-dir build -R expr_graph_test
//
//  迁移说明（引擎 .so 化，T9）：消费方形态——接口/跨界头走 MIT 抽象层
//  （compute_engine.hpp / expr_spec.hpp / expr_opt.hpp / expr_graph.hpp），
//  DSL 与 emitter 头（expr_dsl.hpp / expr_emitter.hpp / expr_glsl_gen.hpp）
//  走引擎侧 LGPL 头（header-only 编译进测试；eval_expr 虚调用走 libnn-cpu.so）。
// ───────────────────────────────────────────────────────────────────────────

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include <neuralnet.cpp/compute_engine.hpp>
#include <neuralnet.cpp/compute_tensor.hpp>
#include <neuralnet.cpp/expr_spec.hpp>
#include <neuralnet.cpp/expr_opt.hpp>
#include <neuralnet.cpp/expr_graph.hpp>
#include <neuralnet.cpp/expr_dsl.hpp>
#include <neuralnet.cpp/expr_fold.hpp>     // P10-③ fold 头链样例（make_fold_rowsum）
#include <neuralnet.cpp/expr_emitter.hpp>
#include <neuralnet.cpp/expr_glsl_gen.hpp>    // 注册 GlslEmitter（IR-D）
#include "expr_fuse_shapes.hpp" // 共享 fold fixture（make_fold_mm_chain，#T5/P10-③ 共用）

using nn::Scalar;
using nn::test_fuse::make_fold_mm_chain;

namespace
{

int g_fail = 0;
#define CHECK(cond, msg)                     \
    do                                       \
    {                                        \
        if (!(cond))                         \
        {                                    \
            std::printf("[FAIL] %s\n", msg); \
            ++g_fail;                        \
        }                                    \
    } while (0)

nn::Tensor make_tensor(nn::ComputeEngine& eng, std::size_t rows, std::size_t cols, float base = 0.0f,
                       float step = 0.01f)
{
    nn::Tensor t = eng.create_tensor(rows, cols);
    std::vector<Scalar> buf(rows * cols);
    for (std::size_t i = 0; i < buf.size(); ++i)
        buf[i] = base + static_cast<float>(i) * step;
    auto w = eng.write(t, std::span<Scalar>(buf));
    if (!w)
    {
        std::printf("[FAIL] make_tensor: write 失败\n");
        ++g_fail;
    }
    return t;
}

// 录制占位 Tensor（真实存储 + virtual_tag，同旧测试 Tensor::cpu + set_virtual_tag）
nn::Tensor make_placeholder(nn::ComputeEngine& eng, std::size_t rows, std::size_t cols,
                            std::uint64_t tag, nn::Precision prec = nn::Precision::F32)
{
    nn::Tensor t = eng.create_tensor(rows, cols, prec);
    t.set_virtual_tag(tag);
    return t;
}

nn::Matrix matrix_of(nn::ComputeEngine& eng, const nn::Tensor& t)
{
    auto m = eng.to_matrix(t);
    if (!m)
    {
        std::printf("[FAIL] to_matrix 失败\n");
        ++g_fail;
        return nn::Matrix(0, 0);
    }
    return std::move(*m);
}

Scalar max_abs_diff(const nn::Matrix& a, const nn::Matrix& b)
{
    Scalar m = 0.0f;
    auto as = a.span(), bs = b.span();
    for (std::size_t i = 0; i < as.size(); ++i)
    {
        const Scalar d = std::fabs(as[i] - bs[i]);
        if (d > m) m = d;
    }
    return m;
}

// 构造逐元素一元指令：out = unary(input0) 或 out = input0 <op> const
nn::ExprSpec make_unary_const(nn::ExprOp op, Scalar c)
{
    nn::ExprSpec s;
    s.views.push_back(nn::expr::linear());
    s.consts.push_back(c);
    nn::ExprInstr in;
    in.op = static_cast<std::uint8_t>(op);
    in.dst = 0;
    in.a = nn::expr::input(0);
    in.b = nn::expr::cst(0);
    s.instrs.push_back(in);
    s.num_regs = 1;
    return s;
}

// 构造逐元素二元：out = input0 <op> input1
nn::ExprSpec make_binary(nn::ExprOp op)
{
    nn::ExprSpec s;
    s.views.push_back(nn::expr::linear());
    s.views.push_back(nn::expr::linear());
    nn::ExprInstr in;
    in.op = static_cast<std::uint8_t>(op);
    in.dst = 0;
    in.a = nn::expr::input(0);
    in.b = nn::expr::input(1);
    s.instrs.push_back(in);
    s.num_regs = 1;
    return s;
}

// 构造柱归约表达式：out = col_sum(input0)（归约边界，(1,cols) 输出）
nn::ExprSpec make_col_sum()
{
    nn::ExprSpec s;
    s.views.push_back(nn::expr::linear());
    nn::ExprInstr in;
    in.op = static_cast<std::uint8_t>(nn::ExprOp::ColSum);
    in.dst = 0;
    in.a = nn::expr::input(0);
    s.instrs.push_back(in);
    s.num_regs = 1;
    return s;
}

// 构造广播应用：out = row_broadcast(input0) <op> const
nn::ExprSpec make_row_broadcast_const(nn::ExprOp op, Scalar c)
{
    nn::ExprSpec s;
    s.views.push_back(nn::expr::row_broadcast());
    s.consts.push_back(c);
    nn::ExprInstr in;
    in.op = static_cast<std::uint8_t>(op);
    in.dst = 0;
    in.a = nn::expr::input(0);
    in.b = nn::expr::cst(0);
    s.instrs.push_back(in);
    s.num_regs = 1;
    return s;
}

// 构造归约表达式：out = row_sum(input0)（归约边界，(rows,1) 输出）
nn::ExprSpec make_row_sum()
{
    nn::ExprSpec s;
    s.views.push_back(nn::expr::linear());
    nn::ExprInstr in;
    in.op = static_cast<std::uint8_t>(nn::ExprOp::RowSum);
    in.dst = 0;
    in.a = nn::expr::input(0);
    s.instrs.push_back(in);
    s.num_regs = 1;
    return s;
}

// ── 测试 1：A→B 链融合正确性 ───────────────────────────────────────────
// A: y = x * 2；B: z = y + 1 → 融合后 z = x*2 + 1，单 kernel
void test_chain_fuse()
{
    auto eng_owner = nn::create_cpu_engine();
    nn::ComputeEngine& eng = *eng_owner;
    const std::size_t rows = 3, cols = 4;
    nn::ExprGraph g;
    nn::Tensor x = make_tensor(eng, rows, cols, 0.0f, 0.5f);

    const int nA = g.add_node(make_unary_const(nn::ExprOp::Mul, Scalar{2}),
                              std::vector<nn::Tensor>{x}, rows, cols, false);
    nn::Tensor ytag =
        make_placeholder(eng, rows, cols, g.tag_of_node[static_cast<std::size_t>(nA)]);
    const int nB = g.add_node(make_unary_const(nn::ExprOp::Add, Scalar{1}),
                              std::vector<nn::Tensor>{ytag}, rows, cols, false);
    (void)nB;

    auto kernels = fuse_expr_graph(g);
    CHECK(kernels.size() == 1, "A→B 链应融合为单个 kernel");
    if (kernels.size() != 1) return;
    CHECK(kernels[0].members.size() == 2, "kernel 应包含 2 个节点");
    CHECK(kernels[0].tail == nB, "kernel tail 应为 B 节点");
    CHECK(kernels[0].spec.extras.size() == 1, "P2 写穿：A 的输出进 extras");
    CHECK(kernels[0].inputs.size() == kernels[0].spec.views.size(),
          "kernel inputs 与 views 一一对应");

    // 融合 kernel 求值（P2 多输出：主输出 = B，extras = A 写穿物化）
    std::vector<nn::Tensor> kinputs;
    for (auto& in : kernels[0].inputs)
        kinputs.push_back(in.external);
    std::vector<nn::Tensor> kouts;
    kouts.push_back(eng.create_tensor(rows, cols));
    kouts.push_back(eng.create_tensor(rows, cols));
    auto mul_r = eng.eval_expr_multi_into(kernels[0].spec, kinputs, rows, cols, kouts);
    CHECK(static_cast<bool>(mul_r), "融合 kernel 多输出求值成功");

    // 参考：逐节点
    auto a_res = eng.eval_expr(make_unary_const(nn::ExprOp::Mul, Scalar{2}),
                               std::vector<nn::Tensor>{x}, rows, cols);
    auto b_res = eng.eval_expr(make_unary_const(nn::ExprOp::Add, Scalar{1}),
                               std::vector<nn::Tensor>{*a_res}, rows, cols);
    CHECK(static_cast<bool>(a_res) && static_cast<bool>(b_res), "参考求值成功");
    if (!mul_r || !a_res || !b_res) return;

    const nn::Matrix mf = matrix_of(eng, kouts[0]);
    const nn::Matrix mr = matrix_of(eng, *b_res);
    const Scalar err = max_abs_diff(mf, mr);
    CHECK(err < 1e-6f, "融合结果与逐节点一致");
    // P2 写穿：extras 槽 = A 的输出（x*2），应与逐节点 A 一致
    CHECK(max_abs_diff(matrix_of(eng, kouts[1]), matrix_of(eng, *a_res)) < 1e-6f,
          "写穿 extras 与逐节点 A 一致");

    // 手工验证数值：x*2+1
    const nn::Matrix mx = matrix_of(eng, x);
    bool ok = true;
    for (std::size_t i = 0; i < rows * cols; ++i)
    {
        const Scalar expect = mx.span()[i] * 2.0f + 1.0f;
        if (std::fabs(mf.span()[i] - expect) > 1e-5f) ok = false;
    }
    CHECK(ok, "融合结果与手工参考 x*2+1 一致");
    std::printf("[OK] test_chain_fuse\n");
}

// ── 测试 2：三节点链融合 ───────────────────────────────────────────────
// A: y=x*2；B: z=y+3；C: w=z-1 → 融合成单 kernel
void test_chain3_fuse()
{
    auto eng_owner = nn::create_cpu_engine();
    nn::ComputeEngine& eng = *eng_owner;
    const std::size_t rows = 2, cols = 5;
    nn::ExprGraph g;
    nn::Tensor x = make_tensor(eng, rows, cols, 1.0f, 0.3f);

    auto add_node_chain = [&](nn::ExprOp op, Scalar c, const nn::Tensor& input, std::size_t r,
                              std::size_t col) -> int
    {
        const int n =
            g.add_node(make_unary_const(op, c), std::vector<nn::Tensor>{input}, r, col, false);
        return n;
    };
    const int nA = add_node_chain(nn::ExprOp::Mul, Scalar{2}, x, rows, cols);
    nn::Tensor yt = make_placeholder(eng, rows, cols, g.tag_of_node[static_cast<std::size_t>(nA)]);
    const int nB = add_node_chain(nn::ExprOp::Add, Scalar{3}, yt, rows, cols);
    nn::Tensor zt = make_placeholder(eng, rows, cols, g.tag_of_node[static_cast<std::size_t>(nB)]);
    const int nC = add_node_chain(nn::ExprOp::Sub, Scalar{1}, zt, rows, cols);
    (void)nC;

    auto kernels = fuse_expr_graph(g);
    CHECK(kernels.size() == 1, "三节点链应融合为单个 kernel");
    if (kernels.size() != 1) return;
    CHECK(kernels[0].members.size() == 3, "kernel 应包含 3 个节点");
    CHECK(kernels[0].spec.extras.size() == 2, "P2 写穿：A/B 输出进 extras");

    std::vector<nn::Tensor> kinputs;
    for (auto& in : kernels[0].inputs)
        kinputs.push_back(in.external);
    // 输出槽序：[0]=tail(C)，[1..]=成员序去尾（A, B）
    std::vector<nn::Tensor> kouts;
    for (int k = 0; k < 3; ++k)
        kouts.push_back(eng.create_tensor(rows, cols));
    auto mul_r = eng.eval_expr_multi_into(kernels[0].spec, kinputs, rows, cols, kouts);
    if (!mul_r || nn::dsl::env_flag("NN_DUMP_GLSL"))
    {
        std::printf("  [diag] multi_into: %s\n", mul_r.error().message.c_str());
        const auto& sp = kernels[0].spec;
        std::printf("  [diag] key=%s instrs=%zu views=%zu num_regs=%u extras=%zu\n",
                    nn::expr_spec_key(sp).c_str(), sp.instrs.size(), sp.views.size(), sp.num_regs,
                    sp.extras.size());
        for (const auto& v : sp.views)
            std::printf("  [diag] view kind=%u\n", static_cast<unsigned>(v.kind));
        for (const auto& ins : sp.instrs)
            std::printf("  [diag] ins op=%u dst=%u a=(%u,%u)\n", static_cast<unsigned>(ins.op),
                        static_cast<unsigned>(ins.dst), static_cast<unsigned>(ins.a.kind),
                        static_cast<unsigned>(ins.a.idx));
        auto gl = nn::emitter_registry::make("glsl");
        if (gl) { std::printf("  [diag] ---- GLSL ----\n%s\n", gl->generate("diag", sp).c_str()); }
    }
    CHECK(static_cast<bool>(mul_r), "融合 kernel 多输出求值成功");

    // 参考：逐节点
    auto a_res = eng.eval_expr(make_unary_const(nn::ExprOp::Mul, Scalar{2}),
                               std::vector<nn::Tensor>{x}, rows, cols);
    auto b_res = eng.eval_expr(make_unary_const(nn::ExprOp::Add, Scalar{3}),
                               std::vector<nn::Tensor>{*a_res}, rows, cols);
    auto c_res = eng.eval_expr(make_unary_const(nn::ExprOp::Sub, Scalar{1}),
                               std::vector<nn::Tensor>{*b_res}, rows, cols);
    CHECK(static_cast<bool>(c_res), "参考求值成功");
    if (!mul_r || !c_res) return;

    const Scalar err = max_abs_diff(matrix_of(eng, kouts[0]), matrix_of(eng, *c_res));
    CHECK(err < 1e-6f, "三节点融合结果与逐节点一致");
    CHECK(max_abs_diff(matrix_of(eng, kouts[1]), matrix_of(eng, *a_res)) < 1e-6f,
          "写穿 extras[0] 与逐节点 A 一致");
    CHECK(max_abs_diff(matrix_of(eng, kouts[2]), matrix_of(eng, *b_res)) < 1e-6f,
          "写穿 extras[1] 与逐节点 B 一致");
    std::printf("[OK] test_chain3_fuse\n");
}

// ── 测试 3：独立分支各自成 kernel ──────────────────────────────────────
// A: y = x*2（无消费者）；B: z = w+1（外部输入 w，不依赖 A）→ 2 个 kernel
void test_independent()
{
    auto eng_owner = nn::create_cpu_engine();
    nn::ComputeEngine& eng = *eng_owner;
    const std::size_t rows = 3, cols = 3;
    nn::ExprGraph g;
    nn::Tensor x = make_tensor(eng, rows, cols, 0.0f, 0.1f);
    nn::Tensor w = make_tensor(eng, rows, cols, 2.0f, 0.2f);

    (void)g.add_node(make_unary_const(nn::ExprOp::Mul, Scalar{2}), std::vector<nn::Tensor>{x}, rows,
                     cols, false);
    (void)g.add_node(make_unary_const(nn::ExprOp::Add, Scalar{1}), std::vector<nn::Tensor>{w}, rows,
                     cols, false);

    auto kernels = fuse_expr_graph(g);
    CHECK(kernels.size() == 2, "独立节点应各自成 kernel");
    if (kernels.size() != 2) return;
    CHECK(kernels[0].members.size() == 1 && kernels[1].members.size() == 1,
          "每个 kernel 只含 1 个节点");

    // 两个 kernel 求值都正确
    std::vector<nn::Tensor> k0in;
    for (auto& in : kernels[0].inputs)
        k0in.push_back(in.external);
    std::vector<nn::Tensor> k1in;
    for (auto& in : kernels[1].inputs)
        k1in.push_back(in.external);
    auto r0 = eng.eval_expr(kernels[0].spec, k0in, rows, cols);
    auto r1 = eng.eval_expr(kernels[1].spec, k1in, rows, cols);
    CHECK(static_cast<bool>(r0) && static_cast<bool>(r1), "两个 kernel 求值成功");
    std::printf("[OK] test_independent\n");
}

// ── 测试 4：多消费者写穿融合（P2：A 被 B、C 同时消费 → 单 kernel 3 输出）──
void test_multi_consumer()
{
    auto eng_owner = nn::create_cpu_engine();
    nn::ComputeEngine& eng = *eng_owner;
    const std::size_t rows = 2, cols = 4;
    nn::ExprGraph g;
    nn::Tensor x = make_tensor(eng, rows, cols, 0.0f, 0.1f);

    // A: y = x*2（输出 y）；B: z = y+1（消费 y）；C: t = y-1（也消费 y）
    const int nA = g.add_node(make_unary_const(nn::ExprOp::Mul, Scalar{2}),
                              std::vector<nn::Tensor>{x}, rows, cols, false);
    nn::Tensor yt = make_placeholder(eng, rows, cols, g.tag_of_node[static_cast<std::size_t>(nA)]);
    (void)g.add_node(make_unary_const(nn::ExprOp::Add, Scalar{1}), std::vector<nn::Tensor>{yt},
                     rows, cols, false);
    nn::Tensor yt2 = make_placeholder(eng, rows, cols, g.tag_of_node[static_cast<std::size_t>(nA)]);
    (void)g.add_node(make_unary_const(nn::ExprOp::Sub, Scalar{1}), std::vector<nn::Tensor>{yt2},
                     rows, cols, false);

    auto kernels = fuse_expr_graph(g);
    // P2 写穿物化：A 的输出落回张量（extras）→ 多消费者不再是融合边界，
    // A/B/C 同形状连通 → 单 kernel 3 输出（成员序 A,B,C；tail=C）
    CHECK(kernels.size() == 1, "P2 写穿：多消费者连通分量融合为单 kernel");
    if (kernels.size() != 1) return;
    CHECK(kernels[0].members.size() == 3, "kernel 应包含 3 个节点");
    CHECK(kernels[0].spec.extras.size() == 2, "A/B 输出进 extras（写穿）");

    // 数值验证：三输出各自与逐节点参考一致
    std::vector<nn::Tensor> kinputs;
    for (auto& in : kernels[0].inputs)
        kinputs.push_back(in.external);
    std::vector<nn::Tensor> kouts;
    for (int k = 0; k < 3; ++k)
        kouts.push_back(eng.create_tensor(rows, cols));
    auto mul_r = eng.eval_expr_multi_into(kernels[0].spec, kinputs, rows, cols, kouts);
    CHECK(static_cast<bool>(mul_r), "多输出求值成功");
    auto a_ref = eng.eval_expr(make_unary_const(nn::ExprOp::Mul, Scalar{2}),
                               std::vector<nn::Tensor>{x}, rows, cols);
    // #T6：参考求值失败必须 FAIL（此前裸 return 跳过断言仍记绿），且
    // a_ref 未检即解引用是 UB——先断言再进后续参考。
    CHECK(static_cast<bool>(a_ref), "逐节点参考 A 求值成功");
    if (!mul_r || !a_ref) return;
    auto b_ref = eng.eval_expr(make_unary_const(nn::ExprOp::Add, Scalar{1}),
                               std::vector<nn::Tensor>{*a_ref}, rows, cols);
    auto c_ref = eng.eval_expr(make_unary_const(nn::ExprOp::Sub, Scalar{1}),
                               std::vector<nn::Tensor>{*a_ref}, rows, cols);
    CHECK(static_cast<bool>(b_ref), "逐节点参考 B 求值成功");
    CHECK(static_cast<bool>(c_ref), "逐节点参考 C 求值成功");
    if (!b_ref || !c_ref) return;
    // 槽序：[0]=tail(C)，[1]=A，[2]=B
    CHECK(max_abs_diff(matrix_of(eng, kouts[0]), matrix_of(eng, *c_ref)) < 1e-6f,
          "tail(C) 与逐节点一致");
    CHECK(max_abs_diff(matrix_of(eng, kouts[1]), matrix_of(eng, *a_ref)) < 1e-6f,
          "extras[0]=A 写穿与逐节点一致");
    CHECK(max_abs_diff(matrix_of(eng, kouts[2]), matrix_of(eng, *b_ref)) < 1e-6f,
          "extras[1]=B 写穿与逐节点一致");
    std::printf("[OK] test_multi_consumer (kernels=%zu)\n", kernels.size());
}

// ── 测试 5（P10-②）：归约头 + 广播应用融合 ─────────────────────────────
// A: y = row_sum(x)（归约向量）；B: z = y + 1（RowBroadcast 消费 y）。
// 归约 kernel 的唯一消费者是匹配轴的广播应用 → 拼成单 kernel，vector_out
// 翻转为广播网格；数值 == 逐节点（归约 → 广播）参考。
void test_reduce_broadcast_fuse()
{
    auto eng_owner = nn::create_cpu_engine();
    nn::ComputeEngine& eng = *eng_owner;
    const std::size_t rows = 3, cols = 4;
    nn::ExprGraph g;
    nn::Tensor x = make_tensor(eng, rows, cols, 0.0f, 0.1f);
    const nn::ExprSpec sum = make_row_sum();
    const nn::ExprSpec add = make_row_broadcast_const(nn::ExprOp::Add, Scalar{1});

    const int nA = g.add_node(sum, std::vector<nn::Tensor>{x}, rows, cols, true);
    nn::Tensor yt = make_placeholder(eng, rows, 1, g.tag_of_node[static_cast<std::size_t>(nA)]);
    (void)g.add_node(add, std::vector<nn::Tensor>{yt}, rows, cols, false);

    auto kernels = fuse_expr_graph(g);
    CHECK(kernels.size() == 1, "归约头 + 匹配轴广播应用应融合为单 kernel");
    if (kernels.size() != 1) return;
    CHECK(kernels[0].members.size() == 2, "kernel 应包含归约 + 消费节点");
    CHECK(!kernels[0].vector_out, "融合后输出翻转为广播全网格");
    CHECK(kernels[0].spec.instrs.size() == 2, "归约指令 + 广播应用尾链");

    std::vector<nn::Tensor> kinputs;
    for (auto& in : kernels[0].inputs)
        kinputs.push_back(in.external);
    auto fused_r = eng.eval_expr(kernels[0].spec, kinputs, rows, cols);
    auto y_ref = eng.eval_expr_reduce(sum, std::vector<nn::Tensor>{x}, rows, cols);
    // #T6：求值失败即 FAIL 再 return——裸 return 会让后续对拍断言被跳过、
    // 测试仍记绿。
    CHECK(static_cast<bool>(fused_r), "融合 kernel 求值成功");
    CHECK(static_cast<bool>(y_ref), "归约参考求值成功");
    if (!fused_r || !y_ref) return;
    auto y_full = eng.eval_expr(add, std::vector<nn::Tensor>{*y_ref}, rows, cols);
    CHECK(static_cast<bool>(y_full), "逐节点广播参考求值成功");
    if (!y_full) return;
    CHECK(max_abs_diff(matrix_of(eng, *fused_r), matrix_of(eng, *y_full)) < 1e-6f,
          "融合 kernel == 逐节点（归约 → 广播）");
    std::printf("[OK] test_reduce_broadcast_fuse (kernels=%zu)\n", kernels.size());
}

// ── 测试 5b：归约轴不匹配 → 仍作为融合边界（兜底独立派发） ──────────────
void test_reduce_axis_mismatch_boundary()
{
    auto eng_owner = nn::create_cpu_engine();
    nn::ComputeEngine& eng = *eng_owner;
    const std::size_t rows = 3, cols = 4;
    nn::ExprGraph g;
    nn::Tensor x = make_tensor(eng, rows, cols, 0.0f, 0.1f);

    // 列归约（输出 (1,cols)）却被 RowBroadcast 消费 → 轴不匹配，拒绝拼接
    const int nA = g.add_node(make_col_sum(), std::vector<nn::Tensor>{x}, rows, cols, true);
    nn::Tensor yt = make_placeholder(eng, 1, cols, g.tag_of_node[static_cast<std::size_t>(nA)]);
    (void)g.add_node(make_row_broadcast_const(nn::ExprOp::Add, Scalar{1}),
                     std::vector<nn::Tensor>{yt}, rows, cols, false);

    auto kernels = fuse_expr_graph(g);
    CHECK(kernels.size() == 2, "归约轴不匹配时保持独立 kernel（走 JIT/单节点路径）");
    std::printf("[OK] test_reduce_axis_mismatch_boundary (kernels=%zu)\n", kernels.size());
}

// ── 测试 5c（P10-③）：fold 头 + 逐元素尾链融合 ─────────────────────────────
// A: s = fold_rowsum(x)（fold 输出网格 (rows,1)，顶层空）；B: y = s + 2.5
// （唯一消费者，Linear 引用 fold 输出）→ 拼成单 kernel：fold 段保留 + 顶层
// instrs = 尾链（尾链读 finalize 末 dst）。融合 spec 必须过 validate，数值
// == 逐节点（fold → 加法）参考；尾链 GLSL 生成（'T' 上下文）非空。
void test_fold_tail_fuse()
{
    auto eng_owner = nn::create_cpu_engine();
    nn::ComputeEngine& eng = *eng_owner;
    const std::size_t rows = 4;
    const std::uint32_t K = 16;
    nn::ExprGraph g;
    nn::Tensor x = make_tensor(eng, rows, K, -1.0f, 0.1f);
    const nn::ExprSpec fold = nn::expr::make_fold_rowsum(K);
    const nn::ExprSpec add = make_unary_const(nn::ExprOp::Add, Scalar{2.5});

    const int nA = g.add_node(fold, std::vector<nn::Tensor>{x}, rows, 1, false);
    nn::Tensor st = make_placeholder(eng, rows, 1, g.tag_of_node[static_cast<std::size_t>(nA)]);
    (void)g.add_node(add, std::vector<nn::Tensor>{st}, rows, 1, false);

    auto kernels = fuse_expr_graph(g);
    CHECK(kernels.size() == 1, "fold 头 + 消费链应融合为单 kernel");
    if (kernels.size() != 1) return;
    CHECK(kernels[0].members.size() == 2, "kernel 应包含 fold + 消费节点");
    CHECK(kernels[0].spec.fold.has_value(), "融合 spec 保留 fold 段");
    CHECK(kernels[0].spec.instrs.size() == 1, "顶层 instrs = 消费链尾链");
    if (kernels[0].spec.fold && !kernels[0].spec.instrs.empty())
    {
        const nn::ExprOperand& a = kernels[0].spec.instrs[0].a;
        CHECK(a.kind == static_cast<uint8_t>(nn::ExprOperandKind::Reg) &&
                  a.idx == kernels[0].spec.fold->finalize.back().dst,
              "尾链读 fold finalize 末指令 dst");
    }
    auto v = nn::validate_expr_spec(kernels[0].spec, kernels[0].spec.views.size());
    CHECK(static_cast<bool>(v), "融合 fold+尾链 spec 通过 validate");
    if (!v) return;

    std::vector<nn::Tensor> kinputs;
    for (auto& in : kernels[0].inputs)
        kinputs.push_back(in.external);
    auto fused_r = eng.eval_expr(kernels[0].spec, kinputs, rows, 1);
    auto y_ref0 = eng.eval_expr(fold, std::vector<nn::Tensor>{x}, rows, 1);
    CHECK(static_cast<bool>(fused_r), "融合 fold+尾链求值成功");
    CHECK(static_cast<bool>(y_ref0), "fold 参考求值成功");
    if (!fused_r || !y_ref0) return;
    auto y_ref = eng.eval_expr(add, std::vector<nn::Tensor>{*y_ref0}, rows, 1);
    CHECK(static_cast<bool>(y_ref), "尾链参考求值成功");
    if (!y_ref) return;
    CHECK(max_abs_diff(matrix_of(eng, *fused_r), matrix_of(eng, *y_ref)) < 1e-6f,
          "融合 fold+尾链 == 逐节点（fold → 加法）");

    // 尾链 GLSL 生成（v1 分支：vec_state_len=0）非空 = 'T' 上下文可发射
    auto gl = nn::emitter_registry::make("glsl");
    if (gl)
    {
        const std::string src = gl->generate("fold_tail", kernels[0].spec);
        CHECK(!src.empty(), "fold+尾链 GLSL 生成成功（v1）");
        CHECK(src.find("fold tail") != std::string::npos, "生成源包含尾链段");
    }
    std::printf("[OK] test_fold_tail_fuse (kernels=%zu)\n", kernels.size());
}

// ── 测试 5d（P10-③ 边界）：B 自身是 fold 头 → 不拼接 ─────────────────────
// fold 必须位于链首（其 finalize 即自带尾链）；前序链的消费者是 fold 节点时
// 保持独立 kernel。
void test_fold_as_chain_member_rejected()
{
    auto eng_owner = nn::create_cpu_engine();
    nn::ComputeEngine& eng = *eng_owner;
    const std::size_t rows = 3, cols = 4;
    const std::uint32_t K = 4;
    nn::ExprGraph g;
    nn::Tensor x = make_tensor(eng, rows, cols, 0.0f, 0.1f);

    const int nA = g.add_node(make_unary_const(nn::ExprOp::Add, Scalar{1}),
                              std::vector<nn::Tensor>{x}, rows, cols, false);
    nn::Tensor yt = make_placeholder(eng, rows, cols, g.tag_of_node[static_cast<std::size_t>(nA)]);
    // fold B：输入 = A 输出（cols=K 网格），输出 (rows,1)
    (void)g.add_node(nn::expr::make_fold_rowsum(K), std::vector<nn::Tensor>{yt}, rows, 1, false);

    auto kernels = fuse_expr_graph(g);
    CHECK(kernels.size() == 2, "fold 节点作链成员（B）时保持独立 kernel");
    std::printf("[OK] test_fold_as_chain_member_rejected (kernels=%zu)\n", kernels.size());
}

// ── 测试 15b（P10-③ v2）：双域 fold + 尾链（'T' 上下文 / dd 步进循环）──────
// v2 = mm 段 + vecacc（make_fold_mm_chain，输出网格 (rows, veclen)）。尾链两种
// 读法：① 读 finalize 输出寄存器（融合消费链的真实形态）；② 直读 VecState
//（向量域操作数）。验收：validate 通过、GLSL 走 v2 分支且含尾链段、CPU 数值
// == 基线 × 2.5。GPU 执行语义由 expr_fuse_test 的 run_fold_v2_tail 覆盖。
void test_fold_v2_tail()
{
    auto eng_owner = nn::create_cpu_engine();
    nn::ComputeEngine& eng = *eng_owner;
    const std::size_t rows = 4;
    const std::uint32_t fold_k = 16, mm_k = 16, veclen = 8;
    nn::Tensor a0 = make_tensor(eng, rows, mm_k, 0.5f, 0.0f);   // A ≡ 0.5
    nn::Tensor b0 = make_tensor(eng, mm_k, fold_k, 1.0f, 0.0f); // B ≡ 1
    nn::Tensor vv = make_tensor(eng, fold_k, veclen, 1.0f, 0.0f); // V ≡ 1
    const std::vector<nn::Tensor> ins{a0, b0, vv};

    const nn::ExprSpec base = make_fold_mm_chain(fold_k, mm_k, veclen);
    auto mk_tail = [&](bool via_vecstate)
    {
        nn::ExprSpec s = base;
        s.consts.push_back(Scalar{2.5f});
        nn::ExprInstr ti;
        ti.op = static_cast<uint8_t>(nn::ExprOp::Mul);
        ti.dst = 6;
        ti.a = via_vecstate ? nn::expr::vec_state(0) : nn::expr::reg(5);
        ti.b = nn::expr::cst(1);
        s.instrs = {ti};
        s.num_regs = 7;
        return s;
    };
    const nn::ExprSpec tail_reg = mk_tail(false);
    const nn::ExprSpec tail_vec = mk_tail(true);

    CHECK(static_cast<bool>(nn::validate_expr_spec(tail_reg, 3)),
          "v2 尾链（读 finalize dst）validate 通过");
    CHECK(static_cast<bool>(nn::validate_expr_spec(tail_vec, 3)),
          "v2 尾链（直读 VecState）validate 通过");

    auto base_out = eng.eval_expr(base, ins, rows, veclen);
    auto tail_out = eng.eval_expr(tail_reg, ins, rows, veclen);
    auto vec_out = eng.eval_expr(tail_vec, ins, rows, veclen);
    CHECK(static_cast<bool>(base_out), "v2 基线求值成功");
    CHECK(static_cast<bool>(tail_out), "v2 尾链（reg）求值成功");
    CHECK(static_cast<bool>(vec_out), "v2 尾链（VecState）求值成功");
    if (base_out && tail_out && vec_out)
    {
        const auto bm = matrix_of(eng, *base_out);
        const auto tm = matrix_of(eng, *tail_out);
        const auto vm = matrix_of(eng, *vec_out);
        const auto bs = bm.span(), tss = tm.span(), vs = vm.span();
        bool ok = bs.size() == tss.size() && bs.size() == vs.size();
        Scalar err = 0;
        for (std::size_t i = 0; ok && i < bs.size(); ++i)
        {
            const Scalar ref = bs[i] * Scalar{2.5f};
            err = std::fmax(err, std::fabs(tss[i] - ref) / std::fmax(Scalar{1}, std::fabs(ref)));
            err = std::fmax(err, std::fabs(vs[i] - ref) / std::fmax(Scalar{1}, std::fabs(ref)));
        }
        CHECK(ok && err < 1e-5f, "v2 尾链数值 == 基线 × 2.5（reg 与 VecState 两读法）");
    }

    auto gl = nn::emitter_registry::make("glsl");
    if (gl)
    {
        const std::string src = gl->generate("fold_v2_tail", tail_reg);
        CHECK(!src.empty(), "v2 fold+尾链 GLSL 生成成功");
        CHECK(src.find("fold tail") != std::string::npos, "v2 生成源包含尾链段");
        CHECK(src.find("fold v2") != std::string::npos, "走 v2 双域分支");
        if (std::getenv("NN_DUMP_FOLD_TAIL") != nullptr)
            std::printf("===== fold v2 tail GLSL =====\n%s\n===== end =====\n", src.c_str());
    }
    std::printf("[OK] test_fold_v2_tail\n");
}

// ── 测试 6：上限压力（拼接后超 EXPR_MAX_INSTRS 则放弃融合） ───────────
// 用 abs 链（abs 不可代数化简、不可常量折叠）构造可控指令数
nn::ExprSpec make_abs_chain(std::uint8_t n)
{
    nn::ExprSpec s;
    s.views.push_back(nn::expr::linear());
    nn::ExprOperand prev = nn::expr::input(0);
    for (std::uint8_t i = 0; i < n; ++i)
    {
        nn::ExprInstr in;
        in.op = static_cast<std::uint8_t>(nn::ExprOp::Abs);
        in.dst = i;
        in.a = prev;
        s.instrs.push_back(in);
        prev = nn::expr::reg(i);
    }
    s.num_regs = n;
    return s;
}

void test_limit_pressure()
{
    auto eng_owner = nn::create_cpu_engine();
    nn::ComputeEngine& eng = *eng_owner;
    const std::size_t rows = 1, cols = 4;
    nn::ExprGraph g;
    nn::Tensor x = make_tensor(eng, rows, cols, 0.0f, 0.1f);

    // A：40 条 abs 指令；B：1 条 → 拼接后 41 ≤ 64 → 融合
    const int nA = g.add_node(make_abs_chain(40), std::vector<nn::Tensor>{x}, rows, cols, false);
    nn::Tensor yt = make_placeholder(eng, rows, cols, g.tag_of_node[static_cast<std::size_t>(nA)]);
    (void)g.add_node(make_unary_const(nn::ExprOp::Add, Scalar{1}), std::vector<nn::Tensor>{yt},
                     rows, cols, false);

    auto kernels = fuse_expr_graph(g);
    CHECK(kernels.size() == 1, "40+1 条指令应在 EXPR_MAX_INSTRS=64 内 → 融合");

    // A：60 条 abs；B：5 条 abs 链 → 拼接后 65 > 64 → 不融合
    nn::ExprGraph g2;
    nn::Tensor x2 = make_tensor(eng, rows, cols, 0.0f, 0.1f);
    const int nC = g2.add_node(make_abs_chain(60), std::vector<nn::Tensor>{x2}, rows, cols, false);
    nn::Tensor yt2 = make_placeholder(eng, rows, cols, g.tag_of_node[static_cast<std::size_t>(nC)]);
    // B：5 条 abs 链（消费 A 输出）
    nn::ExprSpec sb = make_abs_chain(5);
    (void)g2.add_node(sb, std::vector<nn::Tensor>{yt2}, rows, cols, false);

    auto kernels2 = fuse_expr_graph(g2);
    CHECK(kernels2.size() == 2, "60+5 条指令超 EXPR_MAX_INSTRS → 保守放弃融合");
    std::printf("[OK] test_limit_pressure\n");
}

// ── 测试 7：确定性（同一图多次融合结果一致） ───────────────────────────
void test_determinism()
{
    auto eng_owner = nn::create_cpu_engine();
    nn::ComputeEngine& eng = *eng_owner;
    const std::size_t rows = 2, cols = 4;
    nn::Tensor x = make_tensor(eng, rows, cols, 0.0f, 0.1f);

    auto build = [&](nn::ExprGraph& g)
    {
        const int nA = g.add_node(make_unary_const(nn::ExprOp::Mul, Scalar{2}),
                                  std::vector<nn::Tensor>{x}, rows, cols, false);
        nn::Tensor yt =
            make_placeholder(eng, rows, cols, g.tag_of_node[static_cast<std::size_t>(nA)]);
        (void)g.add_node(make_unary_const(nn::ExprOp::Add, Scalar{1}), std::vector<nn::Tensor>{yt},
                         rows, cols, false);
    };

    nn::ExprGraph g1, g2;
    build(g1);
    build(g2);
    auto k1 = fuse_expr_graph(g1);
    auto k2 = fuse_expr_graph(g2);
    CHECK(k1.size() == k2.size(), "确定性：kernel 数一致");
    if (k1.size() != k2.size()) return;
    const std::string key1 = nn::expr_spec_key(k1[0].spec);
    const std::string key2 = nn::expr_spec_key(k2[0].spec);
    CHECK(key1 == key2, "确定性：融合 kernel 的 key 完全一致");
    CHECK(nn::expr_spec_equal(k1[0].spec, k2[0].spec), "确定性：spec 完全一致");
    std::printf("[OK] test_determinism\n");
}

// ── 测试 8：录制图 add_node 依赖识别 ───────────────────────────────────
void test_dep_recognition()
{
    auto eng_owner = nn::create_cpu_engine();
    nn::ComputeEngine& eng = *eng_owner;
    const std::size_t rows = 2, cols = 4;
    nn::ExprGraph g;
    nn::Tensor x = make_tensor(eng, rows, cols, 0.0f, 0.1f);

    const int nA = g.add_node(make_unary_const(nn::ExprOp::Mul, Scalar{2}),
                              std::vector<nn::Tensor>{x}, rows, cols, false);
    // 占位带 tag → 依赖识别
    nn::Tensor yt = make_placeholder(eng, rows, cols, g.tag_of_node[static_cast<std::size_t>(nA)]);
    const int nB = g.add_node(make_unary_const(nn::ExprOp::Add, Scalar{1}),
                              std::vector<nn::Tensor>{yt}, rows, cols, false);
    CHECK(g.nodes[static_cast<std::size_t>(nB)].dep_of_input[0] == nA,
          "B 的输入 0 应识别为依赖节点 A");
    // 普通输入 → 外部
    const int nC = g.add_node(make_unary_const(nn::ExprOp::Sub, Scalar{1}),
                              std::vector<nn::Tensor>{x}, rows, cols, false);
    CHECK(g.nodes[static_cast<std::size_t>(nC)].dep_of_input[0] == -1, "C 的输入 0 应为外部输入");
    // 未知 tag → 外部
    nn::Tensor fake = make_placeholder(eng, rows, cols, 999);
    const int nD = g.add_node(make_unary_const(nn::ExprOp::Abs, Scalar{0}),
                              std::vector<nn::Tensor>{fake}, rows, cols, false);
    CHECK(g.nodes[static_cast<std::size_t>(nD)].dep_of_input[0] == -1, "未知 tag 视为外部输入");
    std::printf("[OK] test_dep_recognition\n");
}

// ── 测试 9：融合 kernel 输入顺序确定 ───────────────────────────────────
// A: y = x1 * x2（两个输入）；B: z = y + x3 → 融合后 inputs = [x1, x2, x3]
void test_input_order()
{
    auto eng_owner = nn::create_cpu_engine();
    nn::ComputeEngine& eng = *eng_owner;
    const std::size_t rows = 2, cols = 4;
    nn::ExprGraph g;
    nn::Tensor x1 = make_tensor(eng, rows, cols, 0.0f, 0.1f);
    nn::Tensor x2 = make_tensor(eng, rows, cols, 1.0f, 0.1f);
    nn::Tensor x3 = make_tensor(eng, rows, cols, 2.0f, 0.1f);

    // A: y = x1 * x2
    const int nA = g.add_node(make_binary(nn::ExprOp::Mul), std::vector<nn::Tensor>{x1, x2}, rows,
                              cols, false);
    // B: z = y + x3（输入 0 = y 依赖 A，输入 1 = x3 外部）
    nn::Tensor yt = make_placeholder(eng, rows, cols, g.tag_of_node[static_cast<std::size_t>(nA)]);
    nn::ExprSpec sb;
    sb.views.push_back(nn::expr::linear());
    sb.views.push_back(nn::expr::linear());
    nn::ExprInstr in;
    in.op = static_cast<std::uint8_t>(nn::ExprOp::Add);
    in.dst = 0;
    in.a = nn::expr::input(0);
    in.b = nn::expr::input(1);
    sb.instrs.push_back(in);
    sb.num_regs = 1;
    (void)g.add_node(sb, std::vector<nn::Tensor>{yt, x3}, rows, cols, false);

    auto kernels = fuse_expr_graph(g);
    CHECK(kernels.size() == 1, "A→B 链融合为单 kernel");
    if (kernels.size() != 1) return;
    CHECK(kernels[0].inputs.size() == 3, "融合 kernel 应有 3 个外部输入");
    if (kernels[0].inputs.size() != 3) return;
    // 顺序：A 的输入（x1, x2）在前，B 的非链输入（x3）在后
    const nn::Matrix m0 = matrix_of(eng, kernels[0].inputs[0].external);
    const nn::Matrix m1 = matrix_of(eng, kernels[0].inputs[1].external);
    const nn::Matrix m2 = matrix_of(eng, kernels[0].inputs[2].external);
    CHECK(m0.span()[0] == 0.0f, "输入 0 = x1");
    CHECK(m1.span()[0] == 1.0f, "输入 1 = x2");
    CHECK(m2.span()[0] == 2.0f, "输入 2 = x3");

    // 数值验证：z = x1*x2 + x3（P2 多输出：主输出 = B，extras = A 写穿）
    std::vector<nn::Tensor> kinputs;
    for (auto& in : kernels[0].inputs)
        kinputs.push_back(in.external);
    std::vector<nn::Tensor> kouts;
    kouts.push_back(eng.create_tensor(rows, cols));
    kouts.push_back(eng.create_tensor(rows, cols));
    auto mul_r = eng.eval_expr_multi_into(kernels[0].spec, kinputs, rows, cols, kouts);
    CHECK(static_cast<bool>(mul_r), "融合 kernel 多输出求值成功");
    if (!mul_r) return;
    const nn::Matrix mf = matrix_of(eng, kouts[0]);
    bool ok = true;
    for (std::size_t i = 0; i < rows * cols; ++i)
    {
        const Scalar expect = m0.span()[i] * m1.span()[i] + m2.span()[i];
        if (std::fabs(mf.span()[i] - expect) > 1e-5f) ok = false;
    }
    CHECK(ok, "融合结果 = x1*x2 + x3");
    // extras 写穿 = A 的输出（x1*x2）
    const nn::Matrix ma = matrix_of(eng, kouts[1]);
    bool ok2 = true;
    for (std::size_t i = 0; i < rows * cols; ++i)
    {
        const Scalar expect = m0.span()[i] * m1.span()[i];
        if (std::fabs(ma.span()[i] - expect) > 1e-5f) ok2 = false;
    }
    CHECK(ok2, "extras 写穿 = x1*x2");
    std::printf("[OK] test_input_order\n");
}

// ── 测试 10：matmul 参与图融合（S5：matmul 节点 + 尾逐元素节点拼接） ───
//   A: y = matmul(A0,B0) + bias（matmul 段 + Add 尾链）
//   B: z = max(y, 0)（relu）消费 A → 融合为单 kernel：spec.matmul 保留，
//      instrs = A 尾链 + B 指令（matmul 输出经"虚拟寄存器"流入 relu）。
void test_matmul_chain_fuse()
{
    auto eng_owner = nn::create_cpu_engine();
    nn::ComputeEngine& eng = *eng_owner;
    const std::size_t M = 3, K = 4, N = 2;
    nn::ExprGraph g;
    nn::Tensor A0 = make_tensor(eng, M, K, 0.5f, 0.03f);
    nn::Tensor B0 = make_tensor(eng, K, N, -0.2f, 0.04f);
    nn::Tensor bias = make_tensor(eng, M, N, 0.1f, 0.01f);

    // A：matmul(A0,B0) + bias
    nn::ExprSpec sa;
    sa.views = {nn::expr::linear(), nn::expr::linear(), nn::expr::linear()};
    sa.num_regs = 1;
    sa.matmul = nn::MatmulSpec{0, 1, 0, 0, static_cast<std::uint32_t>(K), 1};
    sa.instrs.push_back({static_cast<std::uint8_t>(nn::ExprOp::Add),
                         0,
                         nn::expr::matmul_op(),
                         nn::expr::input(2),
                         {}});
    const int nA = g.add_node(sa, std::vector<nn::Tensor>{A0, B0, bias}, M, N, false);
    nn::Tensor yt = make_placeholder(eng, M, N, g.tag_of_node[static_cast<std::size_t>(nA)]);
    // B：relu（max(y, 0)）消费 A
    const int nB = g.add_node(make_unary_const(nn::ExprOp::Max, Scalar{0}),
                              std::vector<nn::Tensor>{yt}, M, N, false);
    (void)nB;

    auto kernels = fuse_expr_graph(g);
    CHECK(kernels.size() == 1, "matmul+bias → relu 应融合为单个 kernel");
    if (kernels.size() != 1) return;
    CHECK(kernels[0].members.size() == 2, "kernel 应包含 2 个节点");
    CHECK(kernels[0].spec.matmul.has_value(), "融合 spec 应保留 matmul 段");
    CHECK(kernels[0].spec.matmul->a_input == 0 && kernels[0].spec.matmul->b_input == 1,
          "matmul A/B 输入槽保留（views 前缀顺序不变）");
    CHECK(kernels[0].inputs.size() == 3, "融合 kernel 应有 3 个外部输入（A0/B0/bias）");
    if (!kernels[0].spec.matmul) return;

    // 融合 kernel 求值 vs 逐节点参考
    std::vector<nn::Tensor> kinputs;
    for (auto& in : kernels[0].inputs)
        kinputs.push_back(in.external);
    auto fused = eng.eval_expr(kernels[0].spec, kinputs, M, N);
    CHECK(static_cast<bool>(fused), "融合 kernel 求值成功");
    if (!fused) return;

    auto a_res = eng.eval_expr(sa, std::vector<nn::Tensor>{A0, B0, bias}, M, N);
    auto b_res = eng.eval_expr(make_unary_const(nn::ExprOp::Max, Scalar{0}),
                               std::vector<nn::Tensor>{*a_res}, M, N);
    CHECK(static_cast<bool>(b_res), "参考求值成功");
    if (!b_res) return;
    const Scalar err = max_abs_diff(matrix_of(eng, *fused), matrix_of(eng, *b_res));
    CHECK(err < 1e-6f, "matmul 图融合结果与逐节点一致");
    std::printf("[OK] test_matmul_chain_fuse\n");
}

// ── 测试 11：纯 matmul 节点 + 消费节点拼接（S5：tail 引用 → Matmul 操作数）
//   A: y = matmul(A0,B0)（纯 matmul，空指令链）
//   B: z = y + bias 消费 A → 融合为单 kernel（B 对 A 的引用替换为 Matmul 操作数）
void test_matmul_pure_chain_fuse()
{
    auto eng_owner = nn::create_cpu_engine();
    nn::ComputeEngine& eng = *eng_owner;
    const std::size_t M = 2, K = 3, N = 4;
    nn::ExprGraph g;
    nn::Tensor A0 = make_tensor(eng, M, K, 0.4f, 0.05f);
    nn::Tensor B0 = make_tensor(eng, K, N, -0.1f, 0.02f);
    nn::Tensor bias = make_tensor(eng, M, N, 0.2f, 0.01f);

    // A：纯 matmul（空指令链）
    nn::ExprSpec sa;
    sa.views = {nn::expr::linear(), nn::expr::linear()};
    sa.num_regs = 0;
    sa.matmul = nn::MatmulSpec{0, 1, 0, 0, static_cast<std::uint32_t>(K), 1};
    const int nA = g.add_node(sa, std::vector<nn::Tensor>{A0, B0}, M, N, false);
    nn::Tensor yt = make_placeholder(eng, M, N, g.tag_of_node[static_cast<std::size_t>(nA)]);
    // B：Add(y, bias)
    nn::ExprSpec sb;
    sb.views = {nn::expr::linear(), nn::expr::linear()};
    sb.num_regs = 1;
    sb.instrs.push_back({static_cast<std::uint8_t>(nn::ExprOp::Add),
                         0,
                         nn::expr::input(0),
                         nn::expr::input(1),
                         {}});
    const int nB = g.add_node(sb, std::vector<nn::Tensor>{yt, bias}, M, N, false);
    (void)nB;

    auto kernels = fuse_expr_graph(g);
    CHECK(kernels.size() == 1, "纯 matmul → Add 应融合为单个 kernel");
    if (kernels.size() != 1) return;
    CHECK(kernels[0].spec.matmul.has_value(), "融合 spec 应保留 matmul 段");
    CHECK(kernels[0].spec.instrs.size() == 1, "融合指令 = B 的 Add（A 为空链）");
    CHECK(kernels[0].spec.instrs[0].a.kind ==
              static_cast<std::uint8_t>(nn::ExprOperandKind::Matmul),
          "B 对 A 的引用应替换为 Matmul 操作数");
    CHECK(kernels[0].inputs.size() == 3, "融合 kernel 应有 3 个外部输入（A0/B0/bias）");
    if (!kernels[0].spec.matmul) return;

    std::vector<nn::Tensor> kinputs;
    for (auto& in : kernels[0].inputs)
        kinputs.push_back(in.external);
    auto fused = eng.eval_expr(kernels[0].spec, kinputs, M, N);
    CHECK(static_cast<bool>(fused), "融合 kernel 求值成功");
    if (!fused) return;

    // 参考：matmul + bias（逐元素经 DSL；旧 elementwise_binary 已随 eager 收敛删除）
    auto a_res = eng.matmul(A0, B0, false, false);
    CHECK(static_cast<bool>(a_res), "参考 matmul 成功");
    if (!a_res) return;
    auto ref = nn::dsl::compute(eng, nn::dsl::leaf(*a_res) + nn::dsl::leaf(bias), M, N);
    CHECK(static_cast<bool>(ref), "参考加法成功");
    if (!ref) return;
    const Scalar err = max_abs_diff(matrix_of(eng, *fused), matrix_of(eng, *ref));
    CHECK(err < 1e-6f, "纯 matmul 图融合结果与参考一致");
    std::printf("[OK] test_matmul_pure_chain_fuse\n");
}

// ── 测试 11b（P10-①）：多消费者 matmul 头进 P2 分量（写穿其输出寄存器）──
//   A: y = matmul(A0,B0) + bias（带尾链的 matmul 头）；B: y1 = y+1、C: y2 = y-1
//   同时消费 A → 三节点融合为单 kernel（members=A,B,C；tail=C；
//   extras=[A.out, B.out] 写穿），A 的输出寄存器经 extras 物化供组外读取。
//   单消费者控制由既有 test_matmul_chain_fuse 锁定（P1 语义、extras 空不变）。
void test_matmul_multi_consumer_fuse()
{
    auto eng_owner = nn::create_cpu_engine();
    nn::ComputeEngine& eng = *eng_owner;
    const std::size_t M = 3, K = 4, N = 2;
    nn::ExprGraph g;
    nn::Tensor A0 = make_tensor(eng, M, K, 0.5f, 0.03f);
    nn::Tensor B0 = make_tensor(eng, K, N, -0.2f, 0.04f);
    nn::Tensor bias = make_tensor(eng, M, N, 0.1f, 0.01f);

    nn::ExprSpec sa;
    sa.views = {nn::expr::linear(), nn::expr::linear(), nn::expr::linear()};
    sa.num_regs = 1;
    sa.matmul = nn::MatmulSpec{0, 1, 0, 0, static_cast<std::uint32_t>(K), 1};
    sa.instrs.push_back({static_cast<std::uint8_t>(nn::ExprOp::Add),
                         0,
                         nn::expr::matmul_op(),
                         nn::expr::input(2),
                         {}});
    const int nA = g.add_node(sa, std::vector<nn::Tensor>{A0, B0, bias}, M, N, false);
    nn::Tensor yt = make_placeholder(eng, M, N, g.tag_of_node[static_cast<std::size_t>(nA)]);
    // B: y1 = y + 1；C: y2 = y − 1（双消费者，均 Linear 引用 y）
    (void)g.add_node(make_unary_const(nn::ExprOp::Add, Scalar{1}),
                     std::vector<nn::Tensor>{yt}, M, N, false);
    (void)g.add_node(make_unary_const(nn::ExprOp::Sub, Scalar{1}),
                     std::vector<nn::Tensor>{yt}, M, N, false);

    auto kernels = fuse_expr_graph(g);
    // #P10-①：expr_spec.hpp 契约守卫已放开（extras 支持 matmul spec）——
    //   多消费者带尾链的 matmul 头必须融合为**单 kernel**（members=3、
    //   extras=2），下方验收体直调执行对拍逐节点参考。
    //   历史"守卫下退化为 kernels=2 也算 OK"的分支已删除：守卫放开后保留
    //   它会把融合回归（重新分裂）静默记绿 = 假绿陷阱。
    if (kernels.size() != 1)
    {
        std::printf("[diag] kernels=%zu:", kernels.size());
        for (const auto& k : kernels)
            std::printf(" {members=%zu matmul=%d extras=%zu tail=%d}", k.members.size(),
                        k.spec.matmul ? 1 : 0, k.spec.extras.size(), k.tail);
        std::printf("\n");
    }
    CHECK(kernels.size() == 1, "P10-①: 多消费者 matmul 头应融合为单 kernel");
    if (kernels.size() != 1) return;
    CHECK(kernels[0].members.size() == 3, "kernel 应含 matmul 头 + 2 消费者");
    CHECK(kernels[0].spec.matmul.has_value(), "融合 spec 保留 matmul 段");
    CHECK(kernels[0].spec.extras.size() == 2, "extras = [matmul头.out, B.out]（写穿）");
    CHECK(kernels[0].tail == 2, "tail = 第三节点");
    if (!kernels[0].spec.matmul) return;
    // ── 单 kernel 三输出直调执行（守卫放开后的验收体）──
    std::vector<nn::Tensor> kinputs;
    for (auto& in : kernels[0].inputs) kinputs.push_back(in.external);
    std::vector<nn::Tensor> kouts;
    for (int k = 0; k < 3; ++k) kouts.push_back(eng.create_tensor(M, N));
    auto fused_r = eng.eval_expr_multi_into(kernels[0].spec, kinputs, M, N, kouts);
    CHECK(static_cast<bool>(fused_r), "多输出（matmul+extras）求值成功");
    auto a_ref = eng.eval_expr(sa, std::vector<nn::Tensor>{A0, B0, bias}, M, N);
    CHECK(static_cast<bool>(a_ref), "逐节点参考 A 求值成功");
    if (!fused_r || !a_ref) return;
    auto b_ref = eng.eval_expr(make_unary_const(nn::ExprOp::Add, Scalar{1}),
                               std::vector<nn::Tensor>{*a_ref}, M, N);
    auto c_ref = eng.eval_expr(make_unary_const(nn::ExprOp::Sub, Scalar{1}),
                               std::vector<nn::Tensor>{*a_ref}, M, N);
    CHECK(static_cast<bool>(b_ref), "逐节点参考 B 求值成功");
    CHECK(static_cast<bool>(c_ref), "逐节点参考 C 求值成功");
    if (!b_ref || !c_ref) return;
    CHECK(max_abs_diff(matrix_of(eng, kouts[0]), matrix_of(eng, *c_ref)) < 1e-6f,
          "tail(C) 与逐节点一致");
    CHECK(max_abs_diff(matrix_of(eng, kouts[1]), matrix_of(eng, *a_ref)) < 1e-6f,
          "extras[0]=matmul 头输出写穿与逐节点一致");
    CHECK(max_abs_diff(matrix_of(eng, kouts[2]), matrix_of(eng, *b_ref)) < 1e-6f,
          "extras[1]=B 写穿与逐节点一致");
    std::printf("[OK] test_matmul_multi_consumer_fuse (kernels=%zu)\n", kernels.size());
}

// ── 测试 12（#M1）：图计划缓存命中时，运行期值必须按**当前图**重推导 ──
// `graph_cache_key` 只覆盖结构（节点结构哈希 + 形状 + 精度 + 依赖拓扑），而
// matmul 的 k/batch/trans、`fold->k`、`views[].param/param2` 都是**运行期值**
// （`expr_spec_struct_hash`/`expr_spec_key` 明确排除）。命中计划若沿用首次
// 建计划时的值：同结构不同 K 会静默用错 k（K 更大时 GPU 越界读）。
//
// 本用例完全走引擎真实协议（graph_plan_cache + plan_from_kernel +
// instantiate_plan_into），在同一结构、不同运行期值的图上验证：
//   · 只有 k / view param 不同 → 必须同 key（真的命中缓存），值须是当前图的；
//   · 只有 const 值不同 → 池布局依赖值本身（dedupe/renumber），故 key 必须不同；
// 每步变化的 rparams 仍按成员序重填（既有行为，此处一并回归）。
void test_plan_cache_runtime_values()
{
    auto eng_owner = nn::create_cpu_engine();
    nn::ComputeEngine& eng = *eng_owner;
    auto& cache = nn::fused::graph_plan_cache();
    cache.clear();

    const std::size_t M = 4, N = 2;
    struct Case
    {
        std::uint32_t k;        // matmul 收缩维（运行期值，不进结构 key）
        std::uint8_t transB;    // 顶层 matmul 转置（运行期值，不进结构 key）
        std::uint32_t row_mod;  // RowMod 的 param（运行期形状数据，不进结构 key）
        Scalar c;               // const 池值（进图 key：池布局依赖值）
    };

    // node0 = matmul(A0,B0) + bias（matmul 段 + 尾链）；node1 = RowMod(x) + c。
    // 两节点互不依赖 → 两个 kernel，分别覆盖 matmul 段与 views 段的刷新。
    const auto build = [&](const Case& cs)
    {
        auto g = std::make_unique<nn::ExprGraph>();
        nn::Tensor A0 = make_tensor(eng, M, cs.k, 0.5f, 0.03f);
        nn::Tensor B0 = make_tensor(eng, cs.transB ? N : cs.k, cs.transB ? cs.k : N, -0.2f, 0.04f);
        nn::Tensor bias = make_tensor(eng, M, N, 0.1f, 0.01f);
        nn::ExprSpec sa;
        sa.views = {nn::expr::linear(), nn::expr::linear(), nn::expr::linear()};
        sa.num_regs = 1;
        sa.matmul = nn::MatmulSpec{0, 1, 0, cs.transB, cs.k, 1};
        sa.instrs.push_back({static_cast<std::uint8_t>(nn::ExprOp::Add),
                             0,
                             nn::expr::matmul_op(),
                             nn::expr::input(2),
                             {}});
        g->add_node(sa, std::vector<nn::Tensor>{A0, B0, bias}, M, N, false);

        nn::Tensor x = make_tensor(eng, cs.row_mod, N, 0.0f, 1.0f);  // RowMod 要求输入 (param, cols)
        nn::ExprSpec sb;
        nn::ExprView v;
        v.kind = static_cast<std::uint8_t>(nn::ExprViewKind::RowMod);
        v.param = cs.row_mod;
        sb.views.push_back(v);
        sb.consts.push_back(cs.c);
        sb.num_regs = 1;
        sb.instrs.push_back({static_cast<std::uint8_t>(nn::ExprOp::Add),
                             0,
                             nn::expr::input(0),
                             nn::expr::cst(0),
                             {}});
        g->add_node(sb, std::vector<nn::Tensor>{x}, M, N, false);
        return g;
    };

    // 引擎缓存协议（与 GpuEngine::execute_fused_graph 同构）
    int hits = 0;
    const auto run_cached = [&](nn::ExprGraph& g) -> std::vector<nn::FusedKernel>
    {
        const std::uint64_t key = nn::graph_cache_key(g);
        std::vector<nn::FusedKernel> kernels;
        const auto it = cache.find(key);
        if (it != cache.end())
        {
            ++hits;
            kernels.resize(it->second.size());
            for (std::size_t i = 0; i < it->second.size(); ++i)
                nn::instantiate_plan_into(it->second[i], g, kernels[i]);
        }
        else
        {
            kernels = nn::fuse_expr_graph(g);
            std::vector<nn::FusedKernelPlan> plans;
            plans.reserve(kernels.size());
            for (const auto& k : kernels) plans.push_back(nn::plan_from_kernel(k));
            for (std::size_t i = 0; i < kernels.size(); ++i)
                nn::instantiate_plan_into(plans[i], g, kernels[i]);
            cache.emplace(key, std::move(plans));
        }
        return kernels;
    };

    // ① 基线：k=2 / transB=0 / row_mod=2 / c=1
    auto g1 = build(Case{2, 0, 2, Scalar{1}});
    const auto k1 = run_cached(*g1);
    CHECK(k1.size() == 2, "#M1: 图应产出 2 个 kernel");
    if (k1.size() != 2) return;
    CHECK(k1[0].spec.matmul && k1[0].spec.matmul->k == 2, "基线 matmul k");
    CHECK(k1[1].spec.views.size() == 1 && k1[1].spec.views[0].param == 2, "基线 view param");

    // ② 同结构、不同 k / view param（matmul k 与 RowMod param 不进结构 key）
    auto g2 = build(Case{3, 0, 4, Scalar{1}});
    CHECK(nn::graph_cache_key(*g2) == nn::graph_cache_key(*g1),
          "#M1: 结构相同（仅运行期值不同）应同 key —— 用例确为命中路径");
    const auto k2 = run_cached(*g2);
    CHECK(hits == 1, "#M1: 第二张图必须命中缓存计划");
    if (k2.size() != 2) return;
    CHECK(k2[0].spec.matmul && k2[0].spec.matmul->k == 3,
          "#M1: 命中后 matmul k 必须按当前图重推导（否则静默用错 k/越界读）");
    CHECK(k2[1].spec.views.size() == 1 && k2[1].spec.views[0].param == 4,
          "#M1: 命中后 views[].param 必须按当前图重推导");

    // 数值锚：RowMod(4) + 1 —— 陈旧 param=2 会让第 2/3 行读到第 0/1 行
    {
        std::vector<nn::Tensor> ins{k2[1].inputs[0].external};
        nn::Tensor out = eng.create_tensor(M, N);
        CHECK(static_cast<bool>(eng.eval_expr_into(k2[1].spec, ins, M, N, out)),
              "#M1: 命中 kernel 求值成功");
        const nn::Matrix xm = matrix_of(eng, ins[0]);
        const nn::Matrix om = matrix_of(eng, out);
        bool ok = true;
        for (std::size_t r = 0; r < M; ++r)
            for (std::size_t c = 0; c < N; ++c)
            {
                const Scalar expect = xm.span()[(r % 4) * N + c] + Scalar{1};
                if (std::fabs(om.span()[r * N + c] - expect) > 1e-5f) ok = false;
            }
        CHECK(ok, "#M1: 命中 kernel 的数值须按当前图 view param 计算");
    }

    // ③ 同结构、不同 transB（顶层 matmul 转置也不进结构 key）
    auto g3 = build(Case{2, 1, 2, Scalar{1}});
    CHECK(nn::graph_cache_key(*g3) == nn::graph_cache_key(*g1), "#M1: transB 不进结构 key");
    const auto k3 = run_cached(*g3);
    if (k3.size() == 2)
    {
        CHECK(k3[0].spec.matmul && k3[0].spec.matmul->transB == 1,
              "#M1: 命中后 matmul transB 必须按当前图重推导");
        CHECK(k3[0].canon.matmul && k3[0].canon.matmul->transB == 1,
              "#M1: canonical 形态（批量组键用）也必须同源刷新");
    }

    // ④ 同结构、不同 const 值：池布局经 dedupe/renumber 后依赖值本身，
    //    无法按下标回填 → 必须走不同 key（miss），绝不能复用陈旧常量。
    auto g4 = build(Case{2, 0, 2, Scalar{5}});
    CHECK(nn::graph_cache_key(*g4) != nn::graph_cache_key(*g1),
          "#M1: const 值不同必须得到不同图 key（否则命中后静默用错常量）");
    const auto k4 = run_cached(*g4);
    if (k4.size() == 2)
    {
        CHECK(!k4[1].spec.consts.empty() && k4[1].spec.consts[0] == Scalar{5},
              "#M1: const 值须为当前图的值");
        std::vector<nn::Tensor> ins{k4[1].inputs[0].external};
        nn::Tensor out = eng.create_tensor(M, N);
        auto r4 = eng.eval_expr_into(k4[1].spec, ins, M, N, out);
        if (!r4) std::printf("  [diag] #M1 const 用例: %s\n", r4.error().message.c_str());
        CHECK(static_cast<bool>(r4), "#M1: const 用例求值成功");
        const nn::Matrix xm = matrix_of(eng, ins[0]);
        const nn::Matrix om = matrix_of(eng, out);
        bool ok = true;
        for (std::size_t r = 0; r < M; ++r)
            for (std::size_t c = 0; c < N; ++c)
            {
                const Scalar expect = xm.span()[(r % 2) * N + c] + Scalar{5};
                if (std::fabs(om.span()[r * N + c] - expect) > 1e-5f) ok = false;
            }
        CHECK(ok, "#M1: const 用例数值须用当前图的常量");
    }

    cache.clear();
    std::printf("[OK] test_plan_cache_runtime_values\n");
}

// ── 测试 15（#P4 残余）：直派 canonical+key 备忘录等价性 ───────────────────
// 锁住 canonicalize_with_key（expr_graph.hpp canon_key_memo 头注）的不变量：
//  1. 命中/未命中返回值与现算 canonicalize_expr_spec + expr_spec_key 逐字等价
//     （expr_spec_equal + key 字符串）——**不改任何 key 值**；
//  2. 结构同、rparams 值 / RowMod 运行期 param / matmul k·transB 不同 →
//     同一条目（命中），透传字段按当前 raw 刷新（陈旧值 = #M1 同类静默错值）；
//  3. 结构同、const 值不同 → 不同条目（常量池布局依赖值本身，绝不复用）；
//  4. fold spec 恒等返回、不进缓存（查找键成本 ≈ key 成本，见头注）。
void test_canon_key_memo()
{
    auto& memo = nn::fused::canon_key_memo();
    memo.clear();

    const auto fresh = [](const nn::ExprSpec& raw) -> nn::CanonKey
    {
        nn::CanonKey c;
        c.canon = nn::canonicalize_expr_spec(raw);
        c.key = nn::expr_spec_key(c.canon);
        return c;
    };
    const auto equiv = [](const nn::CanonKey& a, const nn::CanonKey& b)
    { return a.key == b.key && nn::expr_spec_equal(a.canon, b.canon); };

    // ── ① 未命中/命中均与现算等价；const 值不同必须分条目 ──
    const nn::ExprSpec a1 = make_unary_const(nn::ExprOp::Mul, Scalar{2});
    const nn::CanonKey a1_miss = nn::canonicalize_with_key(a1);
    CHECK(equiv(a1_miss, fresh(a1)), "#P4: 未命中返回值须与现算 canonicalize+key 等价");
    const nn::CanonKey a1_hit = nn::canonicalize_with_key(a1);
    CHECK(equiv(a1_hit, fresh(a1)), "#P4: 命中返回值须与现算 canonicalize+key 等价");
    CHECK(a1_hit.key == a1_miss.key, "#P4: 命中/未命中 key 须一致");
    CHECK(memo.size() == 1, "#P4: 同一 spec 两次调用只占一条目");

    const nn::ExprSpec a2 = make_unary_const(nn::ExprOp::Mul, Scalar{3});
    CHECK(nn::canon_key_memo_index(a1) != nn::canon_key_memo_index(a2),
          "#P4: const 值不同必须得到不同查找键（常量池布局依赖值）");
    const nn::CanonKey a2r = nn::canonicalize_with_key(a2);
    CHECK(memo.size() == 2, "#P4: 两种 const 值各占一条目");
    CHECK(equiv(a2r, fresh(a2)), "#P4: 不同 const 值的返回值须各自与现算等价");
    CHECK(!a2r.canon.consts.empty() && a2r.canon.consts[0] == Scalar{3},
          "#P4: 不得复用另一 const 值的常量池");

    // ── ② rparams 值：同条目 + 按当前 raw 刷新（无粘滞陈旧）──
    const auto make_rp = [](Scalar rp)
    {
        nn::ExprSpec s;
        s.views.push_back(nn::expr::linear());
        s.rparams.push_back(rp);
        nn::ExprInstr in;
        in.op = static_cast<std::uint8_t>(nn::ExprOp::Mul);
        in.dst = 0;
        in.a = nn::expr::input(0);
        in.b = nn::expr::rval(0);
        s.instrs.push_back(in);
        s.num_regs = 1;
        return s;
    };
    const nn::ExprSpec b1 = make_rp(Scalar{0.5f});
    const nn::ExprSpec b2 = make_rp(Scalar{0.25f});
    CHECK(nn::canon_key_memo_index(b1) == nn::canon_key_memo_index(b2),
          "#P4: rparams 值不进查找键（数量在结构口径里）");
    const std::size_t before_b = memo.size();
    const nn::CanonKey b1r = nn::canonicalize_with_key(b1);
    CHECK(memo.size() == before_b + 1, "#P4: rparams 基线新增一条目");
    const nn::CanonKey b2r = nn::canonicalize_with_key(b2);
    CHECK(memo.size() == before_b + 1, "#P4: 仅 rparams 值不同须命中同一查找键");
    CHECK(b2r.canon.rparams.size() == 1 && b2r.canon.rparams[0] == Scalar{0.25f},
          "#P4: 命中后 rparams 必须按当前 raw 刷新（否则 push constant 走旧值）");
    CHECK(equiv(b2r, fresh(b2)), "#P4: rparams 刷新后仍与现算逐字等价");
    const nn::CanonKey b1r2 = nn::canonicalize_with_key(b1);
    CHECK(b1r2.canon.rparams.size() == 1 && b1r2.canon.rparams[0] == Scalar{0.5f},
          "#P4: 回切基线 rparams 须回到基线值");
    CHECK(equiv(b1r2, fresh(b1)), "#P4: 回切后与现算等价");

    // ── ③ matmul k/transB 与 RowMod param：同条目 + 透传字段刷新 ──
    const auto make_mm = [](std::uint32_t k, std::uint8_t transB)
    {
        nn::ExprSpec s;
        s.views = {nn::expr::linear(), nn::expr::linear(), nn::expr::linear()};
        s.num_regs = 1;
        s.matmul = nn::MatmulSpec{0, 1, 0, transB, k, 1};
        s.instrs.push_back({static_cast<std::uint8_t>(nn::ExprOp::Add), 0,
                            nn::expr::matmul_op(), nn::expr::input(2), {}});
        return s;
    };
    const nn::ExprSpec c1 = make_mm(2, 0);
    const nn::ExprSpec c2 = make_mm(5, 1);
    CHECK(nn::canon_key_memo_index(c1) == nn::canon_key_memo_index(c2),
          "#P4: matmul k/transB 不进查找键（同 expr_spec_key 口径）");
    const std::size_t before_c = memo.size();
    CHECK(equiv(nn::canonicalize_with_key(c1), fresh(c1)), "#P4: matmul 基线等价");
    const nn::CanonKey c2r = nn::canonicalize_with_key(c2);
    CHECK(memo.size() == before_c + 1, "#P4: matmul 同结构命中同一查找键");
    CHECK(c2r.canon.matmul && c2r.canon.matmul->k == 5 && c2r.canon.matmul->transB == 1,
          "#P4: 命中后 matmul k/transB 必须按当前 raw 刷新");
    CHECK(equiv(c2r, fresh(c2)), "#P4: matmul 刷新后与现算逐字等价");

    const auto make_rm = [](std::uint32_t mod)
    {
        nn::ExprSpec s;
        s.views.push_back(nn::expr::row_mod(mod));
        s.num_regs = 1;
        s.instrs.push_back({static_cast<std::uint8_t>(nn::ExprOp::Add), 0,
                            nn::expr::input(0), nn::expr::input(0), {}});
        return s;
    };
    const nn::ExprSpec d1 = make_rm(2);
    const nn::ExprSpec d2 = make_rm(4);
    CHECK(nn::canon_key_memo_index(d1) == nn::canon_key_memo_index(d2),
          "#P4: RowMod 运行期 param 不进查找键（vp 走运行时填充）");
    const std::size_t before_d = memo.size();
    CHECK(equiv(nn::canonicalize_with_key(d1), fresh(d1)), "#P4: RowMod 基线等价");
    const nn::CanonKey d2r = nn::canonicalize_with_key(d2);
    CHECK(memo.size() == before_d + 1, "#P4: RowMod 同结构命中同一查找键");
    CHECK(d2r.canon.views.size() == 1 && d2r.canon.views[0].param == 4,
          "#P4: 命中后 views[].param 必须按当前 raw 刷新");
    CHECK(equiv(d2r, fresh(d2)), "#P4: RowMod 刷新后与现算逐字等价");

    // ── ④ fold：恒等返回、不进缓存（最小 fold spec，仅测 canonicalize+key 语义）──
    nn::ExprSpec f;
    f.views.push_back(nn::expr::linear());
    f.num_regs = 2;
    f.fold = nn::FoldSpec{};
    f.fold->k = 16;
    f.fold->num_state = 1;
    f.fold->inits = {Scalar{0}};
    {
        nn::ExprInstr b;
        b.op = static_cast<std::uint8_t>(nn::ExprOp::Add);
        b.dst = 1;
        b.a = nn::expr::reg(0);
        b.b = nn::expr::reduce(1);
        f.fold->body.push_back(b);
        nn::ExprInstr fin;
        fin.op = static_cast<std::uint8_t>(nn::ExprOp::Add);
        fin.dst = 0;
        fin.a = nn::expr::reg(0);
        fin.b = nn::expr::vec_state(0);
        f.fold->finalize.push_back(fin);
    }
    const std::size_t before_f = memo.size();
    const nn::CanonKey fr = nn::canonicalize_with_key(f);
    CHECK(memo.size() == before_f, "#P4: fold spec 不进备忘录（恒等 + 查找键无净收益）");
    CHECK(nn::expr_spec_equal(fr.canon, f), "#P4: fold 的 canon 须为 raw 恒等拷贝");
    CHECK(fr.key == nn::expr_spec_key(f), "#P4: fold 的 key 须为 expr_spec_key(raw)");

    memo.clear();
    std::printf("[OK] test_canon_key_memo\n");
}

// ── 测试 13：IR-D emitter 抽象（一份 canonical IR → 多后端） ───────────
void test_emitter()
{
    // 逐元素 spec：out = input0 * 2
    nn::ExprSpec s = make_unary_const(nn::ExprOp::Mul, Scalar{2});

    auto glsl = nn::emitter_registry::make("glsl");
    auto cpu = nn::emitter_registry::make("cpu");
    auto none = nn::emitter_registry::make("nonexistent");
    CHECK(static_cast<bool>(glsl), "GlslEmitter 已注册可创建");
    // CpuEmitter 产物从不参与编译（gen_fused 硬编码 glsl），死代码已清除（docs/17 §1.3）
    CHECK(!cpu, "CpuEmitter 已下线（不再注册）");
    CHECK(!none, "未知后端返回 nullptr");

    if (glsl)
    {
        CHECK(std::string(glsl->name()) == "glsl", "GlslEmitter 后端名");
        const std::string g = glsl->generate("fused_x", s);
        CHECK(!g.empty(), "GlslEmitter 生成逐元素 GLSL 成功");
        CHECK(g.find("#version 450") != std::string::npos, "GLSL 含版本头");
        CHECK(!glsl->generate_reduce("fused_r", make_row_sum()).empty(),
              "GlslEmitter 归约生成成功");
    }
    std::printf("[OK] test_emitter\n");
}

// ── 测试 14（#T5）：fold Ash 容量分档（a8f973d #P9）+ 链尾死存储消除（#P3）──
// 双域 fold（v2：mm 段 + vecacc + 元素链）从三个层面锁住这两个代码生成机制
//（此前只有 GPU 数值对拍间接兜底，无任何针对机制本身的断言）：
//   1. 分档特征（GLSL 文本）：mm_k ∈ {64,128,256,512,1024} → shared float
//      Ash[64|128|256|1024] 只出现本档；变体键 expr_spec_fold_variant_key
//      = 基础键 + "#m<cap>"（基线 1024 档无后缀），非 fold / fold 无 mm 段
//      键逐字不变；
//   2. 死存储（GLSL 文本）：链尾之后无人再读的中间寄存器**不得**出现
//      sreg[..][j] = r.. 写回；仍被后续段/尾部读的必须写回——含
//      vecacc.weight 的 kTailUse 保守计入（漏计 = vecacc 读上块未初始化槽
//      = 静默错值）；
//   3. 执行语义（CPU 数值对拍）：spec 经 CPU fold 执行器与独立参考一致
//      （A≡0.5、B≡1、V≡1 → C ≡ mm_k/2，O = Σ_j exp(exp(C))·1，V 全 1 使
//      vecacc 的 b(j,d) 索引约定不影响期望值）；GPU 侧执行语义由主仓
//      fused_gpu_test 的 fold 数值对拍覆盖（本测试二进制是纯 CPU 消费方形态）。
//
// body 结构（寄存器号 = 下方断言锚点；eidx 经 classify 现算并断言防漂移；
// 函数本体已迁共享 fixture expr_fuse_shapes.hpp——P10-③ 起 expr_fuse_test 共用）：
//   [0] a = Matmul + c0   （r1 元素链首）→ 仅被 [1] 读一次 → 链尾死存储
//   [1] b = exp(a)        （r2 元素）    → 被 [2]（RowSum）与 [4]（链2）读 → 活
//   [2] blk = RowSum(b)   （r3 块归约 dst = 行标量）→ flush 链1
//   [3] r0 += Reduce(blk) （标量行，状态跨块累计）
//   [4] w = exp(b)        （r4 元素末链）→ body 内无人再读，仅 vecacc 读它
// finalize: out = O + c0；vecacc: O += Σ_j w(j)·V(j,d)（无 scale）

void test_fold_ash_tier_and_dead_store()
{
    auto eng_owner = nn::create_cpu_engine();
    nn::ComputeEngine& eng = *eng_owner;
    auto glsl = nn::emitter_registry::make("glsl");
    CHECK(static_cast<bool>(glsl), "GlslEmitter 可创建");
    if (!glsl) return;

    // ── 1. Ash 容量分档：文本声明 + 变体键 ──
    struct Tier
    {
        std::uint32_t mm_k;
        const char* decl;   // shared 声明行（唯一出现点）
        const char* key;    // expr_spec_fold_variant_key 期望
    };
    const Tier tiers[] = {
        {64, "shared float Ash[64];", "K#m64"},
        {128, "shared float Ash[128];", "K#m128"},
        {256, "shared float Ash[256];", "K#m256"},
        {512, "shared float Ash[1024];", "K"},    // >256 → 基线档，键无后缀
        {1024, "shared float Ash[1024];", "K"},   // 基线原值
    };
    for (const Tier& t : tiers)
    {
        const nn::ExprSpec spec = make_fold_mm_chain(16, t.mm_k);
        const auto vr = nn::validate_expr_spec(spec, spec.views.size());
        std::string lab = "tier spec validate mm_k=" + std::to_string(t.mm_k);
        CHECK(vr.has_value(), lab.c_str());
        std::string vkey_st;
        lab = "变体键 #m<cap> mm_k=" + std::to_string(t.mm_k);
        CHECK(nn::expr_spec_fold_variant_key(spec, "K", vkey_st) == t.key, lab.c_str());
        const std::string g = glsl->generate("fold_tier", spec);
        lab = "tier GLSL 生成 mm_k=" + std::to_string(t.mm_k);
        CHECK(!g.empty(), lab.c_str());
        lab = "Ash 分档声明 mm_k=" + std::to_string(t.mm_k);
        CHECK(g.find(t.decl) != std::string::npos, lab.c_str());
        for (const Tier& o : tiers)
        {
            if (std::string(o.decl) == t.decl) continue;
            lab = "非本档 Ash 声明不得出现 mm_k=" + std::to_string(t.mm_k) + " vs " +
                  std::to_string(o.mm_k);
            CHECK(g.find(o.decl) == std::string::npos, lab.c_str());
        }
    }
    // 键保持：非 fold、fold 无 mm 段（无 Ash 预载 → 不分档）
    std::string vkey_st;
    CHECK(nn::expr_spec_fold_variant_key(make_unary_const(nn::ExprOp::Add, Scalar{1}), "K",
                                          vkey_st) == std::string("K"),
          "非 fold spec 键逐字不变");
    nn::ExprSpec no_mm = make_fold_mm_chain(16, 64);
    no_mm.fold->matmul.reset();
    CHECK(nn::expr_spec_fold_variant_key(no_mm, "K", vkey_st) == std::string("K"),
          "fold 无 mm 段键逐字不变");

    // ── 2. 链尾死存储消除：写回集合恰为 {b, w} ──
    const nn::ExprSpec base = make_fold_mm_chain(16, 4);
    CHECK(nn::validate_expr_spec(base, base.views.size()).has_value(), "base spec validate");
    const std::string g = glsl->generate("fold_dead", base);
    CHECK(!g.empty(), "base fold GLSL 生成成功");
    auto cls = nn::expr_fold_classify(*base.fold, base.num_regs);
    CHECK(cls.has_value(), "expr_fold_classify 成功");
    if (!cls || g.empty()) return;
    // eidx = 元素寄存器按号序的 shared 槽位（与生成器同一算法）；锚点断言
    // 防构造的寄存器编号漂移而断言失焦。
    std::vector<int> eidx(base.num_regs, -1);
    int ne = 0;
    for (std::uint32_t r = 0; r < base.num_regs; ++r)
        if ((*cls)[r]) eidx[r] = ne++;
    CHECK(ne == 3 && eidx[1] == 0 && eidx[2] == 1 && eidx[4] == 2,
          "元素槽映射锚点：a→sreg[0] b→sreg[1] w→sreg[2]");
    const auto store = [&](std::uint32_t r)
    {
        return "sreg[" + std::to_string(eidx[r]) + "][j] = r" + std::to_string(r) + ";";
    };
    // 活：b 被 RowSum 与后续链读；w 被 vecacc 尾部读（kTailUse 保守计入）
    CHECK(g.find(store(2)) != std::string::npos,
          "死存储消除：b（跨归约/末链读）必须存回 sreg");
    CHECK(g.find(store(4)) != std::string::npos,
          "死存储消除：w（vecacc 尾部读）必须存回——漏计即静默错值");
    // 死：a 写后全函数无人再读 → 不得出现写回
    CHECK(g.find(store(1)) == std::string::npos,
          "死存储消除：a（链尾无后续读）不得写回 sreg");
    // 链2 首次访问 = 读 b → 必须从 sreg 载入（与 b 的存回配对）
    CHECK(g.find("float r2 = sreg[" + std::to_string(eidx[2]) + "][j];") != std::string::npos,
          "链2 按读集载入 b（与存回配对）");

    // ── 3. CPU 执行语义数值对拍 ──
    const std::size_t rows = 3, K = 16, mmk = 4, veclen = 8;
    const nn::ExprSpec num = make_fold_mm_chain(static_cast<std::uint32_t>(K),
                                                static_cast<std::uint32_t>(mmk),
                                                static_cast<std::uint32_t>(veclen));
    nn::Tensor A = make_tensor(eng, rows, mmk, 0.5f, 0.0f);
    nn::Tensor B = make_tensor(eng, mmk, K, 1.0f, 0.0f);
    nn::Tensor V = make_tensor(eng, K, veclen, 1.0f, 0.0f);
    auto out = eng.eval_expr(num, std::vector<nn::Tensor>{A, B, V}, rows, veclen);
    CHECK(static_cast<bool>(out), "fold v2 CPU 求值成功");
    if (out)
    {
        // 参考：C = Σ_d A·B ≡ mmk·0.5；w = exp(exp(C))；O = Σ_{j<K} w·V ≡ K·w（V≡1）
        const Scalar w = std::exp(std::exp(static_cast<Scalar>(mmk) * Scalar{0.5f}));
        const nn::Matrix om = matrix_of(eng, *out);
        const auto sp = om.span();
        bool ok = sp.size() == rows * veclen;
        for (std::size_t r = 0; ok && r < rows; ++r)
        {
            Scalar acc = 0;
            for (std::size_t j = 0; j < K; ++j) acc += w * Scalar{1.0f};
            for (std::size_t d = 0; d < veclen; ++d)
            {
                const Scalar diff = std::fabs(sp[r * veclen + d] - acc);
                if (!(diff <= Scalar{1e-3f} * std::fabs(acc) + Scalar{1e-3f})) ok = false;
            }
        }
        CHECK(ok, "fold v2 CPU 数值 == 独立参考 K·exp(exp(mm_k/2))");
    }
    std::printf("[OK] test_fold_ash_tier_and_dead_store\n");
}

// ── 测试 13（P10-④）：F16 菱形进 P2 分量（写穿多输出 + NVI 边界 cast）────
// A = x*0.3；B = A*0.7；C = A*1.3（B/C 同时消费 A → 菱形），节点 prec=F16。
// 结构：p2_eligible 精度门 F32/F16（P10-④ 放宽）→ 单 kernel
//       members=3 / extras=[A,B] / prec=F16 / tail=C。
// 执行：eval_expr_multi_into 收 f16 输入 + f16 输出 → NVI 边界 cast
//       （输入抬 f32 → 全 f32 快路跑 V0 解释器 → f16 输出槽 cast_into 落回）；
//       改动前 NVI 对非全 f32 直接报错。
// 数值两条参考（算子全用乘法：避免 fma 收缩在"单 kernel"与"逐节点"两条
// f32 路径间产生不同的舍入点）：
//  ① f32 链（中间不舍入）→ 逐槽按 f16 舍入：融合结果应**逐位一致**——
//    融合在 f32 寄存器内传递中间值、写回时才舍入，与该参考同一舍入点；
//    f16 在本用例值域（O(0.1~1)）相邻值间距 ≥2^-14≈6e-5，1e-6 容差 =
//    逐位等值断言（两个不同的 f16 值差不出这么小）。
//  ② 逐节点 f16 参考（每节点各舍入一次，= 融合前的执行形态）：中间值舍入
//    点不同 → 差 ≤ 传播的 f16 半 ULP：|A|≤0.79 半 ULP≤2.4e-4，
//    B 链 ≤ 0.7×2.4e-4 + 半 ULP(0.55) ≈ 3e-4、C 链 ≤ 1.3×2.4e-4 +
//    半 ULP(1.03) ≈ 8.1e-4 → 容差 2e-3（约 2.5× 余量）。
void test_f16_diamond()
{
    auto eng_owner = nn::create_cpu_engine();
    nn::ComputeEngine& eng = *eng_owner;
    const std::size_t rows = 4, cols = 8;
    nn::ExprGraph g;
    // 输入以 f16 存储上传（先按 f16 舍入，两条路径读到同一位型）
    nn::Matrix xm(rows, cols);
    {
        auto sp = xm.span();
        for (std::size_t i = 0; i < sp.size(); ++i) sp[i] = 0.5f + static_cast<float>(i) * 0.07f;
    }
    auto x_r = eng.from_matrix(xm, nn::Precision::F16);
    CHECK(static_cast<bool>(x_r), "f16 输入构造");
    if (!x_r) return;
    const nn::Tensor& x = *x_r;
    const nn::ExprSpec m30 = make_unary_const(nn::ExprOp::Mul, Scalar{0.3f});
    const nn::ExprSpec m70 = make_unary_const(nn::ExprOp::Mul, Scalar{0.7f});
    const nn::ExprSpec m13 = make_unary_const(nn::ExprOp::Mul, Scalar{1.3f});

    const int nA = g.add_node(m30, std::vector<nn::Tensor>{x}, rows, cols,
                              /*vector_out=*/false, nn::Precision::F16);
    nn::Tensor yt = make_placeholder(eng, rows, cols, g.tag_of_node[static_cast<std::size_t>(nA)],
                                     nn::Precision::F16);
    (void)g.add_node(m70, std::vector<nn::Tensor>{yt}, rows, cols, false, nn::Precision::F16);
    nn::Tensor yt2 = make_placeholder(eng, rows, cols, g.tag_of_node[static_cast<std::size_t>(nA)],
                                      nn::Precision::F16);
    (void)g.add_node(m13, std::vector<nn::Tensor>{yt2}, rows, cols, false, nn::Precision::F16);

    auto kernels = fuse_expr_graph(g);
    CHECK(kernels.size() == 1, "F16 菱形：p2_eligible 放宽后融合为单 kernel");
    if (kernels.size() != 1) return;
    CHECK(kernels[0].members.size() == 3, "F16 菱形：3 成员（A,B,C）");
    CHECK(kernels[0].spec.extras.size() == 2, "F16 菱形：extras=[A,B]（写穿）");
    CHECK(kernels[0].prec == nn::Precision::F16, "F16 菱形：分量 prec=F16");
    CHECK(kernels[0].tail == nA + 2, "F16 菱形：tail=C（末节点）");

    // 执行：f16 输入 + f16 输出（核心执行断言——NVI 边界 cast）
    std::vector<nn::Tensor> kinputs;
    for (auto& in : kernels[0].inputs) kinputs.push_back(in.external);
    std::vector<nn::Tensor> kouts;
    for (int k = 0; k < 3; ++k)
        kouts.push_back(eng.create_tensor(rows, cols, nn::Precision::F16));
    auto mul_r = eng.eval_expr_multi_into(kernels[0].spec, kinputs, rows, cols, kouts);
    CHECK(static_cast<bool>(mul_r), "F16 多输出求值成功（NVI 边界 cast）");
    if (!mul_r)
    {
        std::printf("    err=%s\n", mul_r.error().message.c_str());
        return;
    }
    // 槽序（与 test_multi_consumer 同）：[0]=tail(C)，[1]=A，[2]=B

    // 参考①：f32 链（中间不舍入）→ 逐槽舍入 f16 —— 融合结果逐位一致
    auto a32 = eng.eval_expr(m30, std::vector<nn::Tensor>{x}, rows, cols, nn::Precision::F32);
    CHECK(static_cast<bool>(a32), "参考① f32 链 A 求值");
    if (!a32) return;
    auto b32 = eng.eval_expr(m70, std::vector<nn::Tensor>{*a32}, rows, cols, nn::Precision::F32);
    auto c32 = eng.eval_expr(m13, std::vector<nn::Tensor>{*a32}, rows, cols, nn::Precision::F32);
    CHECK(static_cast<bool>(b32) && static_cast<bool>(c32), "参考① f32 链 B/C 求值");
    if (!b32 || !c32) return;
    auto a16r = eng.cast(*a32, nn::Precision::F16);
    auto b16r = eng.cast(*b32, nn::Precision::F16);
    auto c16r = eng.cast(*c32, nn::Precision::F16);
    CHECK(static_cast<bool>(a16r) && static_cast<bool>(b16r) && static_cast<bool>(c16r),
          "参考① f32→f16 逐槽舍入");
    if (!a16r || !b16r || !c16r) return;
    // 槽 ↔ 参考：[0]=C，[1]=A，[2]=B；1e-6 = 逐位等值（f16 值域 O(0.1~1)，
    // 相邻值间距 ≥2^-14≈6e-5 → 差不出 1e-6，只有"完全相同"能过）
    CHECK(max_abs_diff(matrix_of(eng, kouts[0]), matrix_of(eng, *c16r)) < 1e-6f,
          "融合[tail=C] == f32 链舍入（逐位）");
    CHECK(max_abs_diff(matrix_of(eng, kouts[1]), matrix_of(eng, *a16r)) < 1e-6f,
          "融合[extras A] == f32 链舍入（逐位）");
    CHECK(max_abs_diff(matrix_of(eng, kouts[2]), matrix_of(eng, *b16r)) < 1e-6f,
          "融合[extras B] == f32 链舍入（逐位）");

    // 参考②：逐节点 f16（每节点各舍入一次 = 融合前的执行形态）→ 融合前后对拍
    auto a16 = eng.eval_expr(m30, std::vector<nn::Tensor>{x}, rows, cols, nn::Precision::F16);
    CHECK(static_cast<bool>(a16), "参考② 逐节点 f16 A 求值");
    if (!a16) return;
    auto b16 = eng.eval_expr(m70, std::vector<nn::Tensor>{*a16}, rows, cols, nn::Precision::F16);
    auto c16 = eng.eval_expr(m13, std::vector<nn::Tensor>{*a16}, rows, cols, nn::Precision::F16);
    CHECK(static_cast<bool>(b16) && static_cast<bool>(c16), "参考② 逐节点 f16 B/C 求值");
    if (!b16 || !c16) return;
    // A 是根节点（输入未再舍入）→ 与融合 extras[0] 逐位；B/C 走 f16 量级容差
    CHECK(max_abs_diff(matrix_of(eng, kouts[1]), matrix_of(eng, *a16)) < 1e-6f,
          "融合[extras A] == 逐节点 f16 A（根节点逐位）");
    CHECK(max_abs_diff(matrix_of(eng, kouts[2]), matrix_of(eng, *b16)) < 2e-3f,
          "融合[extras B] ≈ 逐节点 f16 B（f16 半 ULP 传播量级）");
    CHECK(max_abs_diff(matrix_of(eng, kouts[0]), matrix_of(eng, *c16)) < 2e-3f,
          "融合[tail C] ≈ 逐节点 f16 C（f16 半 ULP 传播量级）");
    std::printf("[OK] test_f16_diamond (kernels=%zu)\n", kernels.size());
}

} // namespace

int main()
{
    test_chain_fuse();
    test_chain3_fuse();
    test_independent();
    test_multi_consumer();
    test_reduce_broadcast_fuse();
    test_reduce_axis_mismatch_boundary();
    test_fold_tail_fuse();
    test_fold_as_chain_member_rejected();
    test_fold_v2_tail();
    test_limit_pressure();
    test_determinism();
    test_dep_recognition();
    test_input_order();
    test_matmul_chain_fuse();
    test_matmul_pure_chain_fuse();
    test_matmul_multi_consumer_fuse();
    test_plan_cache_runtime_values();
    test_canon_key_memo();
    test_emitter();
    test_fold_ash_tier_and_dead_store();
    test_f16_diamond();

    if (g_fail == 0)
    {
        std::printf("expr_graph_test: ALL PASS\n");
        return 0;
    }
    std::printf("expr_graph_test: %d FAILED\n", g_fail);
    return 1;
}
