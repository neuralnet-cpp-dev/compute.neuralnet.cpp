// ═══════════════════════════════════════════════════════════════════════════
//  consumer_test.cpp — 消费方验证（只 include 抽象层 MIT + 链接 libnn-cpu.so）
//
//  用完整 nn::ComputeEngine 接口 + Tensor 值类型（跨界布局）：
//  工厂 create_cpu_engine() → from_matrix / matmul / 归约 / 原地操作 / transpose。
//  本 TU 不 include 任何引擎实现头（LGPL）。
//  （eval_expr 的 ExprSpec 载荷生产方是引擎 DSL；JIT 现编路径由 kernelgen_smoke
//   在引擎侧验证，消费方不手编 spec。）
// ═══════════════════════════════════════════════════════════════════════════
#include <neuralnet.cpp/compute_engine.hpp>

#include <cmath>
#include <cstdio>
#include <cstring>

using namespace nn;

static int failures = 0;
static void check(bool ok, const char* what)
{
    if (!ok) { std::printf("FAIL: %s\n", what); ++failures; }
}

int main()
{
    auto e = create_cpu_engine();
    check(e != nullptr, "create_cpu_engine");
    if (!e) { std::printf("  last_error: %s\n", last_error() ? last_error() : "(无)"); return 1; }
    std::printf("engine: %s\n", e->describe().text().c_str());

    // ── matmul：A(2x3) × B(3x2) = C(2x2) ─────────────────────────────
    Matrix mA(2, 3), mB(3, 2);
    const float Ad[6] = {1, 2, 3, 4, 5, 6};
    const float Bd[6] = {7, 8, 9, 10, 11, 12};
    std::memcpy(mA.span().data(), Ad, 6 * sizeof(float));
    std::memcpy(mB.span().data(), Bd, 6 * sizeof(float));
    auto rA = e->from_matrix(mA, Precision::F32);
    auto rB = e->from_matrix(mB, Precision::F32);
    check(rA.has_value() && rB.has_value(), "from_matrix");
    if (!rA || !rB) return 1;
    Tensor A = *rA, B = *rB;

    auto rC = e->matmul(A, B, false, false, Precision::F32);
    check(rC.has_value(), "matmul");
    if (!rC) return 1;
    auto rCm = e->to_matrix(*rC);
    check(rCm.has_value(), "to_matrix");
    const float expC[4] = {58, 64, 139, 154};
    bool mm_ok = true;
    for (int i = 0; i < 4; ++i)
        if (std::fabs(rCm->span().data()[i] - expC[i]) > 1e-4f)
        { mm_ok = false; std::printf("  C[%d]=%f exp %f\n", i, rCm->span().data()[i], expC[i]); }
    check(mm_ok, "matmul values");

    // ── 归约 ─────────────────────────────────────────────────────────
    auto rs = e->row_reduce_sum(A);
    auto rsm = e->to_matrix(*rs);
    check(rsm.has_value() && std::fabs(rsm->span().data()[0] - 6) < 1e-4f &&
              std::fabs(rsm->span().data()[1] - 15) < 1e-4f, "row_reduce_sum");
    auto cm = e->col_reduce_max(A);
    auto cmm = e->to_matrix(*cm);
    check(cmm.has_value() && std::fabs(cmm->span().data()[0] - 4) < 1e-4f &&
              std::fabs(cmm->span().data()[2] - 6) < 1e-4f, "col_reduce_max");

    // ── 原地操作 ─────────────────────────────────────────────────────
    auto rc = e->clone(A);
    check(rc.has_value(), "clone");
    check(e->scale_inplace(*rc, 2.0f).has_value(), "scale_inplace");
    check(e->add_inplace(*rc, A).has_value(), "add_inplace");
    auto rcm = e->to_matrix(*rc);
    bool d_ok = true;
    for (int i = 0; i < 6; ++i)
        if (std::fabs(rcm->span().data()[i] - 3 * Ad[i]) > 1e-4f) d_ok = false;
    check(d_ok, "add/scale values (clone==3A)");

    // ── transpose ────────────────────────────────────────────────────
    auto rt = e->transpose(A);
    check(rt.has_value() && rt->rows() == 3 && rt->cols() == 2, "transpose shape");

    // ── 批处理窗口（CPU no-op）──────────────────────────────────────
    check(e->begin_batch().has_value(), "begin_batch");
    auto rC2 = e->matmul(A, B, false, false, Precision::F32);
    check(rC2.has_value(), "matmul in batch");
    check(e->end_batch().has_value(), "end_batch");

    if (failures == 0) { std::printf("ALL PASS\n"); return 0; }
    std::printf("%d FAILURES\n", failures);
    return 1;
}
