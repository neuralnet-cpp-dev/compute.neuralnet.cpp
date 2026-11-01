// ═══════════════════════════════════════════════════════════════════════════
//  consumer_test.cpp — 消费方验证（只 include 抽象层 MIT + 链接 libnn-cpu.so）
//
//  本 TU 不 include 任何引擎头（LGPL）—— 只用抽象层 neuralnet::ComputeEngine
//  接口 + create_cpu_engine() 工厂，验证 C++ 直出的 .so 完整可用。
// ═══════════════════════════════════════════════════════════════════════════
#include <neuralnet.cpp/compute_engine.hpp>

#include <cmath>
#include <cstdio>

using namespace neuralnet;

static int failures = 0;

static void check(bool ok, const char* what)
{
    if (!ok) { std::printf("FAIL: %s\n", what); ++failures; }
}

int main()
{
    auto e = create_cpu_engine();
    check(e != nullptr, "create_cpu_engine");
    if (!e) { std::printf("last_error: %s\n", e ? "?" : "n/a"); return 1; }
    std::printf("engine describe: %s\n", e->describe());

    // ── matmul：A(2x3) × B(3x2) = C(2x2) ─────────────────────────────
    const float Ad[6] = {1, 2, 3, 4, 5, 6};
    const float Bd[6] = {7, 8, 9, 10, 11, 12};
    auto A = e->from_matrix(Ad, 2, 3, Precision::F32);
    auto B = e->from_matrix(Bd, 3, 2, Precision::F32);
    check(A.valid() && B.valid(), "from_matrix");
    auto C = e->matmul(A, B, false, false);
    check(C.valid(), "matmul");
    float c[4];
    std::size_t n = 0;
    check(e->to_matrix(C, c, 4, &n), "to_matrix");
    check(n == 4, "to_matrix shape");
    const float expC[4] = {58, 64, 139, 154};
    bool mm_ok = true;
    for (int i = 0; i < 4; ++i)
        if (std::fabs(c[i] - expC[i]) > 1e-4f) { mm_ok = false; std::printf("  C[%d]=%f exp %f\n", i, c[i], expC[i]); }
    check(mm_ok, "matmul values");

    // ── 归约 ─────────────────────────────────────────────────────────
    // row_reduce_sum(A): 行和 → [6, 15]；col_reduce_max(A): 列 max → [4,5,6]
    auto rs = e->row_reduce_sum(A);
    float rsd[2]; check(e->to_matrix(rs, rsd, 2, &n), "row_reduce_sum");
    check(std::fabs(rsd[0] - 6) < 1e-4f && std::fabs(rsd[1] - 15) < 1e-4f, "row_reduce_sum values");
    auto cm = e->col_reduce_max(A);
    float cmd[3]; check(e->to_matrix(cm, cmd, 3, &n), "col_reduce_max");
    check(std::fabs(cmd[0] - 4) < 1e-4f && std::fabs(cmd[1] - 5) < 1e-4f && std::fabs(cmd[2] - 6) < 1e-4f,
          "col_reduce_max values");

    // ── 原地操作 ─────────────────────────────────────────────────────
    auto D = e->clone(A);                                  // D = A
    check(D.valid(), "clone");
    check(e->scale_inplace(D, 2.0f), "scale_inplace");     // D = 2A
    check(e->add_inplace(D, A), "add_inplace");            // D = 3A
    float dd[6]; check(e->to_matrix(D, dd, 6, &n), "to_matrix D");
    bool d_ok = true;
    for (int i = 0; i < 6; ++i)
        if (std::fabs(dd[i] - 3 * Ad[i]) > 1e-4f) d_ok = false;
    check(d_ok, "add/scale values (D==3A)");

    // ── transpose ────────────────────────────────────────────────────
    auto T = e->transpose(A);                               // A^T (3x2)
    check(T.valid() && T.rows() == 3 && T.cols() == 2, "transpose shape");
    float td[6]; check(e->to_matrix(T, td, 6, &n), "to_matrix T");
    check(std::fabs(td[0] - 1) < 1e-4f && std::fabs(td[1] - 4) < 1e-4f, "transpose values");

    // ── 销毁 + 引擎析构连带回收 ─────────────────────────────────────
    e->destroy_tensor(A);
    e->destroy_tensor(B);
    e->destroy_tensor(C);
    // D/T/rs/cm 随引擎析构连带回收

    if (failures == 0) { std::printf("ALL PASS\n"); return 0; }
    std::printf("%d FAILURES\n", failures);
    return 1;
}
