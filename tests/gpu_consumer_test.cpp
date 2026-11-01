// ═══════════════════════════════════════════════════════════════════════════
//  gpu_consumer_test.cpp — GPU 引擎消费方验证（只 include 抽象层 MIT + 链 libnn-gpu.so）
// ═══════════════════════════════════════════════════════════════════════════
#include <neuralnet.cpp/compute_engine.hpp>

#include <cmath>
#include <cstdio>

using namespace neuralnet;

int main()
{
    auto e = create_gpu_engine(nullptr);
    if (!e)
    {
        // 无可用 Vulkan 设备：非致命，打印说明后以 0 退出（跳过，不判失败）
        std::printf("SKIP: create_gpu_engine 返回 nullptr：%s\n",
                    last_error() ? last_error() : "(无错误信息)");
        return 0;
    }
    std::printf("gpu engine: %s\n", e->describe());

    // matmul：A(2x3) × B(3x2) = C(2x2)
    const float Ad[6] = {1, 2, 3, 4, 5, 6};
    const float Bd[6] = {7, 8, 9, 10, 11, 12};
    auto A = e->from_matrix(Ad, 2, 3, Precision::F32);
    auto B = e->from_matrix(Bd, 3, 2, Precision::F32);
    if (!A.valid() || !B.valid()) { std::printf("FAIL from_matrix: %s\n", e->last_error()); return 1; }
    e->begin_batch();
    auto C = e->matmul(A, B, false, false);
    e->end_batch();
    if (!C.valid()) { std::printf("FAIL matmul: %s\n", e->last_error()); return 1; }
    float c[4];
    std::size_t n = 0;
    if (!e->to_matrix(C, c, 4, &n)) { std::printf("FAIL to_matrix: %s\n", e->last_error()); return 1; }
    const float expC[4] = {58, 64, 139, 154};
    bool ok = true;
    for (int i = 0; i < 4; ++i)
        if (std::fabs(c[i] - expC[i]) > 1e-3f) { ok = false; std::printf("  C[%d]=%f exp %f\n", i, c[i], expC[i]); }
    if (!ok) { std::printf("FAIL matmul values\n"); return 1; }
    std::printf("GPU matmul PASS (C=[58,64,139,154])\n");
    e->destroy_tensor(A);
    e->destroy_tensor(B);
    e->destroy_tensor(C);
    return 0;
}
