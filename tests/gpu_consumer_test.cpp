// ═══════════════════════════════════════════════════════════════════════════
//  gpu_consumer_test.cpp — GPU 引擎消费方验证（只 include 抽象层 MIT + 链 libnn-gpu.so）
// ═══════════════════════════════════════════════════════════════════════════
#include <neuralnet.cpp/compute_engine.hpp>

#include <cmath>
#include <cstdio>
#include <cstring>

using namespace nn;

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
    std::printf("gpu engine: %s\n", e->describe().text().c_str());

    Matrix mA(2, 3), mB(3, 2);
    const float Ad[6] = {1, 2, 3, 4, 5, 6};
    const float Bd[6] = {7, 8, 9, 10, 11, 12};
    std::memcpy(mA.span().data(), Ad, 6 * sizeof(float));
    std::memcpy(mB.span().data(), Bd, 6 * sizeof(float));
    auto rA = e->from_matrix(mA, Precision::F32);
    auto rB = e->from_matrix(mB, Precision::F32);
    if (!rA.has_value() || !rB.has_value()) { std::printf("FAIL from_matrix: %s\n", last_error()); return 1; }
    Tensor A = *rA, B = *rB;
    std::printf("A: valid=%d gpu=%d prec=%d shape=%zux%zu\n", A.valid(), A.is_gpu(),
                (int)A.precision(), A.rows(), A.cols());
    e->begin_batch();
    auto rC = e->matmul(A, B, false, false, Precision::F32);
    e->end_batch();
    if (!rC.has_value()) { std::printf("FAIL matmul: %s\n", rC.error().message.c_str()); return 1; }
    auto rCm = e->to_matrix(*rC);
    if (!rCm.has_value()) { std::printf("FAIL to_matrix: %s\n", last_error()); return 1; }
    const float expC[4] = {58, 64, 139, 154};
    bool ok = true;
    for (int i = 0; i < 4; ++i)
        if (std::fabs(rCm->span().data()[i] - expC[i]) > 1e-3f)
        { ok = false; std::printf("  C[%d]=%f exp %f\n", i, rCm->span().data()[i], expC[i]); }
    if (!ok) { std::printf("FAIL matmul values\n"); return 1; }
    std::printf("GPU matmul PASS (C=[58,64,139,154])\n");
    return 0;
}
