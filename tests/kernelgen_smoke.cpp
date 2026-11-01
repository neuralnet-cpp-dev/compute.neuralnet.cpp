// ═══════════════════════════════════════════════════════════════════════════
//  kernelgen_smoke.cpp — libnn-kernelgen.so 的 C ABI 冒烟测试
//
//  构造最小 ExprSpec（x+y 逐元素）→ encode_expr_spec → nnk_generate_spirv
//  → 验证 NNK_OK + SPIR-V 非空 + 元数据合理。证明 JIT 生成器 .so 可用。
// ═══════════════════════════════════════════════════════════════════════════
#include <neuralnet.cpp/expr_spec.hpp>
#include <neuralnet.cpp/kernel_cache.hpp>   // encode_expr_spec
#include <neuralnet.cpp/kernelgen_api.h>

#include <cstdio>

int main()
{
    // 最小逐元素结构：out = in0 + in1
    nn::ExprSpec spec;
    spec.instrs.push_back(nn::ExprInstr{
        static_cast<std::uint8_t>(nn::ExprOp::Add),  // op
        0,                                          // dst reg 0
        nn::ExprOperand{1, 0},                      // a = Input[0]
        nn::ExprOperand{1, 1},                      // b = Input[1]
        nn::ExprOperand{0, 0}});                    // c 未用
    spec.views = {{0, 0, 0, 0}, {0, 0, 0, 0}};      // 两个 Linear 视图
    spec.num_regs = 1;

    const std::vector<std::uint8_t> bytes = nn::fused::encode_expr_spec(spec);
    if (bytes.empty()) { std::printf("FAIL: encode_expr_spec 为空\n"); return 1; }

    std::printf("generator version: %s\n", nnk_generator_version());

    nnk_kernel_meta meta{};
    std::uint8_t* spirv = nullptr;
    std::size_t len = 0;
    const nnk_status st = nnk_generate_spirv(bytes.data(), bytes.size(), 0, &meta, &spirv, &len);
    if (st != NNK_OK) { std::printf("FAIL: nnk_generate_spirv=%d\n", st); return 1; }
    if (spirv == nullptr || len == 0) { std::printf("FAIL: 空 SPIR-V\n"); return 1; }
    if (len % 4 != 0) { std::printf("FAIL: SPIR-V 长度非 4 对齐\n"); return 1; }

    // SPIR-V magic 头 0x07230203
    const std::uint32_t magic = *reinterpret_cast<const std::uint32_t*>(spirv);
    if (magic != 0x07230203u) { std::printf("FAIL: SPIR-V magic=0x%08x\n", magic); return 1; }

    std::printf("nnk_generate_spirv OK: %zu bytes, magic=0x%08x, reduce_axis=%d, vec_width=%u\n",
                len, magic, meta.reduce_axis, meta.vec_width);
    nnk_free_spirv(spirv);
    std::printf("KERNELGEN SMOKE PASS\n");
    return 0;
}
