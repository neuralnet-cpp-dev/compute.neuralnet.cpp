// ── mm_swz.glsl — vec4 共享 tile 的 bank swizzle（写读同式，involution）─────
// 适用布局：shared vec4 S[stage][BK][SLOTS]，SLOTS = BM/4 或 BN/4（16B/槽）。
//
// 冲突来源：vec4 数组的 k 维步长 = SLOTS 槽 = SLOTS*4 个 word ≡ 0 (mod 32 bank)
// → 标量散布写的 **bank**（不是地址！k 维仍在地址里，任何 BK 都不会互相覆盖）
// 只由 slot 与 m&3 决定。一个 warp 的 32 个线程在同一条 store 指令上覆盖
// 8 个连续 m（m>>2 取相邻两值 q / q+1，m&3 取 0..3）× 4 个 k 组（k>>2 = 0..3），
// 故 bank = 4*((m>>2) ^ f(k>>2)) + (m&3) (mod 32)。
//
// 无冲突条件：q 与 q+1 两组映射出的 slot 在 mod 8 下不相交。q 偶时 q^(q+1) = 1，
// 故需 {f(0..3)} 内任意两值异或 ≠ 1 ⇒ f 只能取偶偏移 {0,2,4,6}：
//   f(a) = (a & 3) << 1   → 4 个 k 组各错开 2 槽 = 8 个 bank → **1-way**
//   （原式 f(a) = a & 3 里 0^1 = 1 → 两两相撞 → 2-way；完全不 swizzle → 4-way）
// 两个前提（模拟器复算，非推测，见 sim_bank.py / sim_kernel.py 的 P4）：
//   · SLOTS >= 8 才放得下 {0,2,4,6}。SLOTS = 4（细 tile 16×16）时 32 线程只能
//     落在 4 槽 × 4 分量 = 16 个 bank 上，**2-way 已是理论下限** → 退回原式。
//   · 一条指令内得有 4 个不同 k 组。BK = 8（128×128/8×8 变体）时每 warp 只有
//     k>>2 ∈ {0,1} 两组 → 两种式子都是 2-way，本式在该形状下**不带来收益**。
//   收益形状：BK = 16 且 SLOTS >= 8（= 64×64 tile 的 transA=0 A 侧、
//   transB=1 B 侧；以及 batched_matmul / matmul_q 的对应路径）。
//
// 整槽 vec4 写（transA=1 的 A、transB=0 的 B）：一条 STS.128 满 512B/warp =
// 4 wavefront，就是理论下限，与 f 无关 → 不受影响，也不受益。
#ifndef MM_SWZ_GLSL
#define MM_SWZ_GLSL

#define MM_SWZ_F(a, slots) (((slots) >= 8u) ? (((a) & 3u) << 1u) : ((a) & 3u))
#define MM_SWZ(k, slot, slots) \
    ((slot) ^ (MM_SWZ_F(((k) >> 2), (slots)) & ((slots) - 1u)))

#endif
