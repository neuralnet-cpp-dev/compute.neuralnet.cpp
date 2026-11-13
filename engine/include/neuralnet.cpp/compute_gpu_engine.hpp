#pragma once

// ── compute_gpu_engine.hpp — GPU 计算引擎实现（纯 GPU 架构）─────────────────────────
// GpuEngine 封装 GpuBackend，实现 ComputeEngine 接口。
//
// 纯 GPU 架构策略：
//   - 所有原语（matmul、elementwise、reduce、broadcast、zero、clone）均在
//     GPU 上原生执行，数据全程驻留 GPU 显存。
//   - from_matrix：上传 CPU Matrix → GPU Tensor（唯一的 PCIe 上传点）
//   - to_matrix：下载 GPU Tensor → CPU Matrix（唯一的 PCIe 下载点）
//   - create_tensor：分配 GPU buffer（用于参数/梯度）
//   - clone：GPU 内 buffer 拷贝（无 PCIe 传输）
//   - 所有原语返回 GPU Tensor，不回退到 CPU
//
//   前向/反向链路全程 GPU：
//     from_matrix(上传) → matmul(GPU) → broadcast(GPU) → elementwise(GPU)
//     → reduce(GPU) → ... → to_matrix(下载)
//   除 batch 边界外，无 PCIe 传输。
//
// 同步模型：
//   batch 录制模式：begin_batch 后所有原语录制到每帧共享的 command buffer，
//   end_batch 提交当前帧**不等待**（消除 per-primitive 的提交/阻塞开销，
//   host 继续录制、GPU 在队列上先行执行，同队列 FIFO 保证执行顺序）；
//   真正要读 GPU 结果（to_matrix / copy_from）前由 wait_in_flight 等待
//   所有在飞帧。to_matrix/from_matrix 会打断 batch（flush 后自动重新
//   begin_batch）。
//
// 原地操作语义：
//   add_inplace / scale_inplace 均直接写回
//   A 自己的 buffer（逐元素 kernel 每线程只读写自己下标一次，read-before-write
//   天然成立；同一 command buffer 内按录制顺序执行）。zero 用 vkCmdFillBuffer。
//   目的：消除「分配新 buffer + 全量写出」在优化器/梯度累积路径上的分配风暴。
//   注意：真原地要求调用方保证 A 不被同一录制窗口内已录制的命令读引用，
//   也不能与别的 Tensor 共享 buffer 且语义上需要保持独立（否则请传副本）。
// ─────────────────────────────────────────────────────────────────────────

#ifdef NN_HAS_VULKAN

#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <cstdio>    // std::remove（.nnk 写失败时清理截断的 tmp）
#include <fstream>
#include <sstream>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#include <fcntl.h>       // open/O_CREAT（.nnk 缓存进程锁）
#include <sys/file.h>    // flock
#include <unistd.h>      // close / rename
#include <sys/stat.h>    // mkdir（.nnk 缓存目录自动创建）
#endif

namespace
{
// .nnk 缓存目录首次运行可能不存在（Windows 尤其：%LOCALAPPDATA% 下无
// neuralnet-cpp），写盘前自动创建（EEXIST / ERROR_ALREADY_EXISTS 无害）。
// 项目禁异常（-fno-exceptions）→ 不用 std::filesystem，走平台 API。
inline void ensure_nnk_cache_dir_(const std::string& path)
{
    const auto pos = path.find_last_of("/\\");
    if (pos == std::string::npos) return;
    const std::string dir = path.substr(0, pos);
#ifdef _WIN32
    CreateDirectoryA(dir.c_str(), nullptr);
#else
    ::mkdir(dir.c_str(), 0755);
#endif
}
}  // namespace

#include "compute_engine.hpp"
#include "expr_opt.hpp"
#include "expr_dsl.hpp"    // matmul_with_bias 经 DSL 融合（单一事实源）
#include "expr_glsl_gen.hpp"  // fused_kernel_meta_of（JIT 懒建 pipeline 的元数据）
#include "ternary_pack.hpp"  // 三值打包布局（trit_pack / matmul_q，P2/F3）
#include "int8_pack.hpp"     // int8 激活打包布局（quantize_i8 / matmul_q_i8，P3）
#include "kernel_cache.hpp"  // KernelCache / kernel_cache_key / encode_expr_spec（Phase 2 JIT）
#include "kernelgen_api.h"   // libnn-kernelgen.so C ABI（Phase 2 JIT 现编）

#ifdef NN_FUSED_REGISTRY_EMBEDDED
#include "fused_registry.hpp"
#endif
#include "backend/compute_vk_backend.hpp"

namespace nn
{

// ══════════════════════════════════════════════════════════════════════════
// GpuEngine — GPU 计算引擎（纯 GPU 架构）
// ══════════════════════════════════════════════════════════════════════════
class GpuEngine final : public ComputeEngine
{
private:
    std::shared_ptr<GpuBackend> backend_;

    // ── Phase 2 运行期 JIT：.nnk 缓存 + 生成器 .so（懒加载，见 docs/internal/33 §2/§10）──
    // 派发 miss → 内存 KernelCache（懒加载自 kernel_cache_path()）→ miss →
    // dlopen libnn-kernelgen.so → nnk_generate_spirv 现编 → insert → 后端懒建
    // pipeline → 执行。内嵌注册表（构建期 AOT）降级为"预热加速器"，非派发硬约束。
    // 三线程/三进程语义：本引擎单线程 dispatch；缓存落盘 temp+rename 原子写。
    struct JitGenerator
    {
        void* handle = nullptr;
        decltype(&nnk_generate_spirv) generate = nullptr;
        decltype(&nnk_free_spirv) free = nullptr;
        decltype(&nnk_generator_version) version = nullptr;
        [[nodiscard]] bool loaded() const noexcept { return handle != nullptr; }
    };
    mutable std::shared_ptr<nn::fused::KernelCache> kernel_cache_;
    mutable JitGenerator jit_;

    // typed JIT（in-kernel f16 直通）失败负缓存：结构级生成/建管线/派发失败
    // 的键不再重试——否则"生成器不支持该形态"会在热路径上每次调用都重付
    // shaderc 现编（量级 10–50 ms/次 × 数百次/步）。键 = .nnk 缓存 K；
    // 首次插入时借 insert().second 打一条回退诊断（每键一次）。
    std::unordered_set<std::string> typed_negative_;
    std::mutex typed_negative_mu_;

    explicit GpuEngine(std::shared_ptr<GpuBackend> backend)
        : ComputeEngine(backend ? backend->identity() : 0), backend_(std::move(backend))
    {
    }

public:
    // 引擎必须由 shared_ptr 管理：Layer / Model / Optimizer 会保存强引用。
    [[nodiscard]] static std::shared_ptr<GpuEngine> create(std::shared_ptr<GpuBackend> backend)
    {
        NN_ASSERT_ALWAYS(backend != nullptr, "GpuEngine::create: backend 为空");
        return std::shared_ptr<GpuEngine>(new GpuEngine(std::move(backend)));
    }

    // 兼容仍是 GpuBackend& 的调用点：从 backend 自身取 keepalive。
    [[nodiscard]] static std::shared_ptr<GpuEngine> create(GpuBackend& backend)
    { return create(backend.keepalive()); }

    [[nodiscard]] Device device() const noexcept override { return Device::GPU; }

    // ── 批处理：激活 GpuBackend 的 command buffer 录制模式 ──────────────
    // begin_batch 后所有原语录制到当前帧的 batch_cmd_，end_batch 提交该帧
    // **不等待**（阻塞点在 wait_in_flight，见文件头「同步模型」）。
    // 这消除了 per-primitive 的 vkQueueSubmit+vkWaitForFences 开销。
    [[nodiscard]] Result<void> begin_batch() override
    {
        NN_PROF_OP("begin_batch");
        return backend_->begin_batch();
    }

    [[nodiscard]] Result<void> end_batch() override
    {
        NN_PROF_OP("end_batch");
        return backend_->end_batch();
    }

    // batch 模式查询（基类默认 false；GPU 委托 backend，供 InferenceScope
    // / Model::forward 自动作用域判定，issue #28）
    [[nodiscard]] bool batch_active() const noexcept override { return backend_->in_batch(); }

    // ── 表达式录制（IR-C）──────────────────────────────
    // begin_expr 开启录制：期间 eval_expr/eval_expr_reduce 在引擎 eval*_impl
    // 顶部拦截入图（方案 A：录制状态读写全部归属引擎 TU，跨 DSO TLS 单副本
    // 生效），返回带 virtual_tag 的占位 GPU 张量（真实存储）。
    // end_expr 融合分析（fuse_expr_graph）→ 复合 kernel 直接写回各 tail
    // 占位存储（output_override），中间量内联为寄存器。复合 spec 未命中
    // AOT 注册表 miss = JIT 现编兜底（须在 scan_exprs 覆盖录制段以预热）。
    [[nodiscard]] Result<void> begin_expr() override
    {
        auto& owner = fused::recording_graph_owner();
        if (owner) NN_FAIL("begin_expr: 嵌套录制（已有未结束的 begin_expr）");
        owner = std::make_unique<ExprGraph>();
        owner->owner = weak_from_this().lock();
        return {};
    }

    [[nodiscard]] Result<void> end_expr() override
    {
        auto& owner = fused::recording_graph_owner();
        // 录制图是 thread_local：跨线程 end_expr 看不到别人线程的 begin_expr。
        // 原实现静默 return {} ⇒ 录制图悬挂到下一次 begin_expr 才以"嵌套录制"
        // 报错（现场丢失）。这里明确报错，把"begin/end 必须同线程配对"变成
        // 可诊断事实。
        if (!owner)
            NN_FAIL("GpuEngine::end_expr: 当前线程没有进行中的 begin_expr"
                    "（begin_expr/end_expr 必须同线程严格配对）");
        ExprGraph g = std::move(*owner);
        owner.reset();
        return execute_fused_graph(g);
    }

    // ── 显存回收（L2）：end_batch 之后归还完全空闲的内存池底材 ──────
    [[nodiscard]] Result<void> release_idle_pool_blocks() override
    {
        NN_PROF_OP("release_idle_pool_blocks");
        return backend_->release_idle_pool_blocks();
    }

    // ── 显存池统计（L2 仪器化） ──────────────────────────────────────
    // 同时上报持久池（参数/权重）与瞬态池（激活/临时），否则只报持久池会
    // 掩盖瞬态池的占用，导致"拆分后显存不降反升"的误判。
    [[nodiscard]] std::string pool_stats() const override
    {
        const std::string p = backend_->memory_pool().pool_debug_stats().to_string();
        const std::string t = backend_->transient_pool().pool_debug_stats().to_string();
        // pending：已析构但因帧未 reap 而尚未归还池的字节（显存峰值归因的
        // 关键缺口——"Tensor 置空但 live 不降"的那部分账）。
        const std::string pd =
            " pending=" + std::to_string(backend_->pending_destroy_bytes() / (1024 * 1024)) + "MB";
        return "persist{" + p + "} transient{" + t + "}" + pd;
    }

    // ── activation offload slab（持久复用缓冲） ───────────────────────
    [[nodiscard]] Result<Tensor> create_offload_buffer_impl(std::size_t bytes) override
    {
        // #P8：slab 以 f32 视图持有一块原始字节缓冲（capacity = bytes）。
        // create_host_visible_empty 的单位是 f32 元素（内部 4 B 对齐），故按
        // 4 字节字向上取整分配——f16 激活（2 B/元素）同样落在这块缓冲里。
        // 容量口径统一见 ComputeEngine::kOffloadSlabElemBytes（#M2/#G1）。
        auto g = GpuTensor::create_host_visible_empty(
            1, (bytes + kOffloadSlabElemBytes - 1) / kOffloadSlabElemBytes, *backend_);
        NN_TRY_CHECK(g);
        return Tensor::from_gpu(std::move(*g));
    }

    // 把 src 按**自身精度**复制到 buffer 的 byte_offset 处（录制式，#P8）
    [[nodiscard]] Result<void> offload_save_impl(const Tensor& buffer, std::size_t byte_offset,
                                                 const Tensor& src) override
    {
        if (src.is_cpu()) return {};
        const std::size_t elem_bytes = (src.precision() == Precision::F16) ? 2u : 4u;
        const VkDeviceSize size = static_cast<VkDeviceSize>(src.size() * elem_bytes);
        // #M2：容量校验必须由引擎做（不能靠调用方的 8B 对齐+容量记账）。
        // 越界直接 fail-fast，否则 vkCmdCopyBuffer 写到子分配之外 = GPU 越界。
        const std::size_t cap = buffer.size() * kOffloadSlabElemBytes;
        if (byte_offset > cap || size > cap - byte_offset)
            NN_FAIL("offload_save: 写入越界（byte_offset=" + std::to_string(byte_offset) +
                    ", size=" + std::to_string(size) + ", slab 容量=" + std::to_string(cap) +
                    " 字节）");
        const VkBuffer sbuf =
            (src.precision() == Precision::F16)
                ? gpu_vk_impl(src.gpu_tensor<Precision::F16>().buffer())
                : gpu_vk_impl(src.gpu_tensor<Precision::F32>().buffer());
        return backend_->copy_buffer_region_gpu(
            sbuf, 0, gpu_vk_impl(buffer.gpu_tensor().buffer()),
            static_cast<VkDeviceSize>(byte_offset), size);
    }

    // 从 buffer 的 byte_offset 处复制 rows×cols 到新 GPU tensor（精度 P）（录制式）
    [[nodiscard]] Result<Tensor> offload_restore_impl(const Tensor& buffer, std::size_t byte_offset,
                                                      std::size_t rows, std::size_t cols,
                                                      Precision P) override
    {
        const std::size_t elem_bytes = (P == Precision::F16) ? 2u : 4u;
        const VkDeviceSize size = static_cast<VkDeviceSize>(rows * cols * elem_bytes);
        // #M2：读源容量同样要校验（越界读同样是 GPU 端 UB），在分配 dst 前先拒绝。
        const std::size_t cap = buffer.size() * kOffloadSlabElemBytes;
        if (byte_offset > cap || size > cap - byte_offset)
            NN_FAIL("offload_restore: 读取越界（byte_offset=" + std::to_string(byte_offset) +
                    ", size=" + std::to_string(size) + ", slab 容量=" + std::to_string(cap) +
                    " 字节）");
        VkBuffer dst_buf = VK_NULL_HANDLE;
        Tensor out;
        if (P == Precision::F16)
        {
            NN_TRY(r16, GpuTensorF16::create_empty(rows, cols, *backend_));
            dst_buf = gpu_vk_impl(r16->buffer());
            out = Tensor::from_gpu(std::move(*r16));
        }
        else
        {
            NN_TRY(r32, GpuTensor::create_empty(rows, cols, *backend_));
            dst_buf = gpu_vk_impl(r32->buffer());
            out = Tensor::from_gpu(std::move(*r32));
        }
        auto r = backend_->copy_buffer_region_gpu(gpu_vk_impl(buffer.gpu_tensor().buffer()),
                                                  static_cast<VkDeviceSize>(byte_offset), dst_buf, 0,
                                                  size);
        NN_TRY_CHECK(r);
        return out;
    }

    // ── 异步标量回读───────────────────────────────────────────
    // 把 (1,1) F32 标量排入一次 D2H 拷贝并提交，不等待。调用时机：产出该
    // 标量的主帧（flush_batch/end_batch）已提交之后——同队列 FIFO 保证拷贝
    // 执行在生产命令之后。poll 非阻塞，就绪即取（见 compute_engine.hpp）。
    [[nodiscard]] Result<void> submit_scalar_readback_impl(std::size_t slot,
                                                           const Tensor& t) override
    {
        if (t.is_cpu()) NN_FAIL("submit_scalar_readback: 期望 GPU 张量");
        if (t.precision() != Precision::F32)
            NN_FAIL("submit_scalar_readback: 仅支持 F32 标量（调用方先 cast）");
        return backend_->submit_scalar_readback(slot, gpu_vk_impl(t.gpu_tensor().buffer()));
    }

    [[nodiscard]] Result<bool> poll_scalar_readback(std::size_t slot, Scalar& out) override
    {
        float v = 0.0f;
        auto r = backend_->poll_scalar_readback(slot, v);
        NN_TRY_CHECK(r);
        out = static_cast<Scalar>(v);
        return *r;
    }

    [[nodiscard]] std::size_t scalar_readback_slots() const override
    { return GpuBackend::scalar_readback_slot_count(); }

    // ── 中点刷新：提交当前 command buffer 并开始新的录制 ──
    // 用于拆分大 batch（如 forward 与 backward 之间），防 TDR。
    [[nodiscard]] Result<void> flush_batch() override
    {
        NN_PROF_OP("flush_batch");
        return backend_->flush_batch();
    }

    // ══════════════════════════════════════════════════════════════════════
    // 张量工厂（纯 GPU：全部创建/上传为 GPU Tensor）
    // ══════════════════════════════════════════════════════════════════════
    // 统一接口：create_tensor / from_matrix / to_matrix（§6.4, §6.5）
    // P 由调用方显式指定（§8.5）：无隐式推导，无 Auto
    // ══════════════════════════════════════════════════════════════════════

    // RV-02：分配失败原因上抛（旧形态 `if (!r) return Tensor();` 把显存/设备
    // 错误整块丢弃，调用方只能看到"空张量"）。
    [[nodiscard]] Result<Tensor> create_tensor_impl(std::size_t rows, std::size_t cols,
                                                    Precision P) override
    {
        if (P == Precision::F16)
        {
            NN_TRY(r, GpuTensorF16::create_empty(rows, cols, *backend_));
            return Tensor::from_gpu(std::move(*r));
        }
        NN_TRY(r, GpuTensor::create_empty(rows, cols, *backend_));
        return Tensor::from_gpu(std::move(*r));
    }

    [[nodiscard]] Result<Tensor> from_matrix_impl(const Matrix& m, Precision P) override
    {
        if (P == Precision::F16)
        {
            MatrixT<Precision::F16> m16(m.rows(), m.cols());
            const auto src = m.span();
            auto dst = m16.span();
            for (std::size_t i = 0; i < src.size(); ++i)
                dst[i] = src[i];
            auto r = GpuTensorF16::from_matrix(m16, *backend_);
            NN_TRY_CHECK(r);
            return Tensor::from_gpu(std::move(*r));
        }
        auto r = GpuTensor::from_matrix(m, *backend_);
        NN_TRY_CHECK(r);
        return Tensor::from_gpu(std::move(*r));
    }

    [[nodiscard]] Result<Matrix> to_matrix(const Tensor& t, Precision P = Precision::F32) override
    {
        NN_PROF_OP("to_matrix");
        if (t.is_cpu())
        {
            if (t.precision() == Precision::F16 && P == Precision::F32)
            {
                const auto& m16 = t.cpu_matrix<Precision::F16>();
                Matrix m32(m16.rows(), m16.cols());
                const auto src = m16.span();
                auto dst = m32.span();
                for (std::size_t i = 0; i < src.size(); ++i)
                    dst[i] = static_cast<float>(src[i]);
                return m32;
            }
            return Matrix(t.cpu_matrix());
        }
        // GPU tensor → drain + download
        if (t.valid() && t.device_domain_() != domain())
            NN_FAIL("to_matrix: 张量不在本 GPU 引擎的设备域（跨 backend 请先用"
                    "创建它的引擎 to_matrix 落宿主）");
        if (in_batch())
        {
            auto r = end_batch();
            NN_TRY_CHECK(r);
            auto rw = backend_->wait_in_flight();
            NN_TRY_CHECK(rw);
            auto rb = begin_batch();
            NN_TRY_CHECK(rb);
        }
        if (t.precision() == Precision::F16)
        {
            auto m16_r = t.gpu_tensor<Precision::F16>().to_matrix(*backend_);
            NN_TRY_CHECK(m16_r);
            if (P == Precision::F32)
            {
                Matrix m32(m16_r->rows(), m16_r->cols());
                const auto src = m16_r->span();
                auto dst = m32.span();
                for (std::size_t i = 0; i < src.size(); ++i)
                    dst[i] = static_cast<float>(src[i]);
                return m32;
            }
        }
        return t.gpu_tensor().to_matrix(*backend_);
    }

    // ── cast 原语（§7.5，统一接口；GPU 原生转换，无 PCIe 往返）─────────────
    [[nodiscard]] Result<Tensor> cast_impl(const Tensor& src, Precision dst) override
    {
        if (src.precision() == dst) return src;

        // 保留精度 Pair（BF16/F64 Phase 1 未实现）清晰报错
        NN_TRY(pcheck, check_precision_supported(src.precision()));
        NN_TRY(pcheck2, check_precision_supported(dst));
        if (src.is_gpu())
        {
            const std::size_t count = src.rows() * src.cols();
            if (src.precision() == Precision::F16 && dst == Precision::F32)
            {
                // f16 GPU → f32 GPU（升 cast，精确无损，GPU kernel）
                auto dst_gpu = GpuTensor::create_empty(src.rows(), src.cols(), *backend_);
                NN_TRY_CHECK(dst_gpu);
                auto r = backend_->cast_gpu(src.gpu_tensor<Precision::F16>().buffer(),
                                            dst_gpu->buffer(), count, /*kind=0*/ 0u);
                NN_TRY_CHECK(r);
                return Tensor::from_gpu(std::move(*dst_gpu));
            }
            if (src.precision() == Precision::F32 && dst == Precision::F16)
            {
                // f32 GPU → f16 GPU（降 cast，round-half-to-even，GPU kernel）
                // create_f16_tensor 保证奇数元素 count 时 word 写入不越界。
                auto dst_gpu = backend_->create_f16_tensor(src.rows(), src.cols());
                NN_TRY_CHECK(dst_gpu);
                auto r = backend_->cast_gpu(src.gpu_tensor().buffer(), dst_gpu->buffer(), count,
                                            /*kind=1*/ 1u);
                NN_TRY_CHECK(r);
                return Tensor::from_gpu(std::move(*dst_gpu));
            }
            NN_FAIL("cast: unsupported precision conversion");
        }
        // v3：GPU 引擎的 cast 只接受本引擎域内的 GPU 张量（CPU 数据必须先
        // import 成 GPU 张量）。原 CPU 回退分支已删除——跨域在入口即被拒。
        NN_FAIL("cast: 操作数不在本 GPU 引擎的设备域（CPU 张量请先 import）");
    }

    // ── cast_into / copy_into（多精度适配层的"写回原存储"路径，§6.5）──────
    // 与 cast 的区别是**落点**：保留 dst 的对象身份与底层 buffer（in-place
    // 语义必需——若替换 dst 对象，其它持有同一张量句柄的缓存会静默失联）。
    // GPU 路径用 vkCmdCopyBuffer / cast 原语直接写 dst 的既有 buffer，无分配。
    [[nodiscard]] Result<void> copy_into(Tensor& dst, const Tensor& src) override
    {
        NN_PROF_OP("copy_into");
        WriteEpochGuard wg{dst};
        if (dst.rows() != src.rows() || dst.cols() != src.cols())
            NN_FAIL("copy_into: shape mismatch");
        if (dst.precision() != src.precision()) NN_FAIL("copy_into: precision mismatch");
        // v3：本引擎的 copy_into 只接受本域内的 GPU 张量（CPU 数据先 import）
        if (dst.device_domain_() != domain() || src.device_domain_() != domain())
            NN_FAIL("copy_into: 操作数不在本 GPU 引擎的设备域");
        const std::size_t count = dst.rows() * dst.cols();
        if (dst.precision() == Precision::F16)
            return backend_->copy_buffer_gpu(gpu_vk_impl(src.gpu_tensor<Precision::F16>().buffer()),
                                             gpu_vk_impl(dst.gpu_tensor<Precision::F16>().buffer()),
                                             static_cast<VkDeviceSize>(count * 2u));
        return backend_->copy_buffer_gpu(gpu_vk_impl(src.gpu_tensor().buffer()),
                                         gpu_vk_impl(dst.gpu_tensor().buffer()),
                                         static_cast<VkDeviceSize>(count * sizeof(float)));
    }

    [[nodiscard]] Result<void> cast_into(const Tensor& src, Tensor& dst) override
    {
        NN_PROF_OP("cast_into");
        WriteEpochGuard wg{dst};
        if (dst.rows() != src.rows() || dst.cols() != src.cols())
            NN_FAIL("cast_into: shape mismatch");
        if (src.precision() == dst.precision()) return copy_into(dst, src);
        // v3：本引擎的 cast_into 只接受本域内 GPU 张量
        if (dst.device_domain_() != domain() || src.device_domain_() != domain())
            NN_FAIL("cast_into: 操作数不在本 GPU 引擎的设备域");
        const std::size_t count = dst.rows() * dst.cols();
        if (src.precision() == Precision::F16 && dst.precision() == Precision::F32)
            return backend_->cast_gpu(src.gpu_tensor<Precision::F16>().buffer(),
                                      dst.gpu_tensor().buffer(), count, /*kind=*/0u);
        if (src.precision() == Precision::F32 && dst.precision() == Precision::F16)
            return backend_->cast_gpu(src.gpu_tensor().buffer(),
                                      dst.gpu_tensor<Precision::F16>().buffer(), count,
                                      /*kind=*/1u);
        NN_FAIL("cast_into: unsupported precision conversion");
    }

    [[nodiscard]] Result<void> copy_from_impl(Tensor& dst, const Matrix& src,
                                              bool fresh_dst) override
    {
        if (dst.rows() != src.rows() || dst.cols() != src.cols())
            NN_FAIL("copy_from: shape mismatch");
        if (dst.device_domain_() != domain()) NN_FAIL("copy_from: dst 不在本 GPU 引擎的设备域");
        // ── 非 fresh（既有张量）：必须先 drain（提交不等待）──────────────
        // dst 是当前帧已引用的既有 buffer——当前帧尚未提交，若先把上传提交
        // 到队列，GPU 会先执行上传、后执行当前帧命令 → 本帧后续对 dst 的
        // 写入覆盖上传数据。故：end_batch 提交当前帧 → wait_in_flight 等完 →
        // 新帧开始录制 → 上传（独立提交）。
        //
        // ── fresh（全新张量，upload_span 的 create→write 相邻形态）：整段
        // drain 跳过（本条热路径的历史画像 copy_from 121 次 / 步 = 全排空，
        // 见 docs/history）。安全性：
        //   · 反向依赖不存在——全新 buffer 无任何已录/在飞命令引用它
        //     （池化复用只在写出方帧 fence 信号、区间归还之后才可能发生，
        //     见 flush_pending_destroys）；
        //   · 正向依赖（本帧及后续帧读 dst）显式保证——upload_blocking 的
        //     时间线信号量 + 各帧 submit 处 collect_staging_waits 等待
        //     （与 from_matrix 同一条既有机制）。
        if (in_batch() && !fresh_dst)
        {
            auto r = end_batch();
            NN_TRY_CHECK(r);
            auto rw = backend_->wait_in_flight();
            NN_TRY_CHECK(rw);
            auto rb = begin_batch();
            NN_TRY_CHECK(rb);
        }
        return backend_->upload_blocking(dst.gpu_tensor(), src.span());
    }

    [[nodiscard]] Result<Tensor> clone_impl(const Tensor& src) override
    {
        // v3：只接受本域内 GPU 张量（CPU 数据先 import；原 CPU 回退已删除）
        if (src.device_domain_() != domain()) NN_FAIL("clone: 操作数不在本 GPU 引擎的设备域");
        if (src.precision() == Precision::F16)   // 原生 f16 字节拷贝（后端模板化实现）
        {
            auto r16 = backend_->clone_gpu(src.gpu_tensor<Precision::F16>());
            NN_TRY_CHECK(r16);
            return Tensor::from_gpu(std::move(*r16));
        }
        auto r = backend_->clone_gpu(src.gpu_tensor());
        NN_TRY_CHECK(r);
        return Tensor::from_gpu(std::move(*r));
    }

    // ── 行切片：GPU 内拷贝连续行区间 ──
    [[nodiscard]] Result<Tensor> slice_rows_impl(const Tensor& src, std::size_t start_row,
                                                 std::size_t count) override
    {
        auto src_gpu = import(src);
        NN_TRY_CHECK(src_gpu);
        if (src_gpu->precision() == Precision::F16)   // 原生 f16 行切片（纯字节拷贝）
        {
            auto r16 =
                backend_->slice_rows_gpu(src_gpu->gpu_tensor<Precision::F16>(), start_row, count);
            NN_TRY_CHECK(r16);
            return Tensor::from_gpu(std::move(*r16));
        }
        auto r = backend_->slice_rows_gpu(src_gpu->gpu_tensor(), start_row, count);
        NN_TRY_CHECK(r);
        return Tensor::from_gpu(std::move(*r));
    }

    // ── 行插入：GPU 内就地写入连续行区间 ──
    // 纯 GPU 架构：dst 必须为 GPU Tensor，传 CPU Tensor 直接报错（无 CPU 回退路径）。
    [[nodiscard]] Result<void> insert_rows_impl(Tensor& dst, std::size_t dst_start_row,
                                                const Tensor& src) override
    {
        if (dst.is_cpu()) NN_FAIL("insert_rows: dst must be GPU tensor in pure-GPU architecture");
        auto src_gpu = import(src);
        NN_TRY_CHECK(src_gpu);
        if (dst.precision() == Precision::F16)   // 原生 f16 行插入（纯字节拷贝）
            return backend_->insert_rows_gpu(dst.gpu_tensor<Precision::F16>(), dst_start_row,
                                             src_gpu->gpu_tensor<Precision::F16>());
        return backend_->insert_rows_gpu(dst.gpu_tensor(), dst_start_row, src_gpu->gpu_tensor());
    }

    // ── gather_rows: 按 indices 从 table 中按行查表 ──
    // GPU-native 实现：全程在 GPU 执行，无 PCIe 传输。
    // indices 支持任意形状，按 flat 遍历所有元素。
    [[nodiscard]] Result<Tensor> gather_rows_impl(const Tensor& table,
                                                  const Tensor& indices) override
    {
        auto tbl_gpu = import(table);
        NN_TRY_CHECK(tbl_gpu);
        auto idx_gpu = import(indices);
        NN_TRY_CHECK(idx_gpu);
        // Phase C1 f16 直读直写（table/out f16；indices 恒 f32——行号整数值
        // 精度要求）。无 f16 pipeline → 引擎内边界 cast 回退。
        if (tbl_gpu->precision() == Precision::F16)
        {
            if (backend_->has(PipelineId::gather_f16) && idx_gpu->precision() == Precision::F32)
            {
                auto r = backend_->gather_gpu(f16_view(*tbl_gpu), idx_gpu->gpu_tensor(),
                                              /*f16_io=*/true);
                NN_TRY_CHECK(r);
                return Tensor::from_gpu(GpuTensorF16(r->shared_buffer(), r->rows(), r->cols()));
            }
            auto t32 = cast(*tbl_gpu, Precision::F32);
            NN_TRY_CHECK(t32);
            std::optional<Tensor> i32;
            if (idx_gpu->precision() != Precision::F32)
            {
                auto c = cast(*idx_gpu, Precision::F32);
                NN_TRY_CHECK(c);
                i32 = std::move(*c);
            }
            const Tensor& idx_ref = i32 ? *i32 : *idx_gpu;
            auto r = backend_->gather_gpu(t32->gpu_tensor(), idx_ref.gpu_tensor());
            NN_TRY_CHECK(r);
            Tensor out32 = Tensor::from_gpu(std::move(*r));
            return cast(std::move(out32), Precision::F16);
        }
        auto r = backend_->gather_gpu(tbl_gpu->gpu_tensor(), idx_gpu->gpu_tensor());
        NN_TRY_CHECK(r);
        return Tensor::from_gpu(std::move(*r));
    }

    // ── scatter_add_rows: 按 indices 把 grad 的行原子累加到 dst ──
    // GPU-native 实现：使用 CAS 循环实现 float atomicAdd，无 PCIe 传输。
    // Phase C2b：dst/grad 均 f16 且 f16 pipeline 可用 → 打包 half CAS 原生
    // （dst 每 word 2 个 half、grad half 直读；indices 恒 float），免去为
    // dst/grad 物化 f32 副本。任一非 f16 或无 pipeline → 引擎内边界 cast
    // 回退（正确性不变）。
    [[nodiscard]] Result<void> scatter_add_rows_impl(Tensor& dst, const Tensor& indices,
                                                     const Tensor& grad) override
    {
        if (!dst.is_gpu()) NN_FAIL("scatter_add_rows: dst must be GPU tensor");
        auto idx_gpu = import(indices);
        NN_TRY_CHECK(idx_gpu);
        auto grad_gpu = import(grad);
        NN_TRY_CHECK(grad_gpu);
        const bool f16_io = dst.precision() == Precision::F16 &&
                            grad_gpu->precision() == Precision::F16 &&
                            backend_->has(PipelineId::scatter_add_f16);
        if (f16_io)
        {
            GpuTensor dst_ref = f16_view(dst);       // 局部保活（禁 &临时）
            const GpuTensor grad_ref = f16_view(*grad_gpu);
            return backend_->scatter_add_gpu(dst_ref, idx_gpu->gpu_tensor(), grad_ref,
                                             /*f16_io=*/true);
        }
        if (dst.precision() == Precision::F16 || grad_gpu->precision() == Precision::F16)
        {
            // 引擎内 cast 回退：dst 若 f16 → cast_into 写回原存储（§8.3）
            auto d32 = (dst.precision() == Precision::F16) ? cast(dst, Precision::F32)
                                                           : Result<Tensor>(dst);
            NN_TRY_CHECK(d32);
            auto g32 = (grad_gpu->precision() == Precision::F16) ? cast(*grad_gpu, Precision::F32)
                                                                 : Result<Tensor>(*grad_gpu);
            NN_TRY_CHECK(g32);
            auto g32_gpu = import(*g32);
            NN_TRY_CHECK(g32_gpu);
            auto r = backend_->scatter_add_gpu(d32->gpu_tensor(), idx_gpu->gpu_tensor(),
                                               g32_gpu->gpu_tensor());
            NN_TRY_CHECK(r);
            if (dst.precision() == Precision::F16) return cast_into(*d32, dst);
            return {};
        }
        return backend_->scatter_add_gpu(dst.gpu_tensor(), idx_gpu->gpu_tensor(),
                                         grad_gpu->gpu_tensor());
    }

private:
    // batch 模式查询（内部使用，to_matrix/from_matrix 需检查）
    [[nodiscard]] bool in_batch() const noexcept { return backend_->in_batch(); }

public:
    // ── 3D 维度转置：(M, B, N) ↔ (B, M, N) ──
    [[nodiscard]] Result<Tensor> rearrange_3d_impl(const Tensor& x, std::size_t M, std::size_t B,
                                                   std::size_t N, bool inverse) override
    {
        auto x_gpu = import(x);
        NN_TRY_CHECK(x_gpu);
        // Phase C1 f16 直读直写；无 f16 pipeline → 引擎内边界 cast 回退
        if (x_gpu->precision() == Precision::F16)
        {
            if (backend_->has(PipelineId::rearrange_3d_f16))
            {
                auto r = backend_->rearrange_3d_gpu(
                    f16_view(*x_gpu), static_cast<uint32_t>(M), static_cast<uint32_t>(B),
                    static_cast<uint32_t>(N), inverse ? 1u : 0u, /*f16_io=*/true);
                NN_TRY_CHECK(r);
                return Tensor::from_gpu(GpuTensorF16(r->shared_buffer(), r->rows(), r->cols()));
            }
            auto x32 = cast(*x_gpu, Precision::F32);
            NN_TRY_CHECK(x32);
            auto r = backend_->rearrange_3d_gpu(x32->gpu_tensor(), static_cast<uint32_t>(M),
                                                static_cast<uint32_t>(B), static_cast<uint32_t>(N),
                                                inverse ? 1u : 0u);
            NN_TRY_CHECK(r);
            Tensor out32 = Tensor::from_gpu(std::move(*r));
            return cast(std::move(out32), Precision::F16);
        }
        auto r = backend_->rearrange_3d_gpu(x_gpu->gpu_tensor(), static_cast<uint32_t>(M),
                                            static_cast<uint32_t>(B), static_cast<uint32_t>(N),
                                            inverse ? 1u : 0u);
        NN_TRY_CHECK(r);
        return Tensor::from_gpu(std::move(*r));
    }

    // ── 矩阵转置：A (R, C) → out (C, R) ──
    [[nodiscard]] Result<Tensor> transpose_impl(const Tensor& A) override
    {
        auto a_gpu = import(A);
        NN_TRY_CHECK(a_gpu);
        // Phase C1 f16 直读直写；无 f16 pipeline（设备能力不足）→ 引擎内
        // 边界 cast 回退（f16→f32→算→f16），正确性不变——绝不把 f16 buffer
        // 绑到 f32 pipeline（静默错值）。
        if (a_gpu->precision() == Precision::F16)
        {
            if (backend_->has(PipelineId::transpose_f16))
            {
                auto r = backend_->transpose_gpu(f16_view(*a_gpu), /*f16_io=*/true);
                NN_TRY_CHECK(r);
                return Tensor::from_gpu(GpuTensorF16(r->shared_buffer(), r->rows(), r->cols()));
            }
            auto a32 = cast(*a_gpu, Precision::F32);
            NN_TRY_CHECK(a32);
            auto r32 = backend_->transpose_gpu(a32->gpu_tensor());
            NN_TRY_CHECK(r32);
            Tensor out32 = Tensor::from_gpu(std::move(*r32));
            return cast(std::move(out32), Precision::F16);
        }
        auto r = backend_->transpose_gpu(a_gpu->gpu_tensor());
        NN_TRY_CHECK(r);
        return Tensor::from_gpu(std::move(*r));
    }

    // ── 卷积/池化窗口展开（纯数据搬运，无算法；契约见 compute_engine.hpp）──
    [[nodiscard]] Result<Tensor> im2col_impl(const Tensor& x, std::size_t C, std::size_t H,
                                             std::size_t W, std::size_t k, std::size_t stride,
                                             std::size_t pad, std::size_t OH,
                                             std::size_t OW) override
    {
        auto x_gpu = import(x);
        NN_TRY_CHECK(x_gpu);
        // Phase C2d：f16 输入/输出 → f16 窗口变体（in/out 均 half，语义
        // "输出精度 = 源精度" §8.4）；无 pipeline → 引擎内边界 cast 回退
        //（绝不把 f16 buffer 绑到 f32 pipeline；CNN 路径与 move_ 直通保障）
        if (x_gpu->precision() == Precision::F16)
        {
            if (backend_->has(PipelineId::im2col_f16))
            {
                auto r = backend_->im2col_gpu(f16_view(*x_gpu), C, H, W, k, stride, pad, OH, OW,
                                              /*f16_io=*/true);
                NN_TRY_CHECK(r);
                return Tensor::from_gpu(GpuTensorF16(r->shared_buffer(), r->rows(), r->cols()));
            }
            auto x32 = cast(*x_gpu, Precision::F32);
            NN_TRY_CHECK(x32);
            auto r = backend_->im2col_gpu(x32->gpu_tensor(), C, H, W, k, stride, pad, OH, OW);
            NN_TRY_CHECK(r);
            Tensor out32 = Tensor::from_gpu(std::move(*r));
            return cast(std::move(out32), Precision::F16);
        }
        auto r = backend_->im2col_gpu(x_gpu->gpu_tensor(), C, H, W, k, stride, pad, OH, OW);
        NN_TRY_CHECK(r);
        return Tensor::from_gpu(std::move(*r));
    }

    [[nodiscard]] Result<Tensor> col2im_impl(const Tensor& col, std::size_t C, std::size_t H,
                                             std::size_t W, std::size_t k, std::size_t stride,
                                             std::size_t pad, std::size_t OH,
                                             std::size_t OW) override
    {
        auto c_gpu = import(col);
        NN_TRY_CHECK(c_gpu);
        // Phase C2d：同 im2col——f16 窗口变体优先，无 pipeline 引擎内回退
        if (c_gpu->precision() == Precision::F16)
        {
            if (backend_->has(PipelineId::col2im_f16))
            {
                auto r = backend_->col2im_gpu(f16_view(*c_gpu), C, H, W, k, stride, pad, OH, OW,
                                              /*f16_io=*/true);
                NN_TRY_CHECK(r);
                return Tensor::from_gpu(GpuTensorF16(r->shared_buffer(), r->rows(), r->cols()));
            }
            auto c32 = cast(*c_gpu, Precision::F32);
            NN_TRY_CHECK(c32);
            auto r = backend_->col2im_gpu(c32->gpu_tensor(), C, H, W, k, stride, pad, OH, OW);
            NN_TRY_CHECK(r);
            Tensor out32 = Tensor::from_gpu(std::move(*r));
            return cast(std::move(out32), Precision::F16);
        }
        auto r = backend_->col2im_gpu(c_gpu->gpu_tensor(), C, H, W, k, stride, pad, OH, OW);
        NN_TRY_CHECK(r);
        return Tensor::from_gpu(std::move(*r));
    }

    // ── 分组归约（契约见 compute_engine.hpp）──────────────────────────
    // Phase C2c：f16 输入 → f16 分组归约 pipeline（half 直读、f32 归约、
    // 输出 f32 (G,N)）；无 pipeline → 引擎内边界 cast 回退。
    [[nodiscard]] Result<Tensor> grouped_reduce_f16_or_cast_(const Tensor& x, std::size_t G,
                                                             std::size_t R, bool is_max)
    {
        auto x_gpu = import(x);
        NN_TRY_CHECK(x_gpu);
        if (x_gpu->precision() == Precision::F16)
        {
            if (backend_->has(PipelineId::group_reduce_f16))
            {
                auto r =
                    backend_->grouped_reduce_gpu(f16_view(*x_gpu), G, R, is_max, /*f16_in=*/true);
                NN_TRY_CHECK(r);
                return Tensor::from_gpu(std::move(*r));
            }
            auto x32 = cast(*x_gpu, Precision::F32);
            NN_TRY_CHECK(x32);
            auto x32_gpu = import(*x32);
            NN_TRY_CHECK(x32_gpu);
            auto r = backend_->grouped_reduce_gpu(x32_gpu->gpu_tensor(), G, R, is_max);
            NN_TRY_CHECK(r);
            return Tensor::from_gpu(std::move(*r));
        }
        auto r = backend_->grouped_reduce_gpu(x_gpu->gpu_tensor(), G, R, is_max);
        NN_TRY_CHECK(r);
        return Tensor::from_gpu(std::move(*r));
    }

    [[nodiscard]] Result<Tensor> grouped_reduce_sum_impl(const Tensor& x, std::size_t G,
                                                         std::size_t R,
                                                         Precision = Precision::F32) override
    { return grouped_reduce_f16_or_cast_(x, G, R, /*is_max=*/false); }

    [[nodiscard]] Result<Tensor> grouped_reduce_max_impl(const Tensor& x, std::size_t G,
                                                         std::size_t R,
                                                         Precision = Precision::F32) override
    { return grouped_reduce_f16_or_cast_(x, G, R, /*is_max=*/true); }

    // ══════════════════════════════════════════════════════════════════════
    // 矩阵级原语
    // ══════════════════════════════════════════════════════════════════════

    // ── 三值打包权重（P2 / F3）：打包字容器 + 量化 GEMM ────────────────────
    // 引擎私有能力：打包 pipeline 与 matmul_q pipeline 都在（缺任一 →
    // `matmul_q_impl` / `trit_pack_impl` 在此 fail-fast；对外事实见 describe()）。
    [[nodiscard]] Result<Tensor> trit_pack_impl(const Tensor& w, const Tensor& scale) override
    {
        const std::size_t rows = w.rows();
        const std::size_t wpr = nn::trit_pack::words_per_row(w.cols());
        auto out = GpuTensor::create_empty(rows, wpr, *backend_);
        NN_TRY_CHECK(out);
        Tensor dst = Tensor::from_gpu(std::move(*out));
        NN_TRY_CHECK(trit_pack_into_impl(w, scale, dst));
        return dst;
    }

    // 目标传递版（T1_58 显存修复 #L2）：打包结果写进既有容器，不分配新缓冲。
    [[nodiscard]] Result<void> trit_pack_into_impl(const Tensor& w, const Tensor& scale,
                                                   Tensor& dst) override
    {
        if (!(backend_->has(PipelineId::trit_pack) && backend_->has(PipelineId::matmul_q)))
            NN_FAIL("trit_pack_into: 该设备未创建三值打包 pipeline（docs 21 §5 P2）");
        if (!w.is_gpu() || !scale.is_gpu() || !dst.is_gpu())
            NN_FAIL("trit_pack_into: GPU 张量必需");
        if (w.precision() != Precision::F32 || scale.precision() != Precision::F32 ||
            dst.precision() != Precision::F32)
            NN_FAIL("trit_pack_into: 需要 f32 权重/尺度/容器");
        if (scale.rows() != w.rows() || scale.cols() != 1)
            NN_FAIL("trit_pack_into: τ 形状必须是 (rows,1)");
        const std::size_t rows = w.rows(), k = w.cols();
        const std::size_t wpr = nn::trit_pack::words_per_row(k);
        if (dst.rows() != rows || dst.cols() != wpr)
            NN_FAIL("trit_pack_into: 容器形状不匹配");
        return backend_->trit_pack_gpu(w.gpu_tensor().buffer(), scale.gpu_tensor().buffer(),
                                       dst.gpu_tensor().buffer(), static_cast<std::uint32_t>(rows),
                                       static_cast<std::uint32_t>(k),
                                       static_cast<std::uint32_t>(wpr));
    }

    [[nodiscard]] Result<Tensor> matmul_q_impl(const Tensor& packed, const Tensor& scale,
                                               const Tensor& B, bool transA, std::size_t out_rows,
                                               Precision P) override
    {
        if (!(backend_->has(PipelineId::trit_pack) && backend_->has(PipelineId::matmul_q)))
            NN_FAIL("matmul_q: 该设备未创建三值 GEMM pipeline（docs 21 §5 P2）");
        if (!packed.is_gpu() || !scale.is_gpu() || !B.is_gpu())
            NN_FAIL("matmul_q: GPU 张量必需（跨设备请先 import）");
        if (packed.precision() != Precision::F32) NN_FAIL("matmul_q: 打包容器必须是 f32 容器");
        // 形状推导与校验（与 CPU 实现同一口径）
        const std::size_t N = B.cols();
        const std::size_t M = transA ? out_rows : packed.rows();
        const std::size_t K = transA ? packed.rows() : B.rows();
        const std::size_t a_len = transA ? M : K;
        if (transA && out_rows == 0) NN_FAIL("matmul_q_t: out_rows 必须显式给出");
        if (B.rows() != K) NN_FAIL("matmul_q: B 行数 != 求和维 K");
        const std::size_t wpr = nn::trit_pack::words_per_row(a_len);
        if (wpr != packed.cols()) NN_FAIL("matmul_q: 打包容器列数与 (K/M) 不匹配");
        if (scale.rows() != packed.rows() || scale.cols() != 1)
            NN_FAIL("matmul_q: τ 形状必须与打包 A 的行数一致");

        // in-kernel f16 变体（B 与输出均 f16）：f16 profile 下的主路径 ——
        // 否则每个 BitLinear 层每步要多付两份 (·,batch) 的 f32 边界 cast 临时量。
        if (P == Precision::F16 && B.precision() == Precision::F16 &&
            backend_->has(PipelineId::matmul_q_f16))
        {
            auto out16 = GpuTensorF16::create_empty(M, N, *backend_);
            NN_TRY_CHECK(out16);
            NN_TRY(r, backend_->matmul_q_gpu(
                          packed.gpu_tensor().buffer(), scale.gpu_tensor().buffer(),
                          B.gpu_tensor<Precision::F16>().buffer(), out16->buffer(),
                          static_cast<std::uint32_t>(M), static_cast<std::uint32_t>(N),
                          static_cast<std::uint32_t>(K), transA ? 1u : 0u,
                          static_cast<std::uint32_t>(wpr), /*f16=*/true));
            return Tensor::from_gpu(std::move(*out16));
        }

        // f32 路径（B 为 f16 时先抬 f32；输出按 P 落回）
        // 注：to_f32 是基类**私有**助手 → 用公共 cast（同 matmul_impl 的口径）
        auto b32 = (B.precision() == Precision::F32) ? Result<Tensor>{B} : cast(B, Precision::F32);
        NN_TRY_CHECK(b32);
        auto out = GpuTensor::create_empty(M, N, *backend_);
        NN_TRY_CHECK(out);
        NN_TRY(r, backend_->matmul_q_gpu(
                      packed.gpu_tensor().buffer(), scale.gpu_tensor().buffer(),
                      b32->gpu_tensor().buffer(), out->buffer(), static_cast<std::uint32_t>(M),
                      static_cast<std::uint32_t>(N), static_cast<std::uint32_t>(K),
                      transA ? 1u : 0u, static_cast<std::uint32_t>(wpr), /*f16=*/false));
        Tensor res = Tensor::from_gpu(std::move(*out));
        if (P == Precision::F32) return res;
        // 静态工厂直构的 GPU 张量：设备域从存储派生（v3），直接进公共 cast
        // （同 CPU matmul_q_impl；17 §5 M6 的基类内部中转）。
        return cast(std::move(res), P);
    }

    // ── int8 激活量化（P3，W1.58A8）：quantize / unpack / matmul_q_i8 ──────
    // 语义见 compute_engine.hpp 公共入口；容器 = f32 张量承载原始字（与
    // trit_pack / matmul_q 同先例，不做浮点运算）。
    [[nodiscard]] Result<Tensor> quantize_i8_impl(const Tensor& x, const Tensor& scale,
                                                  bool col_scale, Precision P) override
    {
        if (!(backend_->has(PipelineId::quantize_i8)))
            NN_FAIL("quantize_i8: 该设备未创建 int8 量化 pipeline（docs 21 §5 P3）");
        if (!x.is_gpu() || !scale.is_gpu()) NN_FAIL("quantize_i8: GPU 张量必需");
        if (P != Precision::F32 || x.precision() != Precision::F32)
        {
            auto x32 =
                (x.precision() == Precision::F32) ? Result<Tensor>{x} : cast(x, Precision::F32);
            NN_TRY_CHECK(x32);
            auto s32 = (scale.precision() == Precision::F32)
                           ? Result<Tensor>{scale}
                           : cast(scale, Precision::F32);
            NN_TRY_CHECK(s32);
            return quantize_i8_impl(*x32, *s32, col_scale, Precision::F32);
        }
        if (scale.precision() != Precision::F32) NN_FAIL("quantize_i8: 尺度必须 f32");
        const std::size_t rows = x.rows(), cols = x.cols();
        if (col_scale && (scale.rows() != 1 || scale.cols() != cols))
            NN_FAIL("quantize_i8: 列尺度形状必须是 (1,N)（per-token）");
        if (!col_scale && (scale.rows() != rows || scale.cols() != 1))
            NN_FAIL("quantize_i8: 行尺度形状必须是 (M,1)");
        const std::size_t wpr = nn::i8_pack::words_per_row(cols);
        auto out = GpuTensor::create_empty(rows, wpr, *backend_);
        NN_TRY_CHECK(out);
        NN_TRY(r, backend_->quantize_i8_gpu(x.gpu_tensor().buffer(), scale.gpu_tensor().buffer(),
                                            out->buffer(), static_cast<std::uint32_t>(rows),
                                            static_cast<std::uint32_t>(cols),
                                            static_cast<std::uint32_t>(wpr), col_scale ? 1u : 0u));
        return Tensor::from_gpu(std::move(*out));
    }

    [[nodiscard]] Result<Tensor> unpack_i8_impl(const Tensor& packed, std::size_t cols) override
    {
        if (!(backend_->has(PipelineId::unpack_i8)))
            NN_FAIL("unpack_i8: 该设备未创建 int8 解包 pipeline（docs 21 §5 P3）");
        if (!packed.is_gpu()) NN_FAIL("unpack_i8: GPU 张量必需");
        if (packed.precision() != Precision::F32) NN_FAIL("unpack_i8: 容器必须 f32");
        const std::size_t rows = packed.rows();
        const std::size_t wpr = nn::i8_pack::words_per_row(cols);
        if (packed.cols() != wpr) NN_FAIL("unpack_i8: 容器列数 != words_per_row(cols)");
        auto out = GpuTensor::create_empty(rows, cols, *backend_);
        NN_TRY_CHECK(out);
        NN_TRY(r, backend_->unpack_i8_gpu(packed.gpu_tensor().buffer(), out->buffer(),
                                          static_cast<std::uint32_t>(rows),
                                          static_cast<std::uint32_t>(cols),
                                          static_cast<std::uint32_t>(wpr)));
        return Tensor::from_gpu(std::move(*out));
    }

    [[nodiscard]] Result<Tensor> matmul_q_i8_impl(const Tensor& packed_w, const Tensor& tau,
                                                  const Tensor& packed_x, const Tensor& sx,
                                                  Precision P) override
    {
        if (!(backend_->has(PipelineId::matmul_q_i8)))
            NN_FAIL("matmul_q_i8: 该设备未创建 int8 量化 GEMM pipeline（docs 21 §5 P3）");
        if (!packed_w.is_gpu() || !tau.is_gpu() || !packed_x.is_gpu() || !sx.is_gpu())
            NN_FAIL("matmul_q_i8: GPU 张量必需");
        if (packed_w.precision() != Precision::F32 || packed_x.precision() != Precision::F32 ||
            tau.precision() != Precision::F32 || sx.precision() != Precision::F32)
            NN_FAIL("matmul_q_i8: 需要 f32 容器/尺度");
        const std::size_t M = packed_w.rows();
        const std::size_t K = packed_x.rows();
        const std::size_t N = sx.cols();
        if (sx.rows() != 1) NN_FAIL("matmul_q_i8: 激活尺度必须是 (1,N)");
        if (tau.rows() != M || tau.cols() != 1) NN_FAIL("matmul_q_i8: τ 必须是 (M,1)");
        const std::size_t wpr_w = nn::trit_pack::words_per_row(K);
        const std::size_t wpr_q = nn::i8_pack::words_per_row(N);
        if (wpr_w != packed_w.cols()) NN_FAIL("matmul_q_i8: 权重容器列数与 K 不匹配");
        if (wpr_q != packed_x.cols()) NN_FAIL("matmul_q_i8: 激活容器列数与 N 不匹配");

        // in-kernel f16 输出变体（f16 profile 下免边界 cast）
        if (P == Precision::F16 && backend_->has(PipelineId::matmul_q_i8_f16))
        {
            auto out16 = GpuTensorF16::create_empty(M, N, *backend_);
            NN_TRY_CHECK(out16);
            NN_TRY(r, backend_->matmul_q_i8_gpu(
                          packed_w.gpu_tensor().buffer(), tau.gpu_tensor().buffer(),
                          packed_x.gpu_tensor().buffer(), sx.gpu_tensor().buffer(),
                          out16->buffer(), static_cast<std::uint32_t>(M),
                          static_cast<std::uint32_t>(N), static_cast<std::uint32_t>(K),
                          static_cast<std::uint32_t>(wpr_w), static_cast<std::uint32_t>(wpr_q),
                          /*f16=*/true));
            return Tensor::from_gpu(std::move(*out16));
        }

        auto out = GpuTensor::create_empty(M, N, *backend_);
        NN_TRY_CHECK(out);
        NN_TRY(r, backend_->matmul_q_i8_gpu(
                      packed_w.gpu_tensor().buffer(), tau.gpu_tensor().buffer(),
                      packed_x.gpu_tensor().buffer(), sx.gpu_tensor().buffer(), out->buffer(),
                      static_cast<std::uint32_t>(M), static_cast<std::uint32_t>(N),
                      static_cast<std::uint32_t>(K), static_cast<std::uint32_t>(wpr_w),
                      static_cast<std::uint32_t>(wpr_q), /*f16=*/false));
        Tensor res = Tensor::from_gpu(std::move(*out));
        if (P == Precision::F32) return res;
        return cast(std::move(res), P);
    }

    // ── 量化状态 AdamW 一步（AdamWInt8，领域原语，docs 21 §4.19）──────────
    // 语义与数值口径见抽象层公共入口；状态容器在设备侧就地读改写。
    [[nodiscard]] Result<void> quantized_adamw_step_impl(
        Tensor& param, const Tensor& grad, Tensor& m_packed, Tensor& v_packed,
        const Tensor& m_scale_old, const Tensor& m_scale_new, const Tensor& v_scale_old,
        const Tensor& v_scale_new, Scalar lr, Scalar beta1, Scalar beta2, Scalar eps, Scalar wd,
        Scalar inv_bc1, Scalar inv_bc2) override
    {
        if (!(backend_->has(PipelineId::quantized_adamw_step)))
            NN_FAIL("quantized_adamw_step: 该设备未创建量化状态 AdamW pipeline（docs 21 §4.19）");
        if (!param.is_gpu() || !grad.is_gpu() || !m_packed.is_gpu() || !v_packed.is_gpu() ||
            !m_scale_old.is_gpu() || !m_scale_new.is_gpu() || !v_scale_old.is_gpu() ||
            !v_scale_new.is_gpu())
            NN_FAIL("quantized_adamw_step: GPU 张量必需");
        if (param.precision() != Precision::F32 || grad.precision() != Precision::F32 ||
            m_packed.precision() != Precision::F32 || v_packed.precision() != Precision::F32 ||
            m_scale_old.precision() != Precision::F32 || m_scale_new.precision() != Precision::F32 ||
            v_scale_old.precision() != Precision::F32 || v_scale_new.precision() != Precision::F32)
            NN_FAIL("quantized_adamw_step: 参数/梯度/尺度/容器必须 f32");
        const std::size_t rows = param.rows(), cols = param.cols();
        if (grad.rows() != rows || grad.cols() != cols)
            NN_FAIL("quantized_adamw_step: 梯度形状不匹配");
        const std::size_t wpr = nn::i8_pack::words_per_row(cols);
        if (m_packed.rows() != rows || m_packed.cols() != wpr || v_packed.rows() != rows ||
            v_packed.cols() != wpr)
            NN_FAIL("quantized_adamw_step: 状态容器形状必须 (rows, ceil(cols/4))");
        if (m_scale_old.rows() != rows || m_scale_old.cols() != 1 || m_scale_new.rows() != rows ||
            m_scale_new.cols() != 1 || v_scale_old.rows() != rows || v_scale_old.cols() != 1 ||
            v_scale_new.rows() != rows || v_scale_new.cols() != 1)
            NN_FAIL("quantized_adamw_step: 尺度必须 (rows,1)");

        return backend_->quantized_adamw_step_gpu(
            param.gpu_tensor().buffer(), grad.gpu_tensor().buffer(),
            m_packed.gpu_tensor().buffer(), v_packed.gpu_tensor().buffer(),
            m_scale_old.gpu_tensor().buffer(), m_scale_new.gpu_tensor().buffer(),
            v_scale_old.gpu_tensor().buffer(), v_scale_new.gpu_tensor().buffer(),
            static_cast<std::uint32_t>(rows), static_cast<std::uint32_t>(cols),
            static_cast<std::uint32_t>(wpr), lr, beta1, Scalar{1} - beta1, beta2, Scalar{1} - beta2,
            eps, wd, inv_bc1, inv_bc2);
    }

    [[nodiscard]] Result<Tensor> matmul_impl(const Tensor& A, const Tensor& B, bool transA,
                                             bool transB, Precision P = Precision::F32) override
    {
        // 确保 A、B 在 GPU 上
        auto a_gpu = import(A);
        NN_TRY_CHECK(a_gpu);
        auto b_gpu = import(B);
        NN_TRY_CHECK(b_gpu);
        // ── F32 路径（现状，零改动）─────────────────────────────────────
        if (P == Precision::F32)
        {
            auto r = backend_->matmul_gpu(a_gpu->gpu_tensor(), b_gpu->gpu_tensor(),
                                          transA ? 1u : 0u, transB ? 1u : 0u);
            NN_TRY_CHECK(r);
            return Tensor::from_gpu(std::move(*r));
        }

        // ── F16 路径（Phase 2 in-kernel f16）──────────────────────────────
        // 两个操作数都是 f16 且设备支持 SSBO 16 位存储 → 走 **f16 存储版 GEMM**
        // （f16 直读 + f32 累加 + f16 写出），省掉"每个操作数一份整份 f32
        // 副本"的边界 cast——f32 副本正是训练 transient 膨胀的主因。小 N
        // （n_out≤8）走 f16 GEMV（C2e），大 N 走 f16 tiled；均不可用才回退
        // 边界 cast。
        if (P == Precision::F16)
        {
            const std::size_t n_out = transB ? B.rows() : B.cols();
            // 大 N：tiled f16；小 N（n_out≤8，GEMV 热路径）：gemv f16
            //（C2e：变体 pipeline 缺失时 f16_ok 为假 → 落到下方边界 cast）
            const bool f16_ok = a_gpu->precision() == Precision::F16 &&
                                b_gpu->precision() == Precision::F16 &&
                                ((n_out > 8 && backend_->has(PipelineId::matmul_tiled_f16)) ||
                                 (n_out <= 8 && backend_->has(PipelineId::matmul_gemv_f16)));
            if (f16_ok)
            {
                auto r = backend_->matmul_gpu(f16_view(*a_gpu), f16_view(*b_gpu), transA ? 1u : 0u,
                                              transB ? 1u : 0u, /*f16_io=*/true);
                NN_TRY_CHECK(r);
                return Tensor::from_gpu(GpuTensorF16(r->shared_buffer(), r->rows(), r->cols()));
            }
        }

        // ── F16 边界 cast 回退（无原生 f16 GEMM 时）──────────────────────
        // 边界 cast：f16 → f32（§7.5 cast 原语）
        if (P == Precision::F16)
        {
            // 边界 cast：f16 → f32（§7.5 cast 原语）
            auto a32 = cast(*a_gpu, Precision::F32);
            NN_TRY_CHECK(a32);
            auto b32 = cast(*b_gpu, Precision::F32);
            NN_TRY_CHECK(b32);
            auto a32_gpu = import(*a32);
            NN_TRY_CHECK(a32_gpu);
            auto b32_gpu = import(*b32);
            NN_TRY_CHECK(b32_gpu);
            // f32 matmul
            auto r = backend_->matmul_gpu(a32_gpu->gpu_tensor(), b32_gpu->gpu_tensor(),
                                          transA ? 1u : 0u, transB ? 1u : 0u);
            NN_TRY_CHECK(r);
            // cast 回 f16
            Tensor result_f32 = Tensor::from_gpu(std::move(*r));
            return cast(std::move(result_f32), Precision::F16);
        }

        NN_FAIL("matmul: unsupported precision");
    }

    // ── matmul + 行广播 bias：经 DSL 融合（单一事实源）──────────────
    // 走 eval_expr → AOT fusion shader 精确匹配（注册表命中优先）。同时让
    // scan_exprs dry-run 收集该"Linear 结构"spec，gen_fused 生成融合 kernel，
    // 使 GPU Linear::forward 全程在 GPU（基类 matmul_with_bias 的默认实现
    // 是 to_matrix/from_matrix 的 CPU 往返路径）。
    //
    // 多精度（Phase 2 in-kernel f16）：P 必须下传——f16 时**先试带类型变体**
    // （f16 直读 / f32 累加 / f16 写回，零边界 cast：Linear::forward 是 cast
    // 临时量最大的单一来源），无变体才在本函数内回退"抬 f32 → f32 融合 →
    // 落回"（**绝不把缺变体变成硬报错**，AOT 注册表原则只在 f32 结构缺失时适用）。
    [[nodiscard]] Result<Tensor> matmul_with_bias_impl(const Tensor& A, const Tensor& B,
                                                       const Tensor& bias, bool transA = false,
                                                       bool transB = false,
                                                       Precision P = Precision::F32) override
    {
        const std::size_t rows = transA ? A.cols() : A.rows();
        const std::size_t cols = transB ? B.rows() : B.cols();
        if (P == Precision::F32)
            return nn::dsl::compute(
                *this, nn::dsl::matmul(A, B, transA, transB, 1) + nn::dsl::row_broadcast(bias),
                rows, cols);

        auto [spec, inputs] = nn::dsl::to_expr_spec(nn::dsl::matmul(A, B, transA, transB, 1) +
                                                    nn::dsl::row_broadcast(bias));
        NN_TRY(v, nn::validate_expr_spec(spec, inputs.size()));
        // 变体选路与"无变体 → 边界 cast 回退"都在 eval_expr_impl 内部
        //（F2 修复后基类入口不再代劳）；这里只需把 P 下传。
        return eval_expr(spec, inputs, rows, cols, P);
    }

    // ── 批量矩阵乘法：按 batch 切分行块，单次 dispatch 处理所有 batch ──
    // alpha 在 shader 写出时一次完成（如注意力 1/sqrt(d_k) 缩放）
    // P: 计算精度（§8.1，同 matmul）
    [[nodiscard]] Result<Tensor> batched_matmul_impl(const Tensor& A, const Tensor& B,
                                                     std::size_t batch, bool transA, bool transB,
                                                     Scalar alpha,
                                                     Precision P = Precision::F32) override
    {
        auto a_gpu = import(A);
        NN_TRY_CHECK(a_gpu);
        auto b_gpu = import(B);
        NN_TRY_CHECK(b_gpu);
        // ── F32 路径（现状，零改动）─────────────────────────────────────
        if (P == Precision::F32)
        {
            auto r = backend_->batched_matmul_gpu(a_gpu->gpu_tensor(), b_gpu->gpu_tensor(),
                                                  static_cast<uint32_t>(batch), transA ? 1u : 0u,
                                                  transB ? 1u : 0u, static_cast<float>(alpha));
            NN_TRY_CHECK(r);
            return Tensor::from_gpu(std::move(*r));
        }

        // ── F16 路径（Phase 2 in-kernel f16）──────────────────────────────
        // 同 matmul：两个操作数都 f16 → f16 存储版 batch GEMM（零 f32 副本）。
        if (P == Precision::F16 && a_gpu->precision() == Precision::F16 &&
            b_gpu->precision() == Precision::F16 && backend_->has(PipelineId::batched_matmul_f16))
        {
            auto r = backend_->batched_matmul_gpu(
                f16_view(*a_gpu), f16_view(*b_gpu), static_cast<uint32_t>(batch), transA ? 1u : 0u,
                transB ? 1u : 0u, static_cast<float>(alpha), /*f16_io=*/true);
            NN_TRY_CHECK(r);
            return Tensor::from_gpu(GpuTensorF16(r->shared_buffer(), r->rows(), r->cols()));
        }

        // ── F16 边界 cast 回退（无原生 f16 GEMM 时）──────────────────────
        if (P == Precision::F16)
        {
            auto a32 = cast(*a_gpu, Precision::F32);
            NN_TRY_CHECK(a32);
            auto b32 = cast(*b_gpu, Precision::F32);
            NN_TRY_CHECK(b32);
            auto a32_gpu = import(*a32);
            NN_TRY_CHECK(a32_gpu);
            auto b32_gpu = import(*b32);
            NN_TRY_CHECK(b32_gpu);
            auto r = backend_->batched_matmul_gpu(a32_gpu->gpu_tensor(), b32_gpu->gpu_tensor(),
                                                  static_cast<uint32_t>(batch), transA ? 1u : 0u,
                                                  transB ? 1u : 0u, static_cast<float>(alpha));
            NN_TRY_CHECK(r);
            Tensor result_f32 = Tensor::from_gpu(std::move(*r));
            return cast(std::move(result_f32), Precision::F16);
        }

        NN_FAIL("batched_matmul: unsupported precision");
    }

    // A += B：真原地，直接写回 A 的 buffer（与 CpuEngine 语义一致）
    // 逐元素 kernel 每线程只读写自己下标一次，read-before-write 天然成立。
    // 免去新 buffer 分配 + 全量写出，消除优化器/梯度累积路径的分配风暴。
    [[nodiscard]] Result<void> add_inplace_impl(Tensor& A, const Tensor& B) override
    {
        if (A.rows() != B.rows() || A.cols() != B.cols()) NN_FAIL("add_inplace: shape mismatch");
        auto a_gpu = import(A);
        NN_TRY_CHECK(a_gpu);
        auto b_gpu = import(B);
        NN_TRY_CHECK(b_gpu);
        // Phase C1b：目标 A 是 f16 → f16 pipeline 原地直加（A 自身永不物化
        // f32 副本，A 的 f32 副本正是 in-place 路径的瞬态开销大头）。
        //   · B 同 f16 → 零副本直加；
        //   · B 是 f32（梯度混合精度）→ 只物化 B 的 f16 小副本再加；
        //   · 无 f16 pipeline → 引擎内回退（A 抬 f32 加完写回，正确性不变）。
        if (a_gpu->precision() == Precision::F16 && backend_->has(PipelineId::elementwise_v2_f16))
        {
            std::optional<Tensor> b16;
            if (b_gpu->precision() != Precision::F16)
            {
                auto c = cast(*b_gpu, Precision::F16);
                NN_TRY_CHECK(c);
                b16 = std::move(*c);
            }
            const GpuTensor& b_ref = b16 ? f16_view(*b16) : f16_view(*b_gpu);
            const GpuTensor a_ref = f16_view(*a_gpu);   // 局部保活（禁 &临时）
            const uint32_t count16 = static_cast<uint32_t>(A.rows() * A.cols());
            auto r = backend_->elementwise_v2_gpu(a_ref, &b_ref, nullptr, count16, 1u, 0u, 0u, 0u,
                                                  0.0f, 0.0f, 0.0f, &a_ref,
                                                  /*f16_io=*/true);  // BINARY, Add, 原地写回 A
            NN_TRY_CHECK(r);
            if (A.is_cpu()) A = std::move(*a_gpu);
            return {};
        }
        if (a_gpu->precision() == Precision::F16)
        {
            // 无 f16 pipeline：A 抬 f32 → 加 → 写回 A（边界 cast 路径内聚在引擎）
            auto a32 = cast(*a_gpu, Precision::F32);
            NN_TRY_CHECK(a32);
            std::optional<Tensor> b32;
            if (b_gpu->precision() != Precision::F32)
            {
                auto c = cast(*b_gpu, Precision::F32);
                NN_TRY_CHECK(c);
                b32 = std::move(*c);
            }
            const GpuTensor& b_ref2 = b32 ? b32->gpu_tensor() : b_gpu->gpu_tensor();
            const uint32_t cnt = static_cast<uint32_t>(A.rows() * A.cols());
            auto r = backend_->elementwise_v2_gpu(a32->gpu_tensor(), &b_ref2, nullptr, cnt, 1u, 0u,
                                                  0u, 0u, 0.0f, 0.0f, 0.0f, &a32->gpu_tensor());
            NN_TRY_CHECK(r);
            return cast_into(*a32, A);   // f32 结果写回 A 的 f16 原存储
        }

        // A 是 f32：B 若为 f16（直通后可能出现）→ 先抬 f32 再加（B 的边界
        // cast 只在引擎内发生，A 永不物化）
        if (b_gpu->precision() == Precision::F16)
        {
            auto b32 = cast(*b_gpu, Precision::F32);
            NN_TRY_CHECK(b32);
            const uint32_t cnt = static_cast<uint32_t>(A.rows() * A.cols());
            auto r = backend_->elementwise_v2_gpu(a_gpu->gpu_tensor(), &b32->gpu_tensor(), nullptr,
                                                  cnt, 1u, 0u, 0u, 0u, 0.0f, 0.0f, 0.0f,
                                                  &a_gpu->gpu_tensor());
            NN_TRY_CHECK(r);
            if (A.is_cpu()) A = std::move(*a_gpu);
            return {};
        }

        const uint32_t count = static_cast<uint32_t>(A.rows() * A.cols());
        auto r = backend_->elementwise_v2_gpu(a_gpu->gpu_tensor(), &b_gpu->gpu_tensor(), nullptr,
                                              count, 1u, 0u, 0u, 0u, 0.0f, 0.0f, 0.0f,
                                              &a_gpu->gpu_tensor());  // BINARY, Add, 原地写回 A
        NN_TRY_CHECK(r);
        // 原地模式下 A 的 buffer 已被更新；若 import 上传了新 Tensor（防御路径），替换 A
        if (A.is_cpu()) A = std::move(*a_gpu);
        return {};
    }

    // A *= s：真原地，直接写回 A 的 buffer（与 CpuEngine 语义一致）
    [[nodiscard]] Result<void> scale_inplace_impl(Tensor& A, Scalar s) override
    {
        auto a_gpu = import(A);
        NN_TRY_CHECK(a_gpu);
        // Phase C1b：f16 → f16 pipeline 原地缩放（A 永不物化 f32 副本）；
        // 无 pipeline → 引擎内回退（抬 f32 → 缩放 → 写回）
        if (a_gpu->precision() == Precision::F16)
        {
            const uint32_t cnt = static_cast<uint32_t>(A.rows() * A.cols());
            if (backend_->has(PipelineId::elementwise_v2_f16))
            {
                const GpuTensor a_ref = f16_view(*a_gpu);   // 局部保活（禁 &临时）
                auto r = backend_->elementwise_v2_gpu(a_ref, nullptr, nullptr, cnt, 1u, 2u, 0u, 1u,
                                                      static_cast<float>(s), 0.0f, 0.0f, &a_ref,
                                                      /*f16_io=*/true);
                NN_TRY_CHECK(r);
                if (A.is_cpu()) A = std::move(*a_gpu);
                return {};
            }
            auto a32 = cast(*a_gpu, Precision::F32);
            NN_TRY_CHECK(a32);
            auto r = backend_->elementwise_v2_gpu(a32->gpu_tensor(), nullptr, nullptr, cnt, 1u, 2u,
                                                  0u, 1u, static_cast<float>(s), 0.0f, 0.0f,
                                                  &a32->gpu_tensor());
            NN_TRY_CHECK(r);
            return cast_into(*a32, A);
        }

        const uint32_t count = static_cast<uint32_t>(A.rows() * A.cols());
        // BINARY, Mul, flags=1 (B is scalar), scalar_b = s
        auto r = backend_->elementwise_v2_gpu(a_gpu->gpu_tensor(), nullptr, nullptr, count, 1u, 2u,
                                              0u, 1u, static_cast<float>(s), 0.0f, 0.0f,
                                              &a_gpu->gpu_tensor());  // 原地写回 A
        NN_TRY_CHECK(r);
        if (A.is_cpu()) A = std::move(*a_gpu);
        return {};
    }

    // A = 0：使用 vkCmdFillBuffer 真原地清零（不分配新 buffer）
    [[nodiscard]] Result<void> zero_impl(Tensor& A) override
    {
        auto a_gpu = import(A);
        NN_TRY_CHECK(a_gpu);
        if (a_gpu->precision() == Precision::F16)   // vkCmdFillBuffer 字节级 → f16 原生可用
        {
            auto r16 = backend_->fill_zero_gpu(a_gpu->gpu_tensor<Precision::F16>());
            NN_TRY_CHECK(r16);
            if (A.is_cpu()) A = std::move(*a_gpu);
            return {};
        }
        auto r = backend_->fill_zero_gpu(a_gpu->gpu_tensor());
        NN_TRY_CHECK(r);
        // fill_zero 修改的是 GPU buffer 本身，Tensor 的 shared_ptr 不变
        // 但若 import 上传了新 Tensor，需要替换 A
        if (A.is_cpu()) A = std::move(*a_gpu);
        return {};
    }

    // ══════════════════════════════════════════════════════════════════════
    // 扫描级原语（RLA）——手写原语 shader + GpuBackend 接线
    // 形状契约见 compute_engine.hpp 扫描级原语注释；纯 GPU 架构：
    // 无 pipeline 时硬报错，不做 CPU 回退。
    // ══════════════════════════════════════════════════════════════════════

    // Phase C2f 辅助：f16 张量 → f32 副本（引擎内边界 cast 回退用）；
    // 已是 f32 → 共享所有权直通（零拷贝）。
    [[nodiscard]] Result<Tensor> cast_if_f16(const Tensor& t)
    {
        if (t.precision() != Precision::F16) return t;
        return cast(t, Precision::F32);
    }

    // Phase C2f：全 f16 输入 + f16 pipeline 可用 → 原生扫描（输入输出均
    // half，消灭适配层全尺寸 f32 副本——输出 (rows*5/3/dk, seq) 是大头）；
    // 混合精度（如 boundary 仍 f32）→ 引擎内把 f16 侧 cast 到 f32 走 f32
    // pipeline（正确性不变）。输出重贴 GpuTensorF16（缓冲按 2B/元素分配）。
    [[nodiscard]] Result<Tensor>
    scan_prefix_outer_impl(const Tensor& K, const Tensor& V, const Tensor& P, const Tensor& R,
                           const Tensor& A0, const Tensor& B0, bool has_state, std::size_t dk,
                           std::size_t heads, bool causal, const Tensor& boundary, bool has_bnd,
                           Precision prec = Precision::F32) override
    {
        NN_TRY(k, import(K));
        NN_TRY(v, import(V));
        NN_TRY(p, import(P));
        NN_TRY(r, import(R));
        NN_TRY(a, import(A0));
        NN_TRY(b, import(B0));
        NN_TRY(bn, import(boundary));
        const bool all_f16 = k->precision() == Precision::F16 && v->precision() == Precision::F16 &&
                             p->precision() == Precision::F16 && r->precision() == Precision::F16 &&
                             a->precision() == Precision::F16 && b->precision() == Precision::F16 &&
                             bn->precision() == Precision::F16;
        // 原生 f16 仅在请求输出也是 f16 时启用（输出契约：prec=F32 必须回
        // f32——走下方引擎内 cast 路径，输出 f32 恰好免适配层落回）
        if (prec == Precision::F16 && all_f16 && backend_->has_scan_f16_pipelines())
        {
            auto res = backend_->scan_prefix_outer_gpu(
                f16_view(*k), f16_view(*v), f16_view(*p), f16_view(*r), f16_view(*a), f16_view(*b),
                has_state, static_cast<uint32_t>(dk), static_cast<uint32_t>(heads), causal,
                f16_view(*bn), has_bnd, /*f16_io=*/true);
            NN_TRY_CHECK(res);
            return Tensor::from_gpu(GpuTensorF16(res->shared_buffer(), res->rows(), res->cols()));
        }
        // 引擎内 cast 回退（混合精度或无 f16 pipeline）：f16 侧抬 f32
        NN_TRY(k32, cast_if_f16(*k));
        NN_TRY(v32, cast_if_f16(*v));
        NN_TRY(p32, cast_if_f16(*p));
        NN_TRY(r32, cast_if_f16(*r));
        NN_TRY(a32, cast_if_f16(*a));
        NN_TRY(b32, cast_if_f16(*b));
        NN_TRY(n32, cast_if_f16(*bn));
        auto res = backend_->scan_prefix_outer_gpu(
            k32->gpu_tensor(), v32->gpu_tensor(), p32->gpu_tensor(), r32->gpu_tensor(),
            a32->gpu_tensor(), b32->gpu_tensor(), has_state, static_cast<uint32_t>(dk),
            static_cast<uint32_t>(heads), causal, n32->gpu_tensor(), has_bnd);
        NN_TRY_CHECK(res);
        return to_prec(Tensor::from_gpu(std::move(*res)), prec);
    }

    [[nodiscard]] Result<Tensor> scan_suffix_outer_impl(const Tensor& D, const Tensor& X,
                                                        const Tensor& Y, std::size_t dk,
                                                        std::size_t heads, bool causal,
                                                        const Tensor& boundary, bool has_bnd,
                                                        Precision prec = Precision::F32) override
    {
        NN_TRY(d, import(D));
        NN_TRY(x, import(X));
        NN_TRY(y, import(Y));
        NN_TRY(bn, import(boundary));
        const bool all_f16 = d->precision() == Precision::F16 && x->precision() == Precision::F16 &&
                             y->precision() == Precision::F16 && bn->precision() == Precision::F16;
        if (prec == Precision::F16 && all_f16 && backend_->has_scan_f16_pipelines())
        {
            auto res = backend_->scan_suffix_outer_gpu(
                f16_view(*d), f16_view(*x), f16_view(*y), static_cast<uint32_t>(dk),
                static_cast<uint32_t>(heads), causal, f16_view(*bn), has_bnd, /*f16_io=*/true);
            NN_TRY_CHECK(res);
            return Tensor::from_gpu(GpuTensorF16(res->shared_buffer(), res->rows(), res->cols()));
        }
        NN_TRY(d32, cast_if_f16(*d));
        NN_TRY(x32, cast_if_f16(*x));
        NN_TRY(y32, cast_if_f16(*y));
        NN_TRY(n32, cast_if_f16(*bn));
        auto res = backend_->scan_suffix_outer_gpu(
            d32->gpu_tensor(), x32->gpu_tensor(), y32->gpu_tensor(), static_cast<uint32_t>(dk),
            static_cast<uint32_t>(heads), causal, n32->gpu_tensor(), has_bnd);
        NN_TRY_CHECK(res);
        return to_prec(Tensor::from_gpu(std::move(*res)), prec);
    }

    [[nodiscard]] Result<Tensor> outer_col_impl(const Tensor& P, const Tensor& R, const Tensor& S,
                                                std::size_t dk, bool has_scale,
                                                Precision prec = Precision::F32) override
    {
        NN_TRY(p, import(P));
        NN_TRY(r, import(R));
        NN_TRY(s, import(S));
        const bool all_f16 = p->precision() == Precision::F16 && r->precision() == Precision::F16 &&
                             s->precision() == Precision::F16;
        if (prec == Precision::F16 && all_f16 && backend_->has(PipelineId::outer_col_f16))
        {
            auto res =
                backend_->outer_col_gpu(f16_view(*p), f16_view(*r), f16_view(*s),
                                        static_cast<uint32_t>(dk), has_scale, /*f16_io=*/true);
            NN_TRY_CHECK(res);
            return Tensor::from_gpu(GpuTensorF16(res->shared_buffer(), res->rows(), res->cols()));
        }
        NN_TRY(p32, cast_if_f16(*p));
        NN_TRY(r32, cast_if_f16(*r));
        NN_TRY(s32, cast_if_f16(*s));
        auto res = backend_->outer_col_gpu(p32->gpu_tensor(), r32->gpu_tensor(), s32->gpu_tensor(),
                                           static_cast<uint32_t>(dk), has_scale);
        NN_TRY_CHECK(res);
        return to_prec(Tensor::from_gpu(std::move(*res)), prec);
    }

    // ══════════════════════════════════════════════════════════════════════
    // 归约原语
    // ══════════════════════════════════════════════════════════════════════

    // Phase C2：f16 输入 → f16 归约 pipeline（输入 float16_t 直读、f32 归约、
    // 输出 f32 向量）——省掉"每个 f16 输入一份整份 f32 副本"的边界 cast。
    // 设备无 16bit 存储时 pipeline 未创建 → 引擎内 cast 回退（正确性不变）。
    [[nodiscard]] Result<GpuTensor> reduce_f16_or_cast_(const Tensor& A, uint32_t mode,
                                                        uint32_t reduce_op)
    {
        auto a_gpu = import(A);
        NN_TRY_CHECK(a_gpu);
        if (a_gpu->precision() == Precision::F16)
        {
            // f16 张量必须经 f16_view（gpu_tensor() 默认 P=F32 → 读 f16 buffer
            // 的 f32 view = 断言失败 [gpu_tensor-fail]）
            if (backend_->has(PipelineId::reduce_f16))
                return backend_->reduce_gpu(f16_view(*a_gpu), mode, reduce_op,
                                            /*f16_in=*/true);
            // 引擎内边界 cast（无 f16 pipeline 的设备）
            auto a32 = cast(*a_gpu, Precision::F32);
            NN_TRY_CHECK(a32);
            auto a32_gpu = import(*a32);
            NN_TRY_CHECK(a32_gpu);
            return backend_->reduce_gpu(a32_gpu->gpu_tensor(), mode, reduce_op);
        }
        return backend_->reduce_gpu(a_gpu->gpu_tensor(), mode, reduce_op);
    }

    [[nodiscard]] Result<Tensor> row_reduce_sum_impl(const Tensor& A,
                                                     Precision P = Precision::F32) override
    {
        // 归约恒 f32 累加；输出按 P 落回（原生 f16 输入由 reduce_f16_or_cast_ 处理）
        auto r = reduce_f16_or_cast_(A, /*mode=*/0u, /*reduce_op=*/0u);
        NN_TRY_CHECK(r);
        return to_prec(Tensor::from_gpu(std::move(*r)), P);
    }

    [[nodiscard]] Result<Tensor> col_reduce_sum_impl(const Tensor& A,
                                                     Precision P = Precision::F32) override
    {
        auto r = reduce_f16_or_cast_(A, /*mode=*/1u, /*reduce_op=*/0u);
        NN_TRY_CHECK(r);
        return to_prec(Tensor::from_gpu(std::move(*r)), P);
    }

    [[nodiscard]] Result<Tensor> col_reduce_max_impl(const Tensor& A,
                                                     Precision P = Precision::F32) override
    {
        auto r = reduce_f16_or_cast_(A, /*mode=*/1u, /*reduce_op=*/1u);
        NN_TRY_CHECK(r);
        return to_prec(Tensor::from_gpu(std::move(*r)), P);
    }

    // ══════════════════════════════════════════════════════════════════════
    // 表达式求值（AOT 注册表 + JIT 兜底）
    //
    // 折叠内联表达式 → expr_spec_key → 查构建期合成的融合 shader 注册表
    // （fused_registry.hpp，由 scan_exprs 收集 + gen_fused 合成）；命中则
    // dispatch 单个融合 shader；未命中走运行期 JIT 现编（懒加载
    // libnn-kernelgen.so → 生成 SPIR-V → 懒建 pipeline）。
    //
    // 形状无关融合：RowMod/RotateHalf 的周期/块大小（如 RoPE 的 d_k）是
    // 运行时视图参数（不进 key），dispatch 时按实际 spec 填充 push constant
    // vp 槽 → 同结构不同形状共享一个融合 shader，任意 d_k 都全融合。

    // ── 融合输入的类型擦除收集：绑定只关心底层 buffer（元素类型由 shader 声明
    //    决定）→ 同一结构可在 f32 / f16 两种存储上跑（Phase 2 in-kernel f16）。
    //    owners 必须与 bufs 同寿命：nsure_gpu 上传出的 Tensor 若不被持有，
    //    其 GpuBuffer（此刻唯一 owner）会在录制中途析构 → 绑定悬空（铁律 6：
    //    录制期生命周期；实测表现为偶发错值，如 A+=B err=0.5）。
    struct FusedInputs
    {
        std::vector<Tensor> owners;
        std::vector<const GpuBuffer*> bufs;
    };

    [[nodiscard]] Result<FusedInputs> fused_buffers_(std::span<const Tensor> ts)
    {
        FusedInputs out;
        out.owners.reserve(ts.size());
        out.bufs.reserve(ts.size());
        for (const auto& t : ts)
        {
            auto g = import(t);
            NN_TRY_CHECK(g);
            if (g->precision() == Precision::F16)
                out.bufs.push_back(&g->gpu_tensor<Precision::F16>().buffer());
            else out.bufs.push_back(&g->gpu_tensor().buffer());
            out.owners.push_back(std::move(*g));   // 保活（见上）
        }
        return out;
    }

#ifdef NN_FUSED_REGISTRY_EMBEDDED
    // ── 变体选择：native16（`key#a`）优先，运行期精度分派（`key#x`）回退 ──
    //   · psig == 0 → 全 f32 结构键 `key`（V0）。
    //   · 结构谓词 expr_prec_sig_native16 通过 → `key#a`（V2b 原生 f16 算术）。
    //     **必须在运行期用真实 spec 重跑谓词**：常量值已不进 key（同结构可带
    //     不同常量），生成期是对"生成那一份时的常量"判定，运行期实例的常量
    //     可能不满足谓词（如 0 < |c| < 6e-8 的 eps 在 f16 下 flush-to-0）。
    //     生成期判定与运行期判定不一致 = 静默错值。
    //   · 其余 → `key#x`（V1 运行期精度分派）：一份 shader 覆盖任意签名。
    //     ⚠ "任何非零签名都必定命中"只在**生成端产出了 `#x`** 且设备启用了
    //     16bit 存储时成立；生成被跳过（`[skip]`）/ `NN_SCAN_NO_DISPATCH=1` /
    //     设备无 16bit storage 时仍会 miss → 基类 NVI 入口回退边界 cast。
    // registry 与 pipeline **都**命中才选（设备无 shaderFloat16 时 `#a` 的
    // pipeline 不创建 → 自动落 `#x`）。
    //
    // 诊断（NN_PREC_TRACE=1）：记**实际选中的变体**（V1/V2），不记 V0 默认
    // 路径（f32 热路径，逐调用打日志会淹没输出）。两个约束：
    //   ① 只在**确认命中**后打（此前 V1 分支先打后查，miss 也报 alu-hit）；
    //   ② 只由真正 dispatch 的调用方传 log_hit=true ——
    //      `supports_expr_precision_variant` 只是查询（NVI 入口每次都调），
    //      它打日志会让计数翻倍（实测 12040 行 = 6020 次真实分派 × 2）。
    [[nodiscard]] const nn::fused::FusedShader* find_prec_variant_(const std::string& key,
                                                                   nn::ExprPrecSig psig,
                                                                   const nn::ExprSpec& spec,
                                                                   bool log_hit = false) const
    {
        if (psig == 0) return nn::fused::find_fused(key);      // V0：全 f32 结构键
        const bool trace = log_hit && nn::dsl::env_flag("NN_PREC_TRACE");
        const auto hit = [&](const std::string& k) -> const nn::fused::FusedShader*
        {
            const auto* fs = nn::fused::find_fused(k);
            if (!fs || !backend_->has_fused_shader(k)) return nullptr;
            if (trace) std::fprintf(stderr, "[prec][alu-hit] %s\n", k.c_str());
            return fs;
        };
        // V2b：native16 原生 f16 算术（运行期用真实 spec 复核结构谓词）
        if (nn::expr_prec_sig_native16(spec, psig))
            if (const auto* fs = hit(key + nn::EXPR_PREC_ALU_SUFFIX)) return fs;
        // V1t/V1s：storage-only **编译期特化**（热循环里没有 `prec` 三元）。
        // 与 V1 分派变体数值逐字等价；只需要 16 位存储，无 shaderFloat16 的
        // 设备（Maxwell）同样命中。优先于 V1。
        //   ① `key#xxxx`（V1t）：构建期扫描**观察到的签名**（dry-run + 模型
        //      pass 各跑 f32/f16 两遍，见 scan_exprs / ExprRegistry::add_variant）；
        //   ② `key#s`（V1s）："全输入 f16 + 输出 f16"兜底——始终发射，覆盖
        //      扫描没走到的路径（特化集合只是优化，`#x` 仍是语义兜底）。
        if (const auto* fs = hit(nn::expr_prec_sig_key(key, psig))) return fs;
        if (nn::expr_prec_sig_all_f16(spec, psig))
            if (const auto* fs = hit(key + nn::EXPR_PREC_STORAGE_SUFFIX)) return fs;
        // V1：运行期精度分派
        const std::string xkey = key + nn::EXPR_PREC_DISPATCH_SUFFIX;
        if (const auto* fs = hit(xkey)) return fs;
        if (trace)
            std::fprintf(stderr, "[prec][fallback] %s 未预生成 → 回退基类边界 cast\n",
                         xkey.c_str());
        return nullptr;
    }
#endif
    // ══════════════════════════════════════════════════════════════════════
    // Phase 2 运行期 JIT（miss → 现编；docs/internal/33 §2/§10）
    //
    // 派发 miss（内嵌注册表 + 后端 pipeline 均无）→ 引擎 JIT：懒加载
    // libnn-kernelgen.so → nnk_generate_spirv 现编 SPIR-V → 后端懒建 pipeline
    // → 执行。内嵌注册表降级为"预热加速器"（避免 warmup 全 JIT），非派发硬约束。
    // ══════════════════════════════════════════════════════════════════════

    // ── JIT 两段前置：.nnk 缓存加载 与 生成器 dlopen（彼此解耦）────────────
    // 缓存读取不需要 glslang——"预热命中跑"与 miss 后现编都先查 .nnk 缓存，
    // 只有真正 miss 才 dlopen libnn-kernelgen.so。
    [[nodiscard]] Result<void> ensure_cache_() const
    {
        if (kernel_cache_) return {};
        const std::string& cpath = nn::fused::kernel_cache_path();
        std::string bytes;
        if (std::ifstream f(cpath, std::ios::binary); f)
        {
            std::ostringstream ss;
            ss << f.rdbuf();
            bytes = ss.str();
        }
        const auto fresh = [&]()
        {
            auto c = std::make_shared<nn::fused::KernelCache>();
            c->generator_version = nn::fused::kKernelCacheGeneratorVersion;
            c->rec.set("nnk_format_version", nn::fused::kKernelCacheFormatVersion);
            c->rec.set("generator_version", nn::fused::kKernelCacheGeneratorVersion);
            kernel_cache_ = std::move(c);
        };
        if (bytes.empty())
        {
            fresh();
        }
        else
        {
            auto r = nn::fused::KernelCache::from_bytes(bytes, nn::fused::kKernelCacheGeneratorVersion);
            if (!r)
            {
                // 坏缓存不得让启动永久失败（#E1）：隔离改名留证（.corrupt），
                // 以空缓存重建——后续 JIT miss 重新填充并由 persist_cache_ 重写；
                // 改名失败（权限/占用）则退回删除原文件，同样消除"每次启动都
                // 解析失败"的死路。
                const std::string bad = cpath + ".corrupt";
                std::remove(bad.c_str());
                if (std::rename(cpath.c_str(), bad.c_str()) == 0)
                    std::fprintf(stderr, "[nnk] .nnk 缓存解析失败，已隔离为 %s，重建空缓存：%s\n",
                                 bad.c_str(), r.error().message.c_str());
                else
                {
                    std::remove(cpath.c_str());
                    std::fprintf(stderr, "[nnk] .nnk 缓存解析失败，隔离失败已删除原文件，重建空缓存：%s\n",
                                 r.error().message.c_str());
                }
                fresh();
            }
            else
            {
                kernel_cache_ = std::make_shared<nn::fused::KernelCache>(std::move(*r));
            }
        }
        return {};
    }

    // 生成器 .so 懒加载（仅"缓存 miss 且非 strict"才走到；失败 NN_FAIL 带 dlerror）。
    // 失败**不留"已尝试"痕迹**：只有 handle + 三个符号全部就绪才提交 jit_，
    // 下次调用会重新尝试（调用方修正 NN_KERNELGEN_PATH / 部署环境后可再 warmup）。
    // 旧实现先置 tried=true 再 dlopen：失败一次后二次调用会静默"成功"，随后
    // 解引用空 generate 指针（#M7 状态残留）。
    [[nodiscard]] Result<void> ensure_generator_() const
    {
        if (jit_.loaded()) return {};
        // dlopen libnn-kernelgen.so（NN_KERNELGEN_PATH 覆盖 → 系统库搜索）；
        // Windows 平台产物为 nn-kernelgen.dll（40HX 实测 LoadLibrary 按 .so 名
        // 查找报 win32 error 126）
        std::string path = nn::detail::env_string_raw("NN_KERNELGEN_PATH");
#ifdef _WIN32
        if (path.empty()) path = "nn-kernelgen.dll";
#else
        if (path.empty()) path = "libnn-kernelgen.so";
#endif
#ifdef _WIN32
        HMODULE h = LoadLibraryA(path.c_str());
        if (!h)
        {
            const auto err = static_cast<unsigned long>(GetLastError());
            NN_FAIL(path + " 加载失败（JIT 需要生成器，可设 NN_KERNELGEN_PATH）："
                    "win32 error " +
                    std::to_string(err));
        }
        auto sym = [h](const char* n) { return reinterpret_cast<void*>(GetProcAddress(h, n)); };
#else
        void* h = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!h)
            NN_FAIL("libnn-kernelgen.so 加载失败（JIT 需要生成器，可设 NN_KERNELGEN_PATH）：" +
                    std::string(dlerror()));
        auto sym = [h](const char* n) { return dlsym(h, n); };
#endif
        JitGenerator next;
        next.handle = h;
        next.generate = reinterpret_cast<decltype(next.generate)>(sym("nnk_generate_spirv"));
        next.free = reinterpret_cast<decltype(next.free)>(sym("nnk_free_spirv"));
        next.version = reinterpret_cast<decltype(next.version)>(sym("nnk_generator_version"));
        if (!next.generate || !next.free || !next.version)
        {
#ifdef _WIN32
            FreeLibrary(h);
#else
            ::dlclose(h);
#endif
            NN_FAIL("libnn-kernelgen.so 符号不完整（缺 nnk_generate_spirv/free/version）");
        }
        jit_ = next;
        return {};
    }

    // 兼容调用点：缓存 + 生成器（原 ensure_jit_ 语义）。
    [[nodiscard]] Result<void> ensure_jit_() const
    {
        NN_TRY(a, ensure_cache_());
        (void)a;
        return ensure_generator_();
    }

    // 原子写盘（temp+rename）+ 进程锁（flock，POSIX）：锁内重读盘上缓存并合并
    // 并发进程已 JIT 落盘的 kernel（防 lost-update），再原子替换。Windows 侧暂
    // 只做原子替换（LockFileEx 留后续）。miss 罕见，写放大可接受（doc §8）。
    [[nodiscard]] Result<void> persist_cache_() const
    {
        if (!kernel_cache_) return {};
        // 写 tmp 并**检查写入结果**：磁盘满/IO 错误时 ofstream 只置 failbit
        // （默认不抛），只管 write 不管 flush/close 会把截断的 tmp rename 成
        // 正式缓存 → 下次加载解析失败、整份缓存报废（触发 ensure_cache_ 的
        // 隔离重建，#E1，白丢全部已编译 kernel）。失败即删 tmp。
        const auto write_tmp = [](const std::string& tmp, const std::string& data)
        {
            std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
            if (!f) return false;
            f.write(data.data(), static_cast<std::streamsize>(data.size()));
            f.close();   // 冲刷缓冲区；写入/冲刷/关闭任一失败都会置 failbit
            if (!f)
            {
                std::remove(tmp.c_str());
                return false;
            }
            return true;
        };
#ifndef _WIN32
        // 进程锁：同一缓存路径的 .lock 文件上排他 flock；不同进程串行写。
        const std::string& path = nn::fused::kernel_cache_path();
        const std::string lockpath = path + ".lock";
        const int lfd = ::open(lockpath.c_str(), O_CREAT | O_RDWR, 0644);
        if (lfd < 0) NN_FAIL("无法打开 .nnk 锁文件：" + lockpath);
        struct ::flock lk;
        lk.l_type = F_WRLCK;
        lk.l_whence = SEEK_SET;
        lk.l_start = 0;
        lk.l_len = 0;
        if (::fcntl(lfd, F_SETLKW, &lk) != 0)
        {
            ::close(lfd);
            NN_FAIL("无法获取 .nnk 缓存写锁（fcntl）：" + lockpath);
        }
        // 锁内重读盘上缓存，合并其它进程已写入的 kernel
        {
            std::string disk_bytes;
            if (std::ifstream f(path, std::ios::binary); f)
            {
                std::ostringstream ss;
                ss << f.rdbuf();
                disk_bytes = ss.str();
            }
            if (!disk_bytes.empty())
            {
                auto disk = nn::KeyValueRecord::parse(disk_bytes);
                if (disk)
                {
                    std::string dgen;
                    const std::string ogen =
                        disk->get("generator_version", dgen)
                            ? dgen
                            : std::string(nn::fused::kKernelCacheGeneratorVersion);
                    if (ogen == kernel_cache_->generator_version)
                        NN_TRY_CHECK(kernel_cache_->merge_kernels_from(*disk));
                }
            }
        }
        const std::string data = kernel_cache_->serialize();
        ensure_nnk_cache_dir_(path);
        const std::string tmp = path + ".tmp";
        if (!write_tmp(tmp, data))
        {
            ::close(lfd);
            NN_FAIL("无法写 .nnk 缓存临时文件（磁盘满/IO 错误？）：" + tmp);
        }
        const bool renamed = (std::rename(tmp.c_str(), path.c_str()) == 0);
        ::close(lfd);
        if (!renamed)
        {
            std::remove(tmp.c_str());   // 半成品 tmp 不留盘（下次不误读）
            NN_FAIL("无法原子替换 .nnk 缓存：" + path);
        }
        return {};
#else
        const std::string data = kernel_cache_->serialize();
        const std::string& path = nn::fused::kernel_cache_path();
        ensure_nnk_cache_dir_(path);
        const std::string tmp = path + ".tmp";
        if (!write_tmp(tmp, data))
            NN_FAIL("无法写 .nnk 缓存临时文件（磁盘满/IO 错误？）：" + tmp);
        if (!MoveFileExA(tmp.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING))
        {
            std::remove(tmp.c_str());   // 半成品 tmp 不留盘（下次不误读）
            NN_FAIL("无法原子替换 .nnk 缓存：" + path);
        }
        return {};
#endif
    }

    // 编译（或缓存命中）得到某 key 的 SPIR-V；miss → 生成器现编 → 内存缓存 insert
    // + 落盘。prec_sig=0（V0）或 EXPR_PREC_SIG_DISPATCH（`#x`）。
    [[nodiscard]] Result<std::vector<std::uint32_t>> jit_compile_(const std::string& key,
                                                                  const nn::ExprSpec& spec,
                                                                  std::uint32_t prec_sig) const
    {
        NN_TRY(unused, ensure_cache_());
        (void)unused;
        const std::string K = nn::fused::kernel_cache_key(key, prec_sig);
        if (const auto* spv = kernel_cache_->hit(K)) return *spv;
        NN_TRY(unused2, ensure_generator_());
        (void)unused2;
        const auto spec_bytes = nn::fused::encode_expr_spec(spec);
        std::uint8_t* spirv = nullptr;
        std::size_t len = 0;
        nnk_kernel_meta meta;
        const nnk_status st = jit_.generate(spec_bytes.data(), spec_bytes.size(), prec_sig, &meta,
                                            &spirv, &len);
        if (st != NNK_OK || spirv == nullptr)
            NN_FAIL("JIT 现编失败（status=" + std::to_string(static_cast<int>(st)) + "），K=" + K);
        std::vector<std::uint32_t> spv(
            reinterpret_cast<const std::uint32_t*>(spirv),
            reinterpret_cast<const std::uint32_t*>(spirv) + len / sizeof(std::uint32_t));
        jit_.free(spirv);
        NN_TRY_CHECK(kernel_cache_->insert(K, spv));
        // .nnk 落盘是**纯优化**：磁盘满/只读/IO 错误只意味着"本次不更新盘上缓存"。
        // 此时 SPIR-V 已生成、已进内存缓存，本次派发完全可用——若把落盘失败
        // 当求值失败抛给调用方，调用方拿到的 Result 里没有任何输出，失败路径上
        // 继续用该结果就会崩（实测 0xC0000005）。截断 tmp 已由 persist_cache_
        // 清理，这里只留可观测告警。
        if (auto pr = persist_cache_(); !pr)
            std::fprintf(stderr, "[nnk] .nnk 缓存落盘失败（本次不写盘，继续执行）：%s\n",
                         pr.error().message.c_str());
        return spv;
    }

    // ── typed JIT 直通（in-kernel f16；AOT typed 变体的运行期接替者）────────
    // psig≠0 时按**具体签名**现编/命中 .nnk（键 = v<gen>.<结构键>#xxxx），f16
    // 输入直接绑定、输出按签名位直写——零 f32 边界副本。边界 cast 的 transient
    // 膨胀就是 2026-11-12 基准 `--f16` 显存 6972 MB 的根因（history「多精度
    // Phase 2」同形态实测 2795→6687 MB / backward OOM）；11-08 删 AOT 注册表后
    // 单表达式路径缺的那根线。与图路径 run_graph_kernel_into_ 的
    // jit_compile_(…, psig) 同源（静态签名特化，f32 算术 + f16 存储）。
    // 返回语义：
    //   · Result 错误 = 硬失败（缓存/生成器/输入绑定——V0 同样跑不了）
    //   · nullopt     = 该形态不可 typed（16bit 存储缺失 / 生成器不支持 /
    //                   建管线或派发失败）→ 调用方回退边界 cast；键进负缓存
    //                   （生成失败的 [kernelgen] 诊断由生成器自身打印，C5）
    //   · Tensor      = 已执行（override 直写时为 run_fused_gpu 的返回值）
    [[nodiscard]] Result<std::optional<Tensor>> typed_jit_eval_(
        const nn::ExprSpec& spec, const std::string& key, std::span<const Tensor> inputs,
        std::size_t rows, std::size_t cols, nn::ExprPrecSig psig, bool vector_out,
        GpuTensor* output_override = nullptr)
    {
        if (!backend_->device().has_16bit_storage()) return std::nullopt;
        // fold 档位键（#m<cap>；非 fold 恒等直通，与 jit_run_ 同源）
        std::string vkey_storage;
        const std::string& vkey = nn::expr_spec_fold_variant_key(spec, key, vkey_storage);
        const std::string K = nn::fused::kernel_cache_key(vkey, psig);
        {
            std::lock_guard lk(typed_negative_mu_);
            if (typed_negative_.contains(K)) return std::nullopt;
        }
        // SPIR-V：缓存命中直取；miss 现编。NNK_COMPILE_FAIL = 生成器不支持该
        // 形态 → 负缓存 + 回退（不重试）。
        NN_TRY(cc, ensure_cache_());
        (void)cc;
        std::vector<std::uint32_t> spv_owned;
        const std::vector<std::uint32_t>* spv = kernel_cache_->hit(K);
        if (spv == nullptr)
        {
            NN_TRY(gd, ensure_generator_());   // 生成器缺失 = 硬失败（V0 同需）
            (void)gd;
            const auto spec_bytes = nn::fused::encode_expr_spec(spec);
            std::uint8_t* buf = nullptr;
            std::size_t len = 0;
            nnk_kernel_meta meta{};
            const nnk_status st =
                jit_.generate(spec_bytes.data(), spec_bytes.size(), psig, &meta, &buf, &len);
            if (st != NNK_OK || buf == nullptr)
            {
                std::lock_guard lk(typed_negative_mu_);
                typed_negative_.insert(K);
                return std::nullopt;
            }
            spv_owned.assign(reinterpret_cast<const std::uint32_t*>(buf),
                             reinterpret_cast<const std::uint32_t*>(buf) +
                                 len / sizeof(std::uint32_t));
            jit_.free(buf);
            spv = &spv_owned;
            NN_TRY(ins, kernel_cache_->insert(K, *spv));
            (void)ins;
            // 落盘失败只告警（同 jit_compile_：SPIR-V 已在内存可用）
            if (auto pr = persist_cache_(); !pr)
                std::fprintf(stderr, "[nnk] .nnk 缓存落盘失败（本次不写盘，继续执行）：%s\n",
                             pr.error().message.c_str());
        }
        const nn::FusedKernelMeta meta = nn::fused_kernel_meta_of(spec);
        auto neg = [this](const std::string& k, const char* msg)
        {
            std::lock_guard lk(typed_negative_mu_);
            if (typed_negative_.insert(k).second)
                std::fprintf(stderr, "[prec][typed-jit] %s 建管/派发失败，回退边界 cast：%s\n",
                             k.c_str(), msg != nullptr ? msg : "");
        };
        auto ek = backend_->ensure_fused_kernel(K, *spv, meta, spec);
        if (!ek)
        {
            neg(K, ek.error().message.c_str());
            return std::nullopt;
        }
        auto fi = fused_buffers_(inputs);
        NN_TRY_CHECK(fi);
        const auto vp = nn::expr_spec_runtime_view_params(spec);
        const bool out_f16 = (output_override == nullptr) && nn::expr_prec_sig_out_f16(psig);
        auto run = backend_->run_fused_gpu(
            K, fi->bufs, spec.consts, rows, cols, vector_out, vp, spec.rparams, output_override,
            nn::expr_spec_runtime_matmul_k(spec), nn::expr_spec_runtime_matmul_batch(spec),
            nn::expr_spec_runtime_matmul_trans(spec), nn::expr_spec_runtime_fold_k(spec), out_f16,
            psig);
        if (!run)
        {
            neg(K, run.error().message.c_str());
            return std::nullopt;
        }
        if (out_f16)
        {
            // f16 输出按 2B/元素分配后重贴 GpuTensorF16；归约向量按归约轴收形
            const int raxis = nn::expr_spec_reduce_axis(spec);
            const std::size_t orows = vector_out ? ((raxis == 1) ? 1 : rows) : rows;
            const std::size_t ocols = vector_out ? ((raxis == 1) ? cols : 1) : cols;
            return std::optional<Tensor>(
                Tensor::from_gpu(GpuTensorF16(run->shared_buffer(), orows, ocols)));
        }
        return std::optional<Tensor>(Tensor::from_gpu(std::move(*run)));
    }

    // 完整 miss 路径：JIT 编译 V0（f32）→ 后端懒建 pipeline → 执行。
    // 调用方在 psig!=0 时已抬 f32 重入（eval_expr_impl 边界 cast 语义），故此处
    // 只 JIT V0 结构键；输出按 P 由调用方回贴（f16 → GpuTensorF16）。
    // fold 形态经同一路径：fold_k 是运行时形态参数（不进 key），kernelgen 的
    // generate() 对 spec.fold 内部分派 generate_glsl_fold(_v2)。
    [[nodiscard]] Result<Tensor> jit_run_(const nn::ExprSpec& spec, const std::string& key,
                                          std::span<const Tensor> inputs, std::size_t rows,
                                          std::size_t cols, Precision P)
    {
        // #P9 选路点：fold mm 段按运行期 mm_k 选 Ash 容量档（与 scan_prefix
        //   的 d32 变体同套——形状值仍走 PC，变体身份进 JIT 缓存/pipeline
        //   名）。1024 档无后缀 → 非 fold / 大 d_k 的键与历史逐字一致。
        //   直通时 vkey 引用 key（不拷贝：结构 key 超出 SSO，拷贝 = 一次堆分配）。
        std::string vkey_storage;
        const std::string& vkey = nn::expr_spec_fold_variant_key(spec, key, vkey_storage);
        auto spv_r = jit_compile_(vkey, spec, /*prec_sig=*/0);
        NN_TRY_CHECK(spv_r);
        auto fi_r = fused_buffers_(inputs);
        NN_TRY_CHECK(fi_r);
        std::vector<const GpuBuffer*>& gpu_inputs = fi_r->bufs;
        const nn::FusedKernelMeta meta = nn::fused_kernel_meta_of(spec);
        // 后端懒建 pipeline（幂等；设备能力不足 → 报错，绝无静默回退）
        NN_TRY_CHECK(backend_->ensure_fused_kernel(vkey, *spv_r, meta, spec));
        const auto vp = nn::expr_spec_runtime_view_params(spec);
        auto out = backend_->run_fused_gpu(
            vkey, gpu_inputs, spec.consts, rows, cols, /*vector_out=*/false, vp, spec.rparams,
            /*output_override=*/nullptr, nn::expr_spec_runtime_matmul_k(spec),
            nn::expr_spec_runtime_matmul_batch(spec), nn::expr_spec_runtime_matmul_trans(spec),
            nn::expr_spec_runtime_fold_k(spec), /*out_f16=*/false, /*prec_sig=*/0);
        NN_TRY_CHECK(out);
        if (P == Precision::F16)
            return Tensor::from_gpu(GpuTensorF16(out->shared_buffer(), rows, cols));
        return Tensor::from_gpu(std::move(*out));
    }


    [[nodiscard]] Result<Tensor> eval_expr_impl(const ExprSpec& raw_spec,
                                                std::span<const Tensor> inputs, std::size_t rows,
                                                std::size_t cols,
                                                Precision P = Precision::F32) override
    {
        // IR-C 录制段：入图不求值（占位 Tensor，end_expr 融合执行）。方案 A：
        // 录制状态读写与 begin_expr/end_expr 同在引擎 TU（跨 DSO TLS 单副本
        // 生效点）；拦截在 canonicalize/fold 选路之前 = 原 NVI 的位置与参数
        // （record_graph_node_ 收 raw spec）。
        if (auto* g = fused::recording_graph())
            return record_graph_node_(*g, raw_spec, inputs, rows, cols,
                                      /*vector_out=*/false, P);
        // 无注册表模式（NN_EXPR_SCAN 收集器 TU / 注册表缺失）：融合分支整体
        // 不编译 → rows/cols 仅在 NN_FUSED_REGISTRY_EMBEDDED 内被使用 →
        // 显式消 unused-parameter（clang -Werror）。
#ifndef NN_FUSED_REGISTRY_EMBEDDED
        (void)rows;
        (void)cols;
#endif
        // ── fold 段：canonicalize 对 fold 恒等（expr_opt 入口
        //    early-return），scan/runtime 两端 key 同源 → 直接用 raw_spec 查表。
        //    PC/分派形态由 run_fused_gpu 的 fold_k 参数走（见其形态校验）。
        if (raw_spec.fold)
        {
            const std::string fkey = nn::expr_spec_key(raw_spec);
#ifdef NN_FUSED_REGISTRY_EMBEDDED
            // fold 变体：native16 谓词排除 fold，find_prec_variant_ 的 `#a`
            // 探测必然落空 → 命中 `#x`（或有真实位图的存储变体）
            const nn::ExprPrecSig fpsig = nn::expr_prec_sig_of(inputs, P);
            const nn::fused::FusedShader* ffs =
                find_prec_variant_(fkey, fpsig, raw_spec, /*log_hit=*/true);
            if (ffs && backend_->has_fused_shader(ffs->key))
            {
                auto fi_r = fused_buffers_(inputs);
                NN_TRY_CHECK(fi_r);
                std::vector<const GpuBuffer*>& gpu_inputs = fi_r->bufs;
                const auto fvp = nn::expr_spec_runtime_view_params(raw_spec);
                const bool fout_f16 = nn::expr_variant_out_f16(ffs->prec_sig, fpsig);
                auto out = backend_->run_fused_gpu(
                    ffs->key, gpu_inputs, raw_spec.consts, rows, cols,
                    /*vector_out=*/false, fvp, raw_spec.rparams,
                    /*output_override=*/nullptr,
                    // 双域 fold 自带 matmul 段的 k/batch（7 槽 PC 的 slot5/6）；
                    // fold 的转置仍在 key 里（生成期定死）→ 此处恒 0（无该槽）
                    nn::expr_spec_runtime_matmul_k(raw_spec),
                    nn::expr_spec_runtime_matmul_batch(raw_spec),
                    /*matmul_trans=*/0u, nn::expr_spec_runtime_fold_k(raw_spec), fout_f16, fpsig);
                NN_TRY_CHECK(out);
                // f16 输出：按 2B/元素分配后重贴 GpuTensorF16（同非 fold 路径）
                if (fout_f16)
                    return Tensor::from_gpu(
                        GpuTensorF16(out->shared_buffer(), out->rows(), out->cols()));
                return Tensor::from_gpu(std::move(*out));
            }
#endif
            // 非 f32（输入或输出精度）→ 先试 typed JIT 直通（fold 带类型生成，
            // f16 直读直写）；不可 typed → 引擎内边界 cast 回退（抬 f32 重入；
            // F2 修复后基类不再代劳）
            if (!all_f32(inputs) || P != Precision::F32)
            {
                const nn::ExprPrecSig fpsig = nn::expr_prec_sig_of(inputs, P);
                NN_TRY(tj, typed_jit_eval_(raw_spec, fkey, inputs, rows, cols, fpsig,
                                           /*vector_out=*/false));
                if (*tj) return std::move(**tj);
                auto in32 = to_f32_all(inputs);
                NN_TRY_CHECK(in32);
                auto r = eval_expr_impl(raw_spec, *in32, rows, cols, Precision::F32);
                NN_TRY_CHECK(r);
                return to_prec(std::move(*r), P);
            }
            // fold 未命中（此处已全 f32）→ JIT：kernelgen 对 spec.fold 内部分派
            // generate_glsl_fold(_v2)；strict 语义在 jit_compile_ 内生效（D6）。
            auto jr_fold = jit_run_(raw_spec, fkey, inputs, rows, cols, P);
            NN_TRY_CHECK(jr_fold);
            return *jr_fold;
        }
        // ── canonical IR：canonicalize 为引擎内部优化（IR-A/IR-B），
        //    key 与 shader 合成两端一致；dispatch 用 canonical 的 consts ──
        // P4 残余：canonical spec + 结构 key 经 thread_local 备忘录（命中免
        // canonicalize 5 道 pass + key 的 FNV/snprintf；键值不变，见
        // expr_graph.hpp 的 canon_key_memo 不变量）。
        const nn::CanonKey ck = nn::canonicalize_with_key(raw_spec);
        const ExprSpec& spec = ck.canon;
        // ── AOT 匹配：按规范结构 key 查预编译融合 shader ──────────────
        const std::string& key = ck.key;
        // psig **无条件计算**（不随 NN_FUSED_REGISTRY_EMBEDDED 走）：`#endif`
        // 之后的「变体未预生成」判定（`psig != 0`）在有/无注册表两种模式下都要
        // 能编译 —— 无注册表 = NN_EXPR_SCAN 收集器 TU（注册表是它自己的产物，
        // 见 compute_vk_backend.hpp 的 NN_EXPR_SCAN 不变量）或注册表缺失的干净
        // 构建。无注册表模式下 supports_expr_precision_variant 恒 false → NVI
        // 入口已把输入抬到 f32 → psig 必为 0 → 正确落到下方注册表 miss 错误。
        const nn::ExprPrecSig psig = nn::expr_prec_sig_of(inputs, P);
#ifdef NN_FUSED_REGISTRY_EMBEDDED
        // 变体优先（native16 `#a` → 运行期精度分派 `#x`）：按**真实输入精度**
        // + 目标输出精度取变体（find_prec_variant_）。命中即 shader 直接半精度
        // 读/写，无需边界 cast；未命中则退回全 f32 key —— 此时输入必须已全 f32
        // （由基类 NVI 入口 cast）。
        const nn::fused::FusedShader* fs = find_prec_variant_(key, psig, spec, /*log_hit=*/true);
        if (fs && backend_->has_fused_shader(fs->key))
        {
            // 命中：收集 GPU 输入（同一 buffer 可重复绑定，如 RoPE 的 q×2）
            // GpuTensor 内部为 shared_ptr<GpuBuffer>，拷贝即共享，零成本
            auto fi_r = fused_buffers_(inputs);
            NN_TRY_CHECK(fi_r);
            std::vector<const GpuBuffer*>& gpu_inputs =
                fi_r->bufs;   // owners 随 fi_r 存活到本作用域末
            // 运行时视图参数（RowMod 周期 / RotateHalf 块大小）：
            // 同结构不同形状共享一个融合 shader，按实际 spec 填充 vp 槽
            const auto vp = nn::expr_spec_runtime_view_params(spec);
            // 输出存储精度由变体签名决定（bit16）：f16 输出要按 2B/元素分配缓冲，
            // 否则 shader 只写前半、返回的却是 f32 标签 → 下游全是垃圾（实测
            // batch32 训练 loss=NaN）。缓冲布局由 out_f16 决定，返回的 GpuTensor
            // 只是占位标签，这里按实际精度重贴 GpuTensorF16。
            // 输出存储精度：分派变体按**运行期真实签名**（bit16），其余按变体身份
            const bool out_f16 = nn::expr_variant_out_f16(fs->prec_sig, psig);
            auto out = backend_->run_fused_gpu(
                fs->key, gpu_inputs, spec.consts, rows, cols,
                /*vector_out=*/false, vp, spec.rparams, /*output_override=*/nullptr,
                nn::expr_spec_runtime_matmul_k(spec), nn::expr_spec_runtime_matmul_batch(spec),
                nn::expr_spec_runtime_matmul_trans(spec),
                /*fold_k=*/std::nullopt, out_f16, psig);
            NN_TRY_CHECK(out);
            if (out_f16) return Tensor::from_gpu(GpuTensorF16(out->shared_buffer(), rows, cols));
            return Tensor::from_gpu(std::move(*out));
        }
#endif

        // ── psig≠0：先试 typed JIT 直通（f16 直读直写、零边界副本）──────────
        // 不可 typed（生成器不支持该形态 / 16bit 存储缺失 / 负缓存命中）→
        // 回退**引擎内边界 cast**：输入抬 f32 → 重入本函数（psig==0，走 V0
        // f32 kernel）→ 输出按 P 落回。绝不把 f16 buffer 绑到 f32 shader 上
        //（静默错值）。此前这段由基类 NVI 入口代为处理；F2 修复后归因/选路
        // 只做一次。
        if (psig != 0)
        {
            NN_TRY(tj, typed_jit_eval_(spec, key, inputs, rows, cols, psig,
                                       /*vector_out=*/false));
            if (*tj) return std::move(**tj);
            if (prec_trace_enabled_()) trace_miss_(raw_spec, inputs, P);
            auto in32 = to_f32_all(inputs);
            NN_TRY_CHECK(in32);
            auto r = eval_expr_impl(raw_spec, *in32, rows, cols, Precision::F32);
            NN_TRY_CHECK(r);
            return to_prec(std::move(*r), P);
        }
        // ── 未命中 → 运行期 JIT 现编（D4 后为唯一 miss 路径；docs/internal/33/35）──
        // 引擎 dlopen libnn-kernelgen.so 现编 SPIR-V、懒建 pipeline、执行，并
        // 持久化进 .nnk 缓存。**绝不静默回退 CPU/边界 cast**。
        // psig==0（此刻输入已全 f32、P==F32）→ 只 JIT V0 结构键 `key`；输出 f32。
        auto jr = jit_run_(spec, key, inputs, rows, cols, P);
        NN_TRY_CHECK(jr);
        return *jr;
    }

    // ── 能力摘要（唯一对外通道；只披露事实，不用于选路）──────────────────
    // 设备事实在 init 期已解析（pipeline 创建结果）；这里只做只读汇总。
    [[nodiscard]] EngineInfo describe() const override
    {
        EngineInfo i;
        i.engine = "vulkan";
        i.device = backend_->device().device_name();
        i.native_f16_move = true;      // 字节级搬运 + 引擎内 cast 回退
        i.native_f16_reduce = backend_->has(PipelineId::reduce_f16);
        i.fused_prec_variant = true;   // 有带类型融合 shader 机制（AOT 注册表）
        i.ternary_gemm =
            backend_->has(PipelineId::trit_pack) && backend_->has(PipelineId::matmul_q);
        i.int8_dot = backend_->device().has_int8_dot();
        i.int8_dot_accelerated = backend_->device().int8_dot_accelerated();
        return i;
    }

    // ── 归约向量原生形状输出（LayerNorm/RMSNorm 小向量缓存） ────────
    // 与 eval_expr 相同，但以 vector_out=1 调度归约融合 shader（thread 0 写
    // (rows,1)/(1,cols) 归约向量，不写全尺寸广播）。
    [[nodiscard]] Result<Tensor> eval_expr_reduce_impl(const ExprSpec& raw_spec,
                                                       std::span<const Tensor> inputs,
                                                       std::size_t rows, std::size_t cols,
                                                       Precision P = Precision::F32) override
    {
        // IR-C 录制段：入图不求值（归约向量占位，end_expr 融合执行）——方案 A，
        // 归属引擎 TU（原 NVI 拦截位置与参数，vector_out=true）。
        if (auto* g = fused::recording_graph())
            return record_graph_node_(*g, raw_spec, inputs, rows, cols,
                                      /*vector_out=*/true, P);
        return eval_expr_reduce_run_(raw_spec, inputs, rows, cols, P, nullptr);
    }

    // 归约向量的目标传递版（T1_58 显存修复 #L2）：结果写进 out 的既有存储。
    // 仅 f32 dst 支持零分配直写；其他精度 → 参考求值 + cast_into 写回原存储。
    // 形状由 run_fused_gpu 的 output_override 校验（(rows,1)/(1,cols)）。
    [[nodiscard]] Result<void> eval_expr_reduce_into_impl(const ExprSpec& raw_spec,
                                                          std::span<const Tensor> inputs,
                                                          std::size_t rows, std::size_t cols,
                                                          Tensor& out) override
    {
        // 「录制段内不支持」守卫（方案 A：与 begin/end 同 TU 才读得到 DLL TLS
        // 副本；文案与原 NVI 逐字一致）。epoch 守卫在 NVI（与录制状态无关）。
        if (fused::is_recording())
            NN_FAIL("eval_expr_reduce_into: 不支持在 begin_expr/end_expr 录制段内使用");
        if (out.precision() != Precision::F32)
        {
            NN_TRY(t, eval_expr_reduce_impl(raw_spec, inputs, rows, cols, Precision::F32));
            return cast_into(*t, out);
        }
        NN_TRY_CHECK(eval_expr_reduce_run_(raw_spec, inputs, rows, cols, Precision::F32,
                                           &out.gpu_tensor()));
        return {};
    }

    // 共用实现：output_override != nullptr ⇒ 结果写进其既有存储（不分配）。
    [[nodiscard]] Result<Tensor> eval_expr_reduce_run_(const ExprSpec& raw_spec,
                                                       std::span<const Tensor> inputs,
                                                       std::size_t rows, std::size_t cols,
                                                       Precision P, GpuTensor* output_override)
    {
        // 无注册表模式：融合分支不编译 → rows/cols 仅在
        // NN_FUSED_REGISTRY_EMBEDDED 内被使用 → 消 unused-parameter。
#ifndef NN_FUSED_REGISTRY_EMBEDDED
        (void)rows;
        (void)cols;
#endif
        // fold 形态只经 eval_expr（PC 需 fold_k，本入口不传）——显式拒绝，
        //   否则落到 run_fused_gpu 的通用缺参错误，误导排查方向
        if (raw_spec.fold)
            NN_FAIL("GpuEngine::eval_expr_reduce: fold 表达式请走 eval_expr（需 fold_k 形态参数）");
        // canonical IR：与 eval_expr 同（canonicalize 为引擎内部优化）
        // P4 残余：经 thread_local 备忘录（键值不变，见 expr_graph.hpp）。
        const nn::CanonKey ck = nn::canonicalize_with_key(raw_spec);
        const ExprSpec& spec = ck.canon;

        // P10-②（E9 支持路径）：归约后的逐元素后处理链**已可执行**——此前这里
        // 硬报错（防生成器静默错值，实测 err≈1.9 @2026-09-19），但列归约分支
        // 其后整体重写（2026-09-25/27「列内组织」+ emit_col_merge、10-04 4 路
        // 累加器），本轮已在 40HX 上按主机参考复核两轴形态（col sqrt(sum/C+ε)
        // err=5.96e-08、row sum*s+b err=2.38e-07，fused_gpu_test
        // run_reduce_postprocess）→ 校验放开。CPU 端按指令下标反向切片早已
        // 支持同形态（两侧可直接对拍）；末指令非归约时 vector_out 输出沿
        // 归约轴恒定的语义由生成器输出段承担（见 generate_glsl_reduce）。

        const std::string& key = ck.key;
        const nn::ExprPrecSig psig = nn::expr_prec_sig_of(inputs, P);
#ifdef NN_FUSED_REGISTRY_EMBEDDED
        // 变体优先（`#a` native16 → `#x` 运行期分派）：与 eval_expr 同款
        // (key,sig) 匹配；归约形态的 native16 谓词恒 false → 自动落 `#x`
        const nn::fused::FusedShader* fs =
            (psig != 0) ? find_prec_variant_(key, psig, spec, /*log_hit=*/true)
                        : nn::fused::find_fused(key);
        if (fs && backend_->has_fused_shader(fs->key))
        {
            auto fi_r = fused_buffers_(inputs);
            NN_TRY_CHECK(fi_r);
            std::vector<const GpuBuffer*>& gpu_inputs =
                fi_r->bufs;   // owners 随 fi_r 存活到本作用域末
            const auto vp = nn::expr_spec_runtime_view_params(spec);
            const bool out_f16 = nn::expr_variant_out_f16(fs->prec_sig, psig);
            auto out = backend_->run_fused_gpu(
                fs->key, gpu_inputs, spec.consts, rows, cols, /*vector_out=*/true, vp, spec.rparams,
                output_override, nn::expr_spec_runtime_matmul_k(spec),
                nn::expr_spec_runtime_matmul_batch(spec), nn::expr_spec_runtime_matmul_trans(spec),
                std::nullopt, out_f16, psig);
            NN_TRY_CHECK(out);
            if (out_f16)
            {
                // 归约向量原生形状：(rows,1)（行）/ (1,cols)（列）
                const int raxis = nn::expr_spec_reduce_axis(spec);
                const std::size_t orows = (raxis == 1) ? 1 : rows;
                const std::size_t ocols = (raxis == 1) ? cols : 1;
                return Tensor::from_gpu(GpuTensorF16(out->shared_buffer(), orows, ocols));
            }
            return Tensor::from_gpu(std::move(*out));
        }
#endif
        // ── psig≠0：先试 typed JIT 直通（归约带类型生成，vector_out=1）──────
        // 不可 typed → 引擎内边界 cast 回退（抬 f32 重入；F2 修复）。
        // override 必须透传给递归：此前这里改调 eval_expr_reduce_impl（无
        // override），f16 输入 + f32 目标的 reduce_into 会写出新张量而 dst
        // 从未被写（静默 no-op）——本分支顺带修掉。
        if (psig != 0)
        {
            NN_TRY(tj, typed_jit_eval_(spec, key, inputs, rows, cols, psig,
                                       /*vector_out=*/true, output_override));
            if (*tj) return std::move(**tj);
            if (prec_trace_enabled_()) trace_miss_(raw_spec, inputs, P);
            auto in32 = to_f32_all(inputs);
            NN_TRY_CHECK(in32);
            auto r = eval_expr_reduce_run_(raw_spec, *in32, rows, cols, Precision::F32,
                                           output_override);
            NN_TRY_CHECK(r);
            return to_prec(std::move(*r), P);
        }
        // ── 注册表 miss → 运行期 JIT（P3/D4：AOT 收集链已删，JIT 是唯一路径）──
        // 归约形态：vector_out=1（thread 0 写 (rows,1)/(1,cols) 归约向量，同 AOT
        // 命中路径参数）；psig==0（非 f32 已在上一块边界 cast 回退）→ 只 JIT V0。
        // strict 语义在 jit_compile_ 内生效（D6）。
        NN_TRY(spv_r, jit_compile_(key, spec, /*prec_sig=*/0));
        {
            auto fi_r = fused_buffers_(inputs);
            NN_TRY_CHECK(fi_r);
            const nn::FusedKernelMeta meta = nn::fused_kernel_meta_of(spec);
            NN_TRY_CHECK(backend_->ensure_fused_kernel(key, *spv_r, meta, spec));
            const auto vp = nn::expr_spec_runtime_view_params(spec);
            auto out = backend_->run_fused_gpu(
                key, fi_r->bufs, spec.consts, rows, cols, /*vector_out=*/true, vp, spec.rparams,
                output_override, nn::expr_spec_runtime_matmul_k(spec),
                nn::expr_spec_runtime_matmul_batch(spec),
                nn::expr_spec_runtime_matmul_trans(spec),
                /*fold_k=*/std::nullopt, /*out_f16=*/false, /*prec_sig=*/0);
            NN_TRY_CHECK(out);
            return Tensor::from_gpu(std::move(*out));
        }
    }

    // ── 目标传递（destination-passing）：结果直接写回已有张量 ─────────────
    // 走 run_fused_gpu 的 output_override —— dst 的 buffer 作为写only输出绑定，
    // 若表达式引用了 leaf(dst) 则同一 buffer 同时作为 readonly 输入绑定：
    // 逐元素同索引"先读后写"，无跨调用危害（与 GPU 原地原语的既有做法一致）。
    // 语义与限制同 CpuEngine::eval_expr_into（仅逐元素表达式；无分配）。
    [[nodiscard]] Result<void> eval_expr_into_impl(const ExprSpec& raw_spec,
                                                   std::span<const Tensor> inputs, std::size_t rows,
                                                   std::size_t cols, Tensor& dst) override
    {
        // IR-C 录制段（P2 写穿物化）：目标传递节点——输出即 dst 既有张量，
        // end_expr 的复合 kernel 直接写回其存储（多输出槽 0）。方案 A：拦截与
        // record_graph_into_node_ 归属引擎 TU（原 NVI 的位置与参数，在 shape/
        // fold 守卫之前）。
        if (auto* g = fused::recording_graph())
            return record_graph_into_node_(*g, raw_spec, inputs, rows, cols, dst);
        if (dst.rows() != rows || dst.cols() != cols) NN_FAIL("eval_expr_into: dst shape mismatch");
        // fold 形态只经 eval_expr（PC 需 fold_k，本入口不传）——显式拒绝
        if (raw_spec.fold)
            NN_FAIL("GpuEngine::eval_expr_into: fold 表达式请走 eval_expr（需 fold_k 形态参数）");
        // P4 残余：经 thread_local 备忘录（键值不变，见 expr_graph.hpp）。
        const nn::CanonKey ck = nn::canonicalize_with_key(raw_spec);
        const ExprSpec& spec = ck.canon;
        if (nn::expr_spec_reduce_axis(spec) != -1)
            NN_FAIL("eval_expr_into: 仅支持逐元素表达式（无归约）");
        const std::string& key = ck.key;
        // 变体优先（`#a` native16 → `#x` 运行期分派）：目标精度 = dst.precision()
        const nn::ExprPrecSig psig = nn::expr_prec_sig_of(inputs, dst.precision());
#ifdef NN_FUSED_REGISTRY_EMBEDDED
        const nn::fused::FusedShader* fs =
            (psig != 0) ? find_prec_variant_(key, psig, spec, /*log_hit=*/true)
                        : nn::fused::find_fused(key);
        if (fs && backend_->has_fused_shader(fs->key))
        {
            auto fi_r = fused_buffers_(inputs);
            NN_TRY_CHECK(fi_r);
            std::vector<const GpuBuffer*>& gpu_inputs =
                fi_r->bufs;   // owners 随 fi_r 存活到本作用域末
            auto dst_gpu = import(dst);
            NN_TRY_CHECK(dst_gpu);
            // ── output_override 必须按目标存储精度取视图 ──────────────────
            // f16 目标必须经 f16_view 包 f16 buffer（同 run_fused_gpu 的
            // out_f16 做法）：按默认 F32 取 gpu_tensor() 会得到空
            // shared_ptr → Debug NN_ASSERT 引爆；Release 空解引用 = UB
            // （MSVC 下拿到空 override → 结果落临时 buffer 被丢弃 →
            //  **参数静默冻结（loss 恒 ln V）**）。存储/标签不一致时
            // 显式报错，绝不静默。
            std::optional<GpuTensor> dst_view;
            GpuTensor* dst_override = nullptr;
            if (dst_gpu->precision() == Precision::F16)
            {
                if (!dst_gpu->gpu_shared<Precision::F16>())
                    NN_FAIL("eval_expr_into: f16 目标缺少 f16 GPU 存储"
                            "（precision 标签与 variant 存储不一致）");
                dst_view = f16_view(*dst_gpu);
                dst_override = &*dst_view;
            }
            else
            {
                if (!dst_gpu->gpu_shared<Precision::F32>())
                    NN_FAIL("eval_expr_into: f32 目标缺少 f32 GPU 存储"
                            "（precision 标签与 variant 存储不一致）");
                dst_override = &dst_gpu->gpu_tensor();
            }
            const auto vp = nn::expr_spec_runtime_view_params(spec);
            auto out = backend_->run_fused_gpu(
                fs->key, gpu_inputs, spec.consts, rows, cols, /*vector_out=*/false, vp,
                spec.rparams, dst_override, nn::expr_spec_runtime_matmul_k(spec),
                nn::expr_spec_runtime_matmul_batch(spec), nn::expr_spec_runtime_matmul_trans(spec),
                /*fold_k=*/std::nullopt, /*out_f16=*/false, psig);
            NN_TRY_CHECK(out);
            // dst 原为 CPU staging 时，import 上传了新 buffer（结果在它上面）
            // → 用 upload 后的张量替换 dst，保证调用方看到更新后的数据
            if (dst.is_cpu()) dst = std::move(*dst_gpu);
            return {};
        }
#endif
        // ── psig≠0：先试 typed JIT 直通（结果直写 dst 既有存储，零临时槽）──
        // f16 目标 = f16 视图绑定（与 AOT 命中路径同款）；不可 typed → 回退
        // 边界 cast（目标 f16：在 f32 空间求值后 cast_into 写回 dst 原存储；F2）
        if (psig != 0)
        {
            auto dst_gpu = import(dst);
            NN_TRY_CHECK(dst_gpu);
            std::optional<GpuTensor> dst_view;
            GpuTensor* dst_override = nullptr;
            if (dst_gpu->precision() == Precision::F16)
            {
                if (!dst_gpu->gpu_shared<Precision::F16>())
                    NN_FAIL("eval_expr_into: f16 目标缺少 f16 GPU 存储"
                            "（precision 标签与存储不一致）");
                dst_view = f16_view(*dst_gpu);
                dst_override = &*dst_view;
            }
            else
            {
                if (!dst_gpu->gpu_shared<Precision::F32>())
                    NN_FAIL("eval_expr_into: f32 目标缺少 f32 GPU 存储"
                            "（precision 标签与存储不一致）");
                dst_override = &dst_gpu->gpu_tensor();
            }
            NN_TRY(tj, typed_jit_eval_(spec, key, inputs, rows, cols, psig,
                                       /*vector_out=*/false, dst_override));
            if (*tj)
            {
                // dst 原为 CPU staging 时，import 上传了新 buffer → 替换 dst
                //（同 V0 直写路径语义）
                if (dst.is_cpu()) dst = std::move(*dst_gpu);
                return {};
            }
            if (prec_trace_enabled_()) trace_miss_(raw_spec, inputs, dst.precision());
            auto in32 = to_f32_all(inputs);
            NN_TRY_CHECK(in32);
            auto tmp = create_tensor(rows, cols, Precision::F32);
            auto r = eval_expr_into_impl(raw_spec, *in32, rows, cols, tmp);
            NN_TRY_CHECK(r);
            return cast_into(tmp, dst);
        }
        // ── miss → 运行期 JIT（P3/D4；output_override = dst，同 AOT 命中路径）──
        // psig==0 ⇒ 输入与 dst 均 f32（f16 目标已在上一块边界 cast 回退）。
        // strict 语义在 jit_compile_ 内生效（D6）。
        NN_TRY(spv_r, jit_compile_(key, spec, /*prec_sig=*/0));
        {
            auto fi_r = fused_buffers_(inputs);
            NN_TRY_CHECK(fi_r);
            auto dst_gpu = import(dst);
            NN_TRY_CHECK(dst_gpu);
            if (!dst_gpu->gpu_shared<Precision::F32>())
                NN_FAIL("eval_expr_into: f32 目标缺少 f32 GPU 存储"
                        "（precision 标签与存储不一致）");
            GpuTensor* dst_override = &dst_gpu->gpu_tensor();
            const nn::FusedKernelMeta meta = nn::fused_kernel_meta_of(spec);
            NN_TRY_CHECK(backend_->ensure_fused_kernel(key, *spv_r, meta, spec));
            const auto vp = nn::expr_spec_runtime_view_params(spec);
            auto out = backend_->run_fused_gpu(
                key, fi_r->bufs, spec.consts, rows, cols, /*vector_out=*/false, vp, spec.rparams,
                dst_override, nn::expr_spec_runtime_matmul_k(spec),
                nn::expr_spec_runtime_matmul_batch(spec), nn::expr_spec_runtime_matmul_trans(spec),
                /*fold_k=*/std::nullopt, /*out_f16=*/false, /*prec_sig=*/0);
            NN_TRY_CHECK(out);
            // dst 原为 CPU staging 时，import 上传了新 buffer（结果在它上面）
            // → 用 upload 后的张量替换 dst，保证调用方看到更新后的数据
            if (dst.is_cpu()) dst = std::move(*dst_gpu);
            return {};
        }
    }

    // ── P2 多输出（IR-C 写穿物化）────────────────────────────────────────
    // 选路 = V0（extras 结构只生成 V0 变体）。经 NVI 进来时输入/输出已是全 f32
    // （NVI 对 f16/混合先边界 cast，P10-④）；图执行直达点（execute_fused_graph
    // 直调 eval_expr_multi_into_canon_）可能带 f16，由 canon_ 开头的同款边界
    // cast 折回。输出视图经 import 取 GpuTensor 句柄，
    // 以 output_override + extra_outs 交给 run_fused_gpu（无分配路径）。
    // P10-①：允许前置 matmul 段（多消费者 matmul 头的分量 kernel）——
    // generate_glsl_matmul 已发 extras（V0 only），派发按 spec 填 mm 形态参数。
    [[nodiscard]] Result<void> eval_expr_multi_into_impl(const ExprSpec& raw_spec,
                                                         std::span<const Tensor> inputs,
                                                         std::size_t rows, std::size_t cols,
                                                         std::span<Tensor> outs) override
    {
        // 「录制段内不支持」守卫（方案 A：归属引擎 TU；文案与原 NVI 逐字一致，
        // 在 fold/归约守卫之前 = 原 NVI 的相对次序）。epoch 守卫在 NVI。
        if (fused::is_recording())
            NN_FAIL("eval_expr_multi_into: 不支持在 begin_expr/end_expr 录制段内使用");
        if (raw_spec.fold || expr_spec_reduce_axis(raw_spec) != -1)
            NN_FAIL("eval_expr_multi_into: 仅支持逐元素/前置 matmul 段 spec（无 fold、无归约）");
        // P4 残余：经 thread_local 备忘录（键值不变，见 expr_graph.hpp）。
        const nn::CanonKey ck = nn::canonicalize_with_key(raw_spec);
        return eval_expr_multi_into_canon_(ck.canon, ck.key, inputs, rows, cols, outs);
    }

    // P4：已 canonical 的入口——融合图计划缓存已带 canon spec 与 key，
    // 直接复用（不再每 dispatch canonicalize + expr_spec_key）。
    [[nodiscard]] Result<void> eval_expr_multi_into_canon_(const ExprSpec& spec,
                                                           const std::string& key,
                                                           std::span<const Tensor> inputs,
                                                           std::size_t rows, std::size_t cols,
                                                           std::span<Tensor> outs)
    {
#ifndef NN_FUSED_REGISTRY_EMBEDDED
        // 无注册表模式（NN_EXPR_SCAN 收集器 TU）：融合分支整体不编译；
        // miss→JIT 兜底仍使用 spec/key。
        (void)inputs;
        (void)rows;
        (void)cols;
        (void)outs;
#endif
        // ── P10-④ f16/混合存储多输出：边界 cast（GPU 图执行直达点）────────
        // execute_fused_graph 为省 per-dispatch canonicalize 直调本函数、绕过
        // NVI，故 f16 分量（成员 prec=F16 → 占位/输出全 f16）必须在此折回：
        // 输入抬 f32 → 自递归走全 f32 快路（V0 复合 kernel）→ f16 输出槽
        // cast_into 逐槽落回原存储（f32 槽直写）。与 NVI eval_expr_multi_into
        // 的边界 cast 同一口径（多输出的 typed 签名不在生成器覆盖内；
        // 单输出 f16 图执行优先 typed JIT，不可 typed 时才回退同款边界 cast）
        // ；递归一次即全 f32，不会重入本分支。
        if (!all_f32(inputs) || !all_f32(outs))
        {
            auto in32 = to_f32_all(inputs);
            NN_TRY_CHECK(in32);
            std::vector<Tensor> slots;
            std::vector<Tensor> owned;   // f16 槽的 f32 临时（保活）
            slots.reserve(outs.size());
            owned.reserve(outs.size());
            for (Tensor& o : outs)
            {
                if (o.precision() == Precision::F32)
                {
                    slots.push_back(o);
                    continue;
                }
                Tensor t = create_tensor(rows, cols, Precision::F32);
                owned.push_back(std::move(t));
                slots.push_back(owned.back());
            }
            NN_TRY(r, eval_expr_multi_into_canon_(spec, key, *in32, rows, cols, slots));
            for (std::size_t i = 0; i < outs.size(); ++i)
            {
                if (outs[i].precision() != Precision::F32)
                {
                    NN_TRY(cr, cast_into(slots[i], outs[i]));
                }
            }
            return {};
        }
#ifdef NN_FUSED_REGISTRY_EMBEDDED
        const nn::fused::FusedShader* fs = nn::fused::find_fused(key);
        if (fs && backend_->has_fused_shader(fs->key))
        {
            auto fi_r = fused_buffers_(inputs);
            NN_TRY_CHECK(fi_r);
            std::vector<GpuTensor> owned;       // 视图持有者（ptrs 指向其元素）
            std::vector<GpuTensor*> ptrs;
            owned.reserve(outs.size());
            ptrs.reserve(outs.size());
            for (auto& t : outs)
            {
                auto g = import(t);
                NN_TRY_CHECK(g);
                if (!g->gpu_shared<Precision::F32>())
                    NN_FAIL("eval_expr_multi_into: f32 输出缺少 f32 GPU 存储");
                owned.push_back(g->gpu_tensor());
                ptrs.push_back(&owned.back());
            }
            std::vector<GpuTensor*> extra_ptrs(ptrs.begin() + 1, ptrs.end());
            const auto vp = nn::expr_spec_runtime_view_params(spec);
            auto r = backend_->run_fused_gpu(fs->key, fi_r->bufs, spec.consts, rows, cols,
                                             /*vector_out=*/false, vp, spec.rparams, ptrs[0],
                                             // P10-①：按 spec 填 matmul 形态参数（无 matmul
                                             // 段时 nullopt/1/0——与 run_fused_gpu 的对称
                                             // 防呆同源；多消费者 matmul 头分量必需）
                                             nn::expr_spec_runtime_matmul_k(spec),
                                             nn::expr_spec_runtime_matmul_batch(spec),
                                             nn::expr_spec_runtime_matmul_trans(spec),
                                             /*fold_k=*/std::nullopt,
                                             /*out_f16=*/false, /*prec_sig=*/0, extra_ptrs);
            NN_TRY_CHECK(r);
            return {};
        }
#endif
        // ── miss → 运行期 JIT（P3/D4；多输出 extras 同 AOT 命中路径）────────
        // NVI 已验全 f32 → 只 JIT V0；kernelgen 的 generate() 对 extras 在
        // sig=0 写穿全部成员输出。strict 语义在 jit_compile_ 内生效（D6）。
        NN_TRY(spv_r, jit_compile_(key, spec, /*prec_sig=*/0));
        {
            auto fi_r = fused_buffers_(inputs);
            NN_TRY_CHECK(fi_r);
            std::vector<GpuTensor> owned;       // 视图持有者（ptrs 指向其元素）
            std::vector<GpuTensor*> ptrs;
            owned.reserve(outs.size());
            ptrs.reserve(outs.size());
            for (auto& t : outs)
            {
                auto g = import(t);
                NN_TRY_CHECK(g);
                if (!g->gpu_shared<Precision::F32>())
                    NN_FAIL("eval_expr_multi_into: f32 输出缺少 f32 GPU 存储");
                owned.push_back(g->gpu_tensor());
                ptrs.push_back(&owned.back());
            }
            const nn::FusedKernelMeta meta = nn::fused_kernel_meta_of(spec);
            NN_TRY_CHECK(backend_->ensure_fused_kernel(key, *spv_r, meta, spec));
            std::vector<GpuTensor*> extra_ptrs(ptrs.begin() + 1, ptrs.end());
            const auto vp = nn::expr_spec_runtime_view_params(spec);
            auto r = backend_->run_fused_gpu(key, fi_r->bufs, spec.consts, rows, cols,
                                             /*vector_out=*/false, vp, spec.rparams, ptrs[0],
                                             // P10-①：同上——按 spec 填 matmul 形态参数
                                             nn::expr_spec_runtime_matmul_k(spec),
                                             nn::expr_spec_runtime_matmul_batch(spec),
                                             nn::expr_spec_runtime_matmul_trans(spec),
                                             /*fold_k=*/std::nullopt,
                                             /*out_f16=*/false, /*prec_sig=*/0, extra_ptrs);
            NN_TRY_CHECK(r);
            return {};
        }
    }

private:
    // ── IR-C：融合执行────────────────────────────────
    // 图 → 融合分析 → kernel 序列 → 逐个 AOT dispatch。每个 kernel 的输出
    // **直接写入**其 tail 节点的占位 buffer（output_override），使调用方持有
    // 的占位 Tensor 在 end_expr 后物化。被融合的中间节点无独立输出（寄存器
    // 内联），其占位 buffer 从未被写入、dispatch 后立即释放归还内存池。
    // D3：node_outputs 随图（thread_local 堆分配）隔离，不用引擎成员。
    [[nodiscard]] Result<void> execute_fused_graph(ExprGraph& g)
    {
        // P2-12 图级缓存跨 step 复用：训练每 step 图结构重复，命中缓存则跳过
        // 融合分析 + 拼接 canonicalize + validate（缓存只复用结构决策，运行时
        // 按 plan 绑定 node_outputs 与外部张量；不改数值、不破确定性）。
        // #M1：命中路径必须**按当前图重推导**运行期值（rparams、views 参数、
        // matmul k/batch/trans、fold k），const 值则进 graph_cache_key——单靠
        // "结构相同"复用陈旧运行期值会静默错值（K 更大时 GPU 越界读）。
        // P11：kernel 容器 thread_local 复用——命中路径按 plan 原位填充
        // （instantiate_plan_into），稳态不再每 step 为每个 kernel 重新分配
        // vector；执行序重排改为**就地 swap**（同样保留各 kernel 的容量）。
        auto& gcache = fused::graph_plan_cache();
        const std::uint64_t gkey = graph_cache_key(g);
        static thread_local std::vector<FusedKernel> kernels_scratch;
        std::vector<FusedKernel>& kernels = kernels_scratch;
        const auto cit = gcache.find(gkey);
        if (cit != gcache.end())
        {
            kernels.resize(cit->second.size());
            for (std::size_t i = 0; i < cit->second.size(); ++i)
                instantiate_plan_into(cit->second[i], g, kernels[i]);
        }
        else
        {
            kernels = fuse_expr_graph(g);
            std::vector<FusedKernelPlan> plans;
            plans.reserve(kernels.size());
            for (const auto& k : kernels)
                plans.push_back(plan_from_kernel(k));
            // P4：本轮（miss 步）也要带 canonical spec/key 派发——按计划原位
            // 回填一次（仅 miss 步发生；后续命中由上面的 instantiate 路径覆盖）。
            for (std::size_t i = 0; i < kernels.size(); ++i)
                instantiate_plan_into(plans[i], g, kernels[i]);
            if (gcache.size() >= fused::GRAPH_PLAN_CACHE_MAX) gcache.clear();
            gcache.emplace(gkey, std::move(plans));
        }

        // ── 稳定拓扑序（P2）：kernel 依赖 = 成员的 dep 落在另一 kernel 的成员 ──
        // 分量分裂/与 P1 kernel 交错时，fuse 的列表序不保证依赖序；按"就绪者
        // 取列表序最前"重排（确定性）。依赖只指向更早节点（add_node 保证）→
        // 无环。
        {
            std::vector<int> kernel_of(g.nodes.size(), -1);
            for (std::size_t ki = 0; ki < kernels.size(); ++ki)
                for (const int m : kernels[ki].members)
                    if (m >= 0 && static_cast<std::size_t>(m) < kernel_of.size())
                        kernel_of[static_cast<std::size_t>(m)] = static_cast<int>(ki);
            std::vector<std::uint8_t> done(kernels.size(), 0);
            // P11：不再把 kernel move 进临时 ordered 向量（那会丢弃每个 kernel
            // 的容器容量）；先求"原下标 → 执行位次"的置换，再就地 swap 应用。
            std::vector<std::size_t> order(kernels.size(), 0);
            std::size_t next_pos = 0;
            while (next_pos < kernels.size())
            {
                bool progressed = false;
                for (std::size_t ki = 0; ki < kernels.size(); ++ki)
                {
                    if (done[ki]) continue;
                    bool ready = true;
                    for (const int m : kernels[ki].members)
                    {
                        const auto& nd = g.nodes[static_cast<std::size_t>(m)];
                        for (const int d : nd.dep_of_input)
                        {
                            if (d < 0) continue;
                            const int kd = kernel_of[static_cast<std::size_t>(d)];
                            if (kd >= 0 && kd != static_cast<int>(ki) &&
                                !done[static_cast<std::size_t>(kd)])
                            {
                                ready = false;
                                break;
                            }
                        }
                        if (!ready) break;
                    }
                    if (ready)
                    {
                        done[ki] = 1;
                        order[ki] = next_pos++;
                        progressed = true;
                        break;   // 重新从头扫 = 稳定（列表序优先）
                    }
                }
                if (!progressed)   // 防御：理论无环，仍保底排完
                {
                    for (std::size_t ki = 0; ki < kernels.size(); ++ki)
                        if (!done[ki])
                        {
                            done[ki] = 1;
                            order[ki] = next_pos++;
                        }
                }
            }
            // 就地应用置换（swap 只交换内容，保留各 kernel 的向量容量供下一步复用）
            for (std::size_t i = 0; i < kernels.size(); ++i)
                while (order[i] != i)
                {
                    std::swap(kernels[i], kernels[order[i]]);
                    std::swap(order[i], order[order[i]]);
                }
        }

        // RV-03 兜底：顺序隐患（同目标写序 / WAR）必须在**实际执行序**下成立。
        // 融合侧已拒绝把隐患两端合组（见 expr_graph.hpp has_hazard），这里是
        // 最后一道闸——任何未来的分组/重排若破坏顺序，在此 fail-fast 而非静默错值。
        NN_TRY(order_ok, nn::validate_dst_order(g, kernels));
        (void)order_ok;

        // 保留占位集合：kernel tail（输出物化）+ kernel 输入源（融合边界，
        // 其 buffer 被后续 kernel 绑定为输入）+ P2 写穿成员（多输出 kernel 的
        // 全部成员都落回各自占位/目标）；其余（被融合中间节点）释放。
        std::vector<std::uint8_t> keep(g.nodes.size(), 0);
        for (const auto& k : kernels)
        {
            if (!k.spec.extras.empty())
            {
                for (const int m : k.members)
                    if (m >= 0 && static_cast<std::size_t>(m) < keep.size())
                        keep[static_cast<std::size_t>(m)] = 1;
            }
            if (k.tail >= 0 && static_cast<std::size_t>(k.tail) < keep.size())
                keep[static_cast<std::size_t>(k.tail)] = 1;
            for (const auto& in : k.inputs)
                if (in.node >= 0 && static_cast<std::size_t>(in.node) < keep.size())
                    keep[static_cast<std::size_t>(in.node)] = 1;
        }

        // ── 逐 kernel 解析输入/输出张量（顺序 = 稳定拓扑序）────────────────
        // 解析一次两用：P3 批量分组的依赖判定（存储级）+ 派发期的缓冲绑定。
        struct DispatchItem
        {
            std::vector<Tensor> ins;        // 与 spec.views 同序
            Tensor out;        // 主输出（tail 占位/目标传递 dst）
            std::vector<Tensor> extras;     // P2 写穿的其余成员输出（成员序）
            std::vector<const void*> in_tok;   // 输入存储身份（hazard 判定）
            std::vector<const void*> out_tok;  // 物化输出存储身份
            bool degraded = false;   // 复合降级 = 逐成员派发
            bool batchable = false;  // P3 批量资格（含 `key#b` 已登记）
            std::string bkey;               // 组键（结构 key + 参数值序列）
            const ExprSpec* bspec = nullptr; // P4：指向 kernel 已缓存的 canonical spec（不拷贝）
            ExprPrecSig b_sig = 0;          // 组内统一精度签名（0 = 全 f32）
            std::string b_prefix;           // 批量变体前缀（不含 `#b`）
        };
        std::vector<DispatchItem> items(kernels.size());
        // P3 逃生阀：NN_IRC_NO_BATCH=1 关闭批量派发（A/B 对照用；每次调用读取，
        // 测试可在同进程切换）。逐 kernel 原路径不受影响（数值不变）。
        // 行为开关走裸读（env_flag_raw）——探针门控的 env_flag 在默认构建下恒 false，
        // 会让该逃生阀静默失效。
        const bool batch_enabled = !::nn::detail::env_flag_raw("NN_IRC_NO_BATCH");
        for (std::size_t ki = 0; ki < kernels.size(); ++ki)
        {
            FusedKernel& k = kernels[ki];
            DispatchItem& d = items[ki];
            d.ins.reserve(k.inputs.size());
            for (auto& in : k.inputs)
            {
                if (in.node >= 0)
                {
                    const auto it = g.node_outputs.find(in.node);
                    if (it == g.node_outputs.end())
                        NN_FAIL("GpuEngine::end_expr: 依赖节点输出缺失（图 IR 状态损坏）");
                    d.ins.push_back(it->second);
                }
                else
                {
                    d.ins.push_back(in.external);
                }
            }
            const auto out_it = g.node_outputs.find(k.tail);
            if (out_it == g.node_outputs.end())
                NN_FAIL("GpuEngine::end_expr: tail 占位缺失（图 IR 状态损坏）");
            d.out = out_it->second;
            // 复合降级判定（与原循环同判据：写穿前置条件不满足即逐成员派发）
            d.degraded = k.members.size() > 1 && !composite_kernel_available_(k, d.ins, d.out);
            // P2 写穿成员输出（成员序，tail 之后；派发输出槽 [1..]，与
            // spec.extras 同序——fuse 构造时即按 members 去尾序登记）
            if (!k.spec.extras.empty() && !d.degraded)
            {
                for (const int m : k.members)
                {
                    if (m == k.tail) continue;
                    const auto mo = g.node_outputs.find(m);
                    if (mo == g.node_outputs.end())
                        NN_FAIL("GpuEngine::end_expr: 写穿成员占位缺失（图 IR 状态损坏）");
                    d.extras.push_back(mo->second);
                }
            }
            // 物化输出全集（hazard 判定）：降级/写穿形态 = 全部成员占位，
            // 其余 = 仅 tail（被融合中间节点不落回存储，无跨 kernel 消费者）
            if (d.degraded || !k.spec.extras.empty())
            {
                for (const int m : k.members)
                {
                    const auto mo = g.node_outputs.find(m);
                    if (mo != g.node_outputs.end())
                        d.out_tok.push_back(tensor_storage_token_(mo->second));
                }
            }
            else d.out_tok.push_back(tensor_storage_token_(d.out));
            d.in_tok.reserve(d.ins.size());
            for (const Tensor& t : d.ins)
                d.in_tok.push_back(tensor_storage_token_(t));
            // P3 批量资格：纯逐元素（含 extras 复合）+ 全 f32 GPU + 不降级 +
            // `key#b` 已登记（无 BDA 时注册期即跳过 → 天然不可批量）。
            // P4：bspec 直接指向 kernel 计划缓存的 canonical 形态与 key 同源，
            // 不再每 kernel 每 step 重新 canonicalize。#M1：运行期字段（rparams、
            // views 参数、matmul k/batch/trans、fold k）已由 instantiate_plan_into
            // 按当前图重填，故 bspec 的 vp/consts/rparams 与本次图一致。
            const ExprSpec& bspec = k.canon;
            d.bspec = &bspec;
            if (!d.degraded && !k.vector_out && nn::expr_spec_batchable(bspec) && batch_enabled &&
                (k.rows == 0 || k.cols <= UINT32_MAX / k.rows))
            {
                // 组内统一的精度签名：输入 + 主输出（extras 不在签名编码里
                // → 带 extras 的结构只走 V0 全 f32，见 batch_variant_available_
                // 命中条件与 fused_generate 的发射目标）。
                bool gpu_ok = true;
                for (const Tensor& t : d.ins)
                    gpu_ok = gpu_ok && t.valid() && t.is_gpu();
                for (const Tensor& t : d.extras)
                    gpu_ok = gpu_ok && t.valid() && t.is_gpu() && t.precision() == Precision::F32;
                gpu_ok = gpu_ok && d.out.valid() && d.out.is_gpu();
                const nn::ExprPrecSig b_sig = (gpu_ok && d.ins.size() < 16u)
                                                  ? nn::expr_prec_sig_of(d.ins, d.out.precision())
                                                  : 0u;
                const std::string b_prefix = (gpu_ok && (b_sig == 0u || d.extras.empty()))
                                                 ? batch_variant_prefix_(bspec, b_sig)
                                                 : std::string{};
                if (!b_prefix.empty())
                {
                    d.batchable = true;
                    d.b_sig = b_sig;
                    d.b_prefix = b_prefix;
                    d.bkey = batch_group_key_(b_prefix, bspec);
                }
            }
        }

        // ── P3 同签名分组（跨链批量派发）──────────────────────────────────
        // 组键 = 结构 key + consts/rparams/vp **值序列**（批量派发共享一份 PC，
        // 形状数据走实例表）；贪心按列表序收集，单组 CAP 32。入组条件：
        //   ① 同组键；② 未满 CAP；③ 组内成员两两互不依赖（存储级：无任一
        //   成员消费/覆写另一成员的物化输出——覆盖目标传递同 dst 的无边形态）；
        //   ④ 移入较早的组不得逆置任何既有顺序：凡与更早 kernel 有存储冲突者，
        //   其派发位置必须先于本组（组在其首成员位置一次性派发 = 整组前移）。
        // 决策只按列表序 + 键值扫描（确定性铁律 8），不依赖容器无序迭代。
        constexpr std::size_t kBatchCap = 32;
        struct BatchGroup
        {
            std::string key;
            std::vector<std::size_t> members;   // items 下标（列表序）
            std::size_t first = 0;    // 首成员下标 = 派发位置
        };
        std::vector<BatchGroup> groups;
        std::vector<int> group_of(items.size(), -1);
        // 已定派发位置：入组 = 组首成员下标；未入组 = 自身下标
        const auto action_pos = [&](std::size_t yi) -> std::size_t
        {
            const int gy = group_of[yi];
            return gy >= 0 ? groups[static_cast<std::size_t>(gy)].first : yi;
        };
        const auto hazard = [](const DispatchItem& a, const DispatchItem& b)
        {
            const auto hit =
                [](const std::vector<const void*>& x, const std::vector<const void*>& y)
            {
                for (const void* p : x)
                    if (p != nullptr && std::find(y.begin(), y.end(), p) != y.end()) return true;
                return false;
            };
            return hit(a.out_tok, b.in_tok) || hit(a.out_tok, b.out_tok) ||
                   hit(b.out_tok, a.in_tok);
        };
        for (std::size_t xi = 0; xi < items.size(); ++xi)
        {
            const DispatchItem& x = items[xi];
            if (!x.batchable) continue;
            int chosen = -1;
            for (std::size_t gi = 0; gi < groups.size() && chosen < 0; ++gi)
            {
                const BatchGroup& gr = groups[gi];
                if (gr.key != x.bkey || gr.members.size() >= kBatchCap) continue;
                bool ok = true;
                for (const std::size_t m : gr.members)
                    if (hazard(x, items[m]))
                    {
                        ok = false;
                        break;
                    }
                if (!ok) continue;
                for (std::size_t yi = 0; yi < xi && ok; ++yi)
                {
                    if (group_of[yi] == static_cast<int>(gi))
                        continue;   // 同组成员与 x 已由上面排除 hazard
                    if (hazard(x, items[yi]) && action_pos(yi) >= gr.first) ok = false;
                }
                if (ok) chosen = static_cast<int>(gi);
            }
            if (chosen >= 0)
            {
                groups[static_cast<std::size_t>(chosen)].members.push_back(xi);
                group_of[xi] = chosen;
            }
            else
            {
                groups.push_back(BatchGroup{x.bkey, {xi}, xi});
                group_of[xi] = static_cast<int>(groups.size() - 1);
            }
        }

        // ── 派发：动作序 = kernel 列表序；批量组在其首成员位置一次性派发 ──
        for (std::size_t ki = 0; ki < kernels.size(); ++ki)
        {
            FusedKernel& k = kernels[ki];
            DispatchItem& d = items[ki];
            const int gi = group_of[ki];
            if (gi >= 0)
            {
                const BatchGroup& gr = groups[static_cast<std::size_t>(gi)];
                if (gr.members.front() != ki) continue;   // 已随本组首成员一次性派发
                if (gr.members.size() >= 2)
                {
                    if (::nn::detail::env_flag_raw("NN_IRC_TRACE"))
                        std::fprintf(stderr, "[ir-c][batch] %zu 实例 → 1 派发 %s\n",
                                     gr.members.size(), gr.key.c_str());
                    std::vector<FusedBatchInst> insts;
                    insts.reserve(gr.members.size());
                    // 取实例缓冲：**必须按张量自己的精度**取槽位
                    //（`gpu_tensor()` 默认模板实参是 F32 → f16 张量会命中
                    // "无 P 精度存储" 断言）。批量变体的元素类型是编译期定死的
                    // （组键里带签名），所以这里只需拿地址。
                    const auto gpu_buf = [](const Tensor& t) -> const GpuBuffer*
                    {
                        return t.precision() == Precision::F16
                                   ? &t.gpu_tensor<Precision::F16>().buffer()
                                   : &t.gpu_tensor<Precision::F32>().buffer();
                    };
                    for (const std::size_t m : gr.members)
                    {
                        const DispatchItem& dm = items[m];
                        FusedBatchInst bi;
                        bi.inputs.reserve(dm.ins.size());
                        for (const Tensor& t : dm.ins)
                            bi.inputs.push_back(gpu_buf(t));
                        bi.out = gpu_buf(dm.out);
                        bi.extra_outs.reserve(dm.extras.size());
                        for (const Tensor& t : dm.extras)
                            bi.extra_outs.push_back(gpu_buf(t));
                        bi.count = static_cast<std::uint32_t>(kernels[m].rows * kernels[m].cols);
                        bi.cols = static_cast<std::uint32_t>(kernels[m].cols);
                        insts.push_back(std::move(bi));
                    }
                    const ExprSpec& s = *items[gr.members.front()].bspec;
                    auto br = backend_->run_fused_gpu_batch(
                        items[gr.members.front()].b_prefix + nn::EXPR_BATCH_SUFFIX, insts, s.consts,
                        nn::expr_spec_runtime_view_params(s), s.rparams);
                    NN_TRY_CHECK(br);
                    continue;
                }
                // 单成员组 = 常规逐 kernel（单 kernel 现有语义不变）
            }
            // ── 通用降级可观测（NN_IRC_TRACE=1）：逐元素候选但未成组 ──────
            if (::nn::detail::env_flag_raw("NN_IRC_TRACE") && !d.degraded && !k.vector_out &&
                nn::expr_spec_batchable(*d.bspec))
                std::fprintf(stderr, "[ir-c][batch-degrade] %s → 逐 kernel 派发（%s）\n",
                             nn::expr_spec_key(*d.bspec).c_str(),
                             !batch_enabled ? "NN_IRC_NO_BATCH"
                             : !d.batchable ? "key#b 未登记/无 BDA"
                                            : "同签名未集齐 2 实例（依赖/上限）");

            // ── 复合 kernel 的注册表降级（通用机制，非单算子特例）─────────
            // 融合**分组本身依赖精度**：P2 写穿组要求成员全 F32、P1 链要求成员
            // 同精度。于是混精度 profile（`--f16` = optimizer:F32 / param:F16）
            // 会产出 f32 扫描期根本不存在的复合结构（实测 Adam 的 {K3,K4} F16
            // 尾链）→ 构建期按精度组合枚举是 2^4 爆炸，不可取。改为运行期降级：
            // 复合 kernel 未登记（或写穿前置条件不满足：多输出 V0 要求全 f32）
            // 时拆回成员逐个派发——每个成员都是**已登记的单节点结构**（scan 的
            // 个体结构恒登记），注册表契约仍然成立，只是该段退化为"1 节点 = 1
            // dispatch"（= IR-C 之前的形态）。未登记的**单节点**仍是硬报错。
            if (d.degraded)
            {
                // 通用诊断（NN_IRC_TRACE=1；与 NN_PREC_TRACE 同哲学：降级不是
                // 错误，但必须可观测——否则"某段悄悄没融合"无从归因）。
                // C3 裁决：NN_IRC_TRACE 是非探针门行为开关（AGENTS §7）→ 裸读。
                if (::nn::detail::env_flag_raw("NN_IRC_TRACE"))
                    std::fprintf(stderr,
                                 "[ir-c][degrade] 复合 kernel(%zu 成员, extras=%zu)"
                                 " 未登记 → 逐成员派发\n",
                                 k.members.size(), k.spec.extras.size());
                for (const int m : k.members)
                {
                    NN_TRY(mr, dispatch_graph_member_(g, m));
                }
                continue;
            }

            // C3（36 号台账，2026-11-06）：每 kernel 的 `[ir-c][run]` 打印已删
            // ——命中执行是热路径常态，逐 kernel 刷屏只有排查期价值；降级
            // （[ir-c][degrade]）与批量（[ir-c][batch]/[batch-degrade]）打印保留
            // （低频、可观测"某段悄悄没融合"，AGENTS §7 口径）。
            if (k.spec.extras.empty())
            {
                // P4：canonical spec + 结构 key 由计划缓存带入，派发不再重复
                // canonicalize / expr_spec_key（命中路径零构造）。
                auto r = run_graph_kernel_into_(k.spec, d.ins, k.rows, k.cols, k.vector_out, d.out,
                                                &k.canon, &k.spec_key);
                NN_TRY_CHECK(r);
            }
            else
            {
                // P2 写穿物化：输出槽 [0] = tail，[1..] = 其余成员（成员序，
                // 与 spec.extras 同序——fuse 构造时即按 members 去尾序登记）
                std::vector<Tensor> outs;
                outs.reserve(d.extras.size() + 1);
                outs.push_back(d.out);
                for (const Tensor& t : d.extras)
                    outs.push_back(t);
                auto r = eval_expr_multi_into_canon_(k.canon, k.spec_key, d.ins, k.rows, k.cols, outs);
                NN_TRY_CHECK(r);
            }
        }

        // 释放被融合中间节点的占位 buffer（显存中间张量消除，同旧实现）。
        // 安全依据：被融合节点的占位只被 node_outputs 持有；tail 与输入源
        // 保留（调用方句柄与之共享 buffer）。
        for (auto it = g.node_outputs.begin(); it != g.node_outputs.end();)
        {
            const int nd = it->first;
            if (nd >= 0 && static_cast<std::size_t>(nd) < keep.size() &&
                !keep[static_cast<std::size_t>(nd)])
                it = g.node_outputs.erase(it);
            else ++it;
        }
        return {};
    }

    // 融合 kernel 单次执行：结果写进 out 的既有存储（output_override）。
    // 形态覆盖 fold（fold_k 形态参数）/ matmul 段（S5）/ 归约向量输出
    // （vector_out）/ 精度变体（`#a`/`#x`/V0）——与 eval_expr_impl /
    // eval_expr_into_impl 的选路同源，output_override 精度视图同
    // eval_expr_into_impl（f16 目标必须经 f16_view 包 f16 buffer）。
    // P4：`canon_hint` / `key_hint` 非空时表示调用方已持有 canonical spec 与
    // 结构 key（图计划缓存算一次），跳过本函数的 canonicalize / expr_spec_key；
    // 无 hint 的直派（eval_expr 系 / 降级派发 dispatch_graph_member_）改走
    // thread_local 备忘录（canonicalize_with_key，键值不变——见 expr_graph.hpp）。
    [[nodiscard]] Result<void> run_graph_kernel_into_(const ExprSpec& raw_spec,
                                                      std::span<const Tensor> inputs,
                                                      std::size_t rows, std::size_t cols,
                                                      bool vector_out, Tensor& out,
                                                      const ExprSpec* canon_hint = nullptr,
                                                      const std::string* key_hint = nullptr)
    {
        const bool is_fold = raw_spec.fold.has_value();
        // P4：命中 plan 缓存的 kernel 已带 canonical spec 与结构 key —— 直接复用；
        // 无 hint 且非 fold 走备忘录（fold 的 canonicalize 恒等，直接用 raw）。
        std::optional<nn::CanonKey> memo_own;
        const ExprSpec* spec_p = canon_hint;
        if (spec_p == nullptr)
        {
            if (is_fold) spec_p = &raw_spec;
            else
            {
                memo_own = nn::canonicalize_with_key(raw_spec);
                spec_p = &memo_own->canon;
            }
        }
        const ExprSpec& spec = *spec_p;
        std::string key_own;
        const std::string* key_p = key_hint;
        if (key_p == nullptr)
        {
            if (memo_own) key_p = &memo_own->key;   // 随备忘录一并返回（不重复格式化）
            else
            {
                key_own = nn::expr_spec_key(spec);   // fold：raw 的 key（恒等）
                key_p = &key_own;
            }
        }
        const std::string& key = *key_p;
        const nn::ExprPrecSig psig = nn::expr_prec_sig_of(inputs, out.precision());
        // ── psig≠0（任一 f16 参与）→ 先试 typed JIT 直通，不可则回退边界 cast ──
        // C4 裁决（2026-11-07）确认边界 cast 链正确（四层对拍全程净，历史
        // full-f16 NaN 根因是融合内联了图外持久输出，pitfalls §2.6，已修）；
        // 2026-11-13 起图内单输出成员改走与 eval_expr_* 同源的 typed JIT：
        // f16 直读直写、f32 算术、零边界副本。边界 cast 的 transient 正是
        // 2026-11-12 基准 §8.3 残余（optimizer 小张量双 cast ~1.1 ms/step）。
        // 不可 typed（16bit 存储缺失 / 生成器不支持该形态 / 建管或派发失败 /
        // 负缓存命中）→ 回退边界 cast：输入抬 f32 → V0 kernel → cast_into 写回
        // out 的真实存储（正确性保底路径不变）。
        if (psig != 0)
        {
            auto out_gpu = import(out);
            NN_TRY_CHECK(out_gpu);
            std::optional<GpuTensor> out_view;
            GpuTensor* out_override = nullptr;
            if (out_gpu->precision() == Precision::F16)
            {
                if (!out_gpu->gpu_shared<Precision::F16>())
                    NN_FAIL("run_graph_kernel_into: f16 占位缺少 f16 GPU 存储"
                            "（precision 标签与存储不一致）");
                out_view = f16_view(*out_gpu);
                out_override = &*out_view;
            }
            else
            {
                if (!out_gpu->gpu_shared<Precision::F32>())
                    NN_FAIL("run_graph_kernel_into: f32 占位缺少 f32 GPU 存储"
                            "（precision 标签与存储不一致）");
                out_override = &out_gpu->gpu_tensor();
            }
            NN_TRY(tj, typed_jit_eval_(spec, key, inputs, rows, cols, psig, vector_out,
                                       out_override));
            if (*tj)
            {
                // out 原为 CPU staging 时，import 上传了新 buffer → 替换 out
                if (out.is_cpu()) out = std::move(*out_gpu);
                return {};
            }
            auto in32 = to_f32_all(inputs);
            NN_TRY_CHECK(in32);
            if (vector_out)
            {
                auto tmp = create_tensor(out.rows(), out.cols(), Precision::F32);
                auto r0 = run_graph_kernel_into_(raw_spec, *in32, rows, cols, vector_out, tmp,
                                                 canon_hint, key_hint);
                NN_TRY_CHECK(r0);
                return cast_into(tmp, out);
            }
            auto r = eval_expr_impl(raw_spec, *in32, rows, cols, Precision::F32);
            NN_TRY_CHECK(r);
            return cast_into(*r, out);
        }
#ifndef NN_FUSED_REGISTRY_EMBEDDED
        // 无注册表模式（P3/D4 起的常态）：融合分支整体不编译
        (void)rows;
        (void)cols;
        (void)vector_out;
        (void)psig;
#endif
#ifdef NN_FUSED_REGISTRY_EMBEDDED
        const nn::fused::FusedShader* fs =
            (psig != 0) ? find_prec_variant_(key, psig, spec, /*log_hit=*/true)
                        : nn::fused::find_fused(key);
        if (fs && backend_->has_fused_shader(fs->key))
        {
            auto fi_r = fused_buffers_(inputs);
            NN_TRY_CHECK(fi_r);
            std::vector<const GpuBuffer*>& gpu_inputs = fi_r->bufs;   // owners 随 fi_r 存活
            auto out_gpu = import(out);
            NN_TRY_CHECK(out_gpu);
            std::optional<GpuTensor> out_view;
            GpuTensor* out_override = nullptr;
            if (out_gpu->precision() == Precision::F16)
            {
                if (!out_gpu->gpu_shared<Precision::F16>())
                    NN_FAIL("GpuEngine::end_expr: f16 占位缺少 f16 GPU 存储"
                            "（precision 标签与存储不一致）");
                out_view = f16_view(*out_gpu);
                out_override = &*out_view;
            }
            else
            {
                if (!out_gpu->gpu_shared<Precision::F32>())
                    NN_FAIL("GpuEngine::end_expr: f32 占位缺少 f32 GPU 存储"
                            "（precision 标签与存储不一致）");
                out_override = &out_gpu->gpu_tensor();
            }
            const auto vp = nn::expr_spec_runtime_view_params(spec);
            auto r = backend_->run_fused_gpu(
                fs->key, gpu_inputs, spec.consts, rows, cols, vector_out, vp, spec.rparams,
                out_override, nn::expr_spec_runtime_matmul_k(spec),
                nn::expr_spec_runtime_matmul_batch(spec), nn::expr_spec_runtime_matmul_trans(spec),
                is_fold ? nn::expr_spec_runtime_fold_k(spec) : std::nullopt,
                /*out_f16=*/false, psig);
            NN_TRY_CHECK(r);
            return {};
        }
#endif
        // ── miss → 运行期 JIT（P3/D4：AOT 收集链已删，JIT 是唯一 miss 路径）──
        // 按**具体签名**现编（kernelgen 的 emitter 支持静态签名特化，与构建期
        // key#xxxx 变体同源；psig 是位图、不是 #x 分派哨兵），f16 输入/输出
        // 直接命中、无需边界 cast。管线名 = .nnk 缓存键（含签名后缀），
        // 与 V0 管线（纯 key）互不冲突。strict 语义在 jit_compile_ 内生效（D6）。
        {
            // #P9 选路点（同 jit_run_）：fold mm 段按运行期 mm_k 选 Ash 容量档，
            //   变体身份 = 键后缀 #m<cap>，同时进 .nnk 缓存键与 pipeline 名。
            //   直通时 vkey 引用 key（零拷贝，见 expr_spec_fold_variant_key）。
            std::string vkey_storage;
            const std::string& vkey = nn::expr_spec_fold_variant_key(spec, key, vkey_storage);
            auto spv_r = jit_compile_(vkey, spec, psig);
            NN_TRY_CHECK(spv_r);
            const std::string pname = nn::fused::kernel_cache_key(vkey, psig);
            auto fi_r = fused_buffers_(inputs);
            NN_TRY_CHECK(fi_r);
            auto out_gpu = import(out);
            NN_TRY_CHECK(out_gpu);
            std::optional<GpuTensor> out_view;
            GpuTensor* out_override = nullptr;
            if (out_gpu->precision() == Precision::F16)
            {
                if (!out_gpu->gpu_shared<Precision::F16>())
                    NN_FAIL("run_graph_kernel_into: f16 占位缺少 f16 GPU 存储"
                            "（precision 标签与存储不一致）");
                out_view = f16_view(*out_gpu);
                out_override = &*out_view;
            }
            else
            {
                if (!out_gpu->gpu_shared<Precision::F32>())
                    NN_FAIL("run_graph_kernel_into: f32 占位缺少 f32 GPU 存储"
                            "（precision 标签与存储不一致）");
                out_override = &out_gpu->gpu_tensor();
            }
            const nn::FusedKernelMeta meta = nn::fused_kernel_meta_of(spec);
            NN_TRY_CHECK(backend_->ensure_fused_kernel(pname, *spv_r, meta, spec));
            const auto vp = nn::expr_spec_runtime_view_params(spec);
            auto r = backend_->run_fused_gpu(
                pname, fi_r->bufs, spec.consts, rows, cols, vector_out, vp, spec.rparams,
                out_override, nn::expr_spec_runtime_matmul_k(spec),
                nn::expr_spec_runtime_matmul_batch(spec), nn::expr_spec_runtime_matmul_trans(spec),
                is_fold ? nn::expr_spec_runtime_fold_k(spec) : std::nullopt,
                /*out_f16=*/false, psig);
            NN_TRY_CHECK(r);
            return {};
        }
    }

    // ── 复合 kernel 可用性查询（注册表降级的判据）────────────────────────
    // true = 该复合结构在 AOT 注册表里可执行（含 `#a`/`#x` 精度变体与写穿形态
    // 前置条件）。**只查不判错**：miss 由调用方降级为逐成员派发（见
    // execute_fused_graph）。与 run_graph_kernel_into_ / eval_expr_multi_into
    // 的选路同源——这里若"过宽"（说可用但实际查不到）不会静默错值，只是让
    // 下游把原来的硬报错照常抛出。
    [[nodiscard]] bool composite_kernel_available_(const FusedKernel& k,
                                                   std::span<const Tensor> ins,
                                                   const Tensor& out) const
    {
#ifndef NN_FUSED_REGISTRY_EMBEDDED
        // 无注册表（P3/D4 起的常态）：可用性 = JIT 能否执行。
        //   · 单输出链：JIT 兜底覆盖（fold/reduce/静态签名，run_graph_kernel_into_）→ 可用；
        //   · 多输出（extras）：**F32/F16 存储可用**（P10-④）——eval_expr_multi_into
        //     的 NVI 与 GPU 图执行直达点（eval_expr_multi_into_canon_）各带边界
        //     cast：f16/混合输入抬 f32 跑 V0 复合 kernel、f16 输出 cast_into
        //     落回，与单输出 f16 图执行同一口径；非 {F32,F16} 精度的张量不可
        //     构造，保守分支保留（不可用 = 逐成员降级，可观测非错误）。
        if (k.spec.extras.empty()) return true;
        for (const auto& t : ins)
            if (t.precision() != Precision::F32 && t.precision() != Precision::F16)
                return false;
        return out.precision() == Precision::F32 || out.precision() == Precision::F16;
#else
        // P4 残余：canon/spec_key 随计划缓存必已填充（execute_fused_graph 的
        // 命中/未命中两条路径都过 instantiate_plan_into，运行期值已按当前图
        // 重填；graph_cache_key 保证结构+const 值一致 ⇒ k.canon ≡ 对 k.spec
        // 现算 canonicalize 的结果）——直接复用，不再每 kernel 每 step
        // canonicalize + expr_spec_key。
        if (k.spec.extras.empty())
        {
            // 单输出链：与 run_graph_kernel_into_ 同源的 (key, psig) 选路
            const nn::ExprPrecSig psig = nn::expr_prec_sig_of(ins, out.precision());
            const nn::fused::FusedShader* fs =
                (psig != 0) ? find_prec_variant_(k.spec_key, psig, k.canon, /*log_hit=*/false)
                            : nn::fused::find_fused(k.spec_key);
            return fs != nullptr && backend_->has_fused_shader(fs->key);
        }
        // 写穿物化（多输出）：V0 预编译 + F32/F16 存储（f16 由 canon_ 开头的
        // 边界 cast 折回 f32 后查 V0，与无注册表分支同条件；不满足 = 降级）。
        for (const auto& t : ins)
            if (t.precision() != Precision::F32 && t.precision() != Precision::F16)
                return false;
        if (out.precision() != Precision::F32 && out.precision() != Precision::F16)
            return false;
        const nn::fused::FusedShader* fs = nn::fused::find_fused(k.spec_key);
        return fs != nullptr && backend_->has_fused_shader(fs->key);
#endif
    }

    // ── P3：Tensor → 底层存储身份（批量分组的 hazard 判定用）──────────────
    // 同一底层 buffer（或 CPU 同一 Matrix 存储）→ 同一 token；空 = 不适用。
    // 用途：判"任一成员是否消费/覆写另一成员的物化输出"（含目标传递同 dst 的
    // 无依赖边形态，如 AdamW 的 decay 与 update 都写 p）。
    [[nodiscard]] static const void* tensor_storage_token_(const Tensor& t)
    {
        if (!t.valid()) return nullptr;
        if (t.is_gpu())
        {
            if (t.precision() == Precision::F16)
            {
                auto s = t.gpu_shared<Precision::F16>();
                return s ? static_cast<const void*>(s->shared_buffer().get()) : nullptr;
            }
            auto s = t.gpu_shared<Precision::F32>();
            return s ? static_cast<const void*>(s->shared_buffer().get()) : nullptr;
        }
        if (t.precision() == Precision::F16)
        {
            auto s = t.cpu_shared<Precision::F16>();
            return s ? static_cast<const void*>(s.get()) : nullptr;
        }
        auto s = t.cpu_shared<Precision::F32>();
        return s ? static_cast<const void*>(s.get()) : nullptr;
    }

    // ── P3：`key#b` 批量变体可用性（注册表 + pipeline 双命中）──────────────
    // 设备无 buffer device address 时注册期即跳过 `#b` → 此处恒 false →
    // 调用方逐 kernel 降级（注册表降级，不是错误）。
    // sig = 组内统一精度签名（0 = 全 f32；非 0 = 静态特化批量变体，键 =
    // `key#xxxx#b`，见 fused_generate 的批量发射目标）。
    // **返回命中的批量变体"前缀"**（`key` / `key#xxxx` / `key#s`，不含 `#b`）
    // ——组键与派发键都必须用它，否则"精确签名未发射、退回 `#s`"的一类会被
    // 派发到不存在的键上。未命中返回空串（→ 调用方逐 kernel 降级）。
    [[nodiscard]] std::string batch_variant_prefix_(const ExprSpec& spec, nn::ExprPrecSig sig) const
    {
#ifdef NN_FUSED_REGISTRY_EMBEDDED
        const std::string key = nn::expr_spec_key(spec);
        const auto hit = [this](const std::string& k)
        {
            const nn::fused::FusedShader* fs = nn::fused::find_fused(k);
            return fs != nullptr && backend_->has_fused_shader(fs->key);
        };
        // 与 find_prec_variant_ 同序：精确签名键 → 全 f16 兜底（`key#s`，
        // 全 f16 签名不重复发射 `key#xxxx` 变体）。
        const std::string exact = nn::expr_prec_sig_key(key, sig);
        if (hit(exact + nn::EXPR_BATCH_SUFFIX)) return exact;
        if (nn::expr_prec_sig_all_f16(spec, sig) &&
            hit(key + nn::EXPR_PREC_STORAGE_SUFFIX + nn::EXPR_BATCH_SUFFIX))
            return key + nn::EXPR_PREC_STORAGE_SUFFIX;
        return {};
#else
        (void)spec;
        (void)sig;
        return {};
#endif
    }

    // ── P3 批量组键：结构 key + consts/rparams/vp **值位型序列**────────────
    // 批量派发共享一份 push constants → 值必须逐位一致才可同组（值不进结构
    // key，但进组键）。按 32 位位型（%08x）而非十进制串接：同值同键、异位型
    // 必异键（NaN 载荷 / -0.0 不合并——PC 是逐位 memcpy，合并即改数值）。
    [[nodiscard]] static std::string batch_group_key_(const std::string& b_prefix,
                                                      const ExprSpec& spec)
    {
        static_assert(sizeof(Scalar) == sizeof(std::uint32_t),
                      "batch_group_key_ 按 32 位位型编码标量");
        // 组键 = 批量变体前缀（结构 key + 精度签名身份） + 参数值位型序列：
        // **精度必须进组键** —— 批量变体的元素类型是编译期定死的（f16 槽走
        // float16_t 的 buffer reference），同一组内混签名会把 f16 缓冲按 f32 读。
        std::string s = b_prefix;
        const auto feed_u32 = [&s](std::uint32_t v)
        {
            char buf[9];
            std::snprintf(buf, sizeof(buf), "%08x", v);
            s += '|';
            s += buf;
        };
        const auto feed_scalar = [&feed_u32](Scalar v)
        {
            std::uint32_t bits = 0;
            std::memcpy(&bits, &v, sizeof(bits));
            feed_u32(bits);
        };
        for (Scalar v : spec.consts)
            feed_scalar(v);
        s += "|r";
        for (Scalar v : spec.rparams)
            feed_scalar(v);
        s += "|v";
        for (std::uint32_t v : nn::expr_spec_runtime_view_params(spec))
            feed_u32(v);
        return s;
    }

    // ── 逐成员派发（复合 kernel 未登记时的降级路径）────────────────────────
    // 每个成员按**自己的单节点结构 + 自己的输入/输出占位**执行，等价于 IR-C
    // 之前的逐表达式 dispatch（中间量落回占位存储，不再寄存器内联）。成员的
    // 输入占位此时尚未释放（释放在 end_expr 全部 dispatch 之后），故依赖边
    // 经 node_outputs 正常解析。
    [[nodiscard]] Result<void> dispatch_graph_member_(ExprGraph& g, int m)
    {
        const auto& nd = g.nodes[static_cast<std::size_t>(m)];
        std::vector<Tensor> mins;
        mins.reserve(nd.spec.views.size());
        for (std::size_t k = 0; k < nd.spec.views.size(); ++k)
        {
            const int dep = (k < nd.dep_of_input.size()) ? nd.dep_of_input[k] : -1;
            if (dep >= 0)
            {
                const auto it = g.node_outputs.find(dep);
                if (it == g.node_outputs.end())
                    NN_FAIL("GpuEngine::end_expr: 降级派发依赖输出缺失（图 IR 状态损坏）");
                mins.push_back(it->second);
            }
            else
            {
                mins.push_back(k < nd.input_tensors.size() ? nd.input_tensors[k] : Tensor{});
            }
        }
        const auto mo = g.node_outputs.find(m);
        if (mo == g.node_outputs.end())
            NN_FAIL("GpuEngine::end_expr: 降级派发占位缺失（图 IR 状态损坏）");
        return run_graph_kernel_into_(nd.spec, mins, nd.rows, nd.cols, nd.vector_out, mo->second);
    }

    // ── 辅助：f16 Tensor → GpuTensor **绑定视图** ─────────────────────────
    // 只借用底层 buffer + 形状；字节布局由 f16 存储版 pipeline 决定（调用方
    // 必须同时保证 f16_io 语义）。与 run_fused_gpu 的 out_f16 同一套做法
    // （GpuTensorT<F32> 包裹 f16 buffer），不做任何精度转换。
    [[nodiscard]] static GpuTensor f16_view(const Tensor& t)
    {
        const auto& g = t.gpu_tensor<Precision::F16>();
        return GpuTensor(g.shared_buffer(), g.rows(), g.cols());
    }

    // ── 辅助：确保 Tensor 在 GPU 上 ──────────────────────────────────────
    // 若已是 GPU，返回共享拷贝（零开销）；若为 CPU，上传到 GPU。
    // 纯 GPU 架构下，所有 Tensor 应已是 GPU，此方法为防御性兜底。
    // ── 跨设备拉取：CPU → 设备直传（M6，17 §4.3 / 15 §3.2 P3）────────────
    // 原 `ensure_gpu`（16 §2 的 43 处调用点已全部改为公共入口 `import`）。
    // 语义：
    //   已在设备：同精度 = 零拷贝别名；异精度 = 引擎内 cast；
    //   CPU 源 + f32→f32：直接从宿主存储上载（与原 ensure_gpu 同一条快路径，
    //     不经额外宿主拷贝）；
    //   其余精度组合（如 f16 源）：经宿主中转（升 f32 再按 P 落回）——
    //     原 ensure_gpu 硬取 `cpu_matrix()`（F32 槽）对 f16 源会取到空指针。
    [[nodiscard]] Result<Tensor> import_impl(const Tensor& src, Precision P) override
    {
        if (src.is_gpu())
        {
            if (P == src.precision()) return src; // 同设备同精度 = 零拷贝别名（共享拷贝）
            return cast(src, P); // 同设备异精度 = 引擎内 cast
        }
        if (!src.valid()) NN_FAIL("import: invalid tensor");
        if (src.precision() == Precision::F32 && P == Precision::F32)
        {
            auto r = GpuTensor::from_matrix(src.cpu_matrix(), *backend_);
            NN_TRY_CHECK(r);
            return Tensor::from_gpu(std::move(*r));
        }
        auto m = to_matrix(src, Precision::F32);
        NN_TRY_CHECK(m);
        return from_matrix(*m, P);
    }
};

} // namespace nn

#endif // NN_HAS_VULKAN
