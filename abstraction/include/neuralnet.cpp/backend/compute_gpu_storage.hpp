// ═══════════════════════════════════════════════════════════════════════════
//  compute_gpu_storage.hpp — GPU 存储层声明（MIT 抽象层，零 Vulkan 依赖）
//
//  跨界值类型：Tensor 的 GPU 存储（shared_ptr<GpuBuffer>）被消费方持有，但
//  消费方从不操作其内部。GpuBuffer 用 **PIMPL**（shared_ptr<Impl>，Impl 在
//  引擎 .so 侧定义）——消费方只需前向声明 + shared_ptr 布局，Vulkan 类型
//  （VkDevice/VkBuffer/MemoryPool::Allocation）完全不进抽象层头。
//  shared_ptr 的 deleter 在创建点（引擎侧）捕获，消费方析构时经 control
//  block 调引擎侧析构，消费方不需要 Impl 完整类型。
//
//  ABI 契约：本头（连同 compute_tensor.hpp）的布局**与 NN_HAS_VULKAN 无关**，
//  消费方与 .so 两侧布局恒一致 → 跨界 Tensor 无需同宏编译。
// ═══════════════════════════════════════════════════════════════════════════
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include <neuralnet.cpp/core_errors.hpp>
#include <neuralnet.cpp/core_domain.hpp>
#include <neuralnet.cpp/precision.hpp>

namespace nn
{

class GpuBackend;   // 引擎侧完整定义（compute_vk_backend.hpp，LGPL）
class MemoryPool;   // 引擎侧完整定义（backend/compute_memory_pool.hpp，LGPL）
template <Precision P> class MatrixT;   // algebra_matrix.hpp（MIT）

// ══════════════════════════════════════════════════════════════════════════
// GpuBuffer — GPU 缓冲区 RAII 封装（PIMPL 声明；Impl 与 Vulkan 操作在引擎
// .so 侧：backend/compute_gpu_tensor_impl.hpp / compute_vk_backend.hpp）
//
// 布局拆分（跨界 ABI 稳定的关键）：消费方（Tensor 内联方法）需要访问的
// 字段/方法放**外层**（backend keepalive + write epoch，均不需要 Impl 完整）；
// Vulkan 句柄（VkDevice/VkBuffer/Allocation）藏在 Impl（引擎侧）。
// ══════════════════════════════════════════════════════════════════════════
class GpuBuffer
{
public:
    struct Impl;   // 引擎侧完整定义（Vulkan 句柄 + 分配）

    GpuBuffer() = default;
    // 引擎侧构造（Impl 由引擎创建并填充；keepalive 保证 backend 在张量
    // 存活期内不被析构）
    GpuBuffer(std::shared_ptr<Impl> impl, std::shared_ptr<GpuBackend> backend_keepalive)
        : impl_(std::move(impl)), backend_keepalive_(std::move(backend_keepalive))
    {
    }

    // 析构定义在引擎侧（GpuBackend 完整定义之后，延迟销毁语义，
    // 见 compute_vk_backend.hpp 的 defer_buffer_destroy）。
    ~GpuBuffer();

    GpuBuffer(GpuBuffer&& o) noexcept = default;
    GpuBuffer& operator=(GpuBuffer&& o) noexcept = default;
    GpuBuffer(const GpuBuffer&) = delete;
    GpuBuffer& operator=(const GpuBuffer&) = delete;

    // 设备域标识（生命周期 v3）：backend 对象地址。buffer 持 backend keepalive
    // ⇒ 张量存活期内该 backend 不会被析构、其地址不会被新 backend 复用。
    [[nodiscard]] DomainId backend_identity() const noexcept
    { return reinterpret_cast<DomainId>(backend_keepalive_.get()); }

    [[nodiscard]] std::uint64_t write_epoch() const noexcept { return write_epoch_; }
    void mark_content_changed() noexcept { ++write_epoch_; }

    // 不透明位宽句柄（VkDevice / VkBuffer；定义在引擎侧，Impl 完整才可访问）
    [[nodiscard]] std::uint64_t device_handle() const noexcept;
    [[nodiscard]] std::uint64_t impl() const noexcept;
    [[nodiscard]] bool valid() const noexcept;

    // 引擎侧便捷方法（返回真实 Vk 类型；仅引擎 .so 编译单元可见）
    [[nodiscard]] std::shared_ptr<Impl> shared_impl() const noexcept { return impl_; }

    // 创建 Device Local / Host Visible 缓冲区（引擎侧实现，Vulkan 操作；
    // usage 恒补 SHADER_DEVICE_ADDRESS（P3 实例表按设备地址引用张量缓冲））
    [[nodiscard]] static Result<GpuBuffer>
    create_device_local(std::uint64_t device, MemoryPool& pool, std::size_t byte_count,
                        std::uint32_t usage, std::shared_ptr<GpuBackend> backend_keepalive);
    [[nodiscard]] static Result<GpuBuffer>
    create_host_visible(std::uint64_t device, MemoryPool& pool, std::size_t byte_count,
                        std::uint32_t usage, std::shared_ptr<GpuBackend> backend_keepalive);

private:
    std::shared_ptr<Impl> impl_;
    // backend keepalive（外层）：GpuBuffer 可能晚于 GpuEngine 析构，持强引用
    // 后析构时可安全访问 backend（延迟销毁队列 / memory pool）。shared_ptr
    // 成员不需要 GpuBackend 完整类型（deleter 在创建点捕获）。
    std::shared_ptr<GpuBackend> backend_keepalive_;
    std::uint64_t write_epoch_ = 0;
};

// ══════════════════════════════════════════════════════════════════════════
// GpuTensorT<P> — GPU 矩阵抽象（多精度；声明 + 无 Vulkan 内联）
// buffer 字节数 = rows * cols * sizeof(elem<P>)（f32=4B / f16=2B）
// 需要 GpuBackend 的方法（from_matrix / create_* / to_matrix）在引擎侧实现
// （backend/compute_gpu_tensor_impl.hpp，LGPL）。
// ══════════════════════════════════════════════════════════════════════════
template <Precision P> class GpuTensorT
{
private:
    std::shared_ptr<GpuBuffer> buffer_;
    std::size_t rows_ = 0;
    std::size_t cols_ = 0;

public:
    GpuTensorT() = default;

    GpuTensorT(std::shared_ptr<GpuBuffer> buffer, std::size_t rows, std::size_t cols)
        : buffer_(std::move(buffer)), rows_(rows), cols_(cols)
    {
    }

    [[nodiscard]] static Result<GpuTensorT<P>> from_matrix(const MatrixT<P>& cpu_mat,
                                                           GpuBackend& backend);
    [[nodiscard]] static Result<GpuTensorT<P>> create_empty(std::size_t rows, std::size_t cols,
                                                            GpuBackend& backend);
    [[nodiscard]] static Result<GpuTensorT<P>>
    create_host_visible_empty(std::size_t rows, std::size_t cols, GpuBackend& backend);
    [[nodiscard]] Result<MatrixT<P>> to_matrix(GpuBackend& backend) const;

    [[nodiscard]] std::size_t rows() const noexcept { return rows_; }
    [[nodiscard]] std::size_t cols() const noexcept { return cols_; }
    [[nodiscard]] bool valid() const noexcept { return buffer_ && buffer_->valid(); }
    [[nodiscard]] const GpuBuffer& buffer() const noexcept { return *buffer_; }
    // 底层 buffer 的共享句柄（把同一 buffer 以另一种元素类型重贴标签用，
    // 见 run_fused_gpu 的 out_f16：in-kernel f16 的输出是 f16 字节布局）
    [[nodiscard]] std::shared_ptr<GpuBuffer> shared_buffer() const noexcept { return buffer_; }

    // 零拷贝 reshape：共享底层 buffer，只改变形状元数据
    [[nodiscard]] GpuTensorT<P> with_shape(std::size_t new_rows, std::size_t new_cols) const
    { return GpuTensorT<P>(buffer_, new_rows, new_cols); }
};

using GpuTensor = GpuTensorT<Precision::F32>;
using GpuTensorF32 = GpuTensorT<Precision::F32>;
using GpuTensorF16 = GpuTensorT<Precision::F16>;

} // namespace nn
