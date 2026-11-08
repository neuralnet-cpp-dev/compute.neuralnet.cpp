// ── compute_gpu_tensor_impl.hpp ─────────────────────────────────────────────────
// GpuTensorT<P> 方法实现（需要 MatrixT<P> 和 GpuBackend 的完整定义）
// 此文件在 algebra_matrix.hpp 末尾自动包含，解决循环依赖问题。
// ─────────────────────────────────────────────────────────────────────────

#pragma once

#ifdef NN_HAS_VULKAN

namespace nn
{

// ── GpuBuffer 创建（Vulkan 操作；声明在 MIT 抽象层，Impl 定义见
//    compute_vk_backend.hpp 顶部）────────────────────────────────────────
inline GpuBuffer make_gpu_buffer(VkDevice device, VkBuffer buffer,
                                 const MemoryPool::Allocation& alloc, MemoryPool& pool,
                                 std::shared_ptr<GpuBackend> backend_keepalive)
{
    auto impl = std::make_shared<GpuBuffer::Impl>();
    impl->device = device;
    impl->buffer = buffer;
    impl->alloc = alloc;
    impl->pool = observer_ptr<MemoryPool>(pool);
    return GpuBuffer(std::move(impl), std::move(backend_keepalive));
}

Result<GpuBuffer>
GpuBuffer::create_device_local(std::uint64_t device_u, MemoryPool& pool, std::size_t byte_count,
                               std::uint32_t usage_u, std::shared_ptr<GpuBackend> backend_keepalive)
{
    VkDevice device = reinterpret_cast<VkDevice>(device_u);
    const VkBufferUsageFlags usage = static_cast<VkBufferUsageFlags>(usage_u);
    VkBufferCreateInfo buf_info{};
    buf_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buf_info.size = byte_count;
    buf_info.usage = usage;
    if (pool.device_address_alloc())
        buf_info.usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    buf_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer buffer = VK_NULL_HANDLE;
    VkResult res = vkCreateBuffer(device, &buf_info, nullptr, &buffer);
    if (res != VK_SUCCESS) NN_FAIL("vkCreateBuffer failed: " + std::to_string(res));
    VkMemoryRequirements mem_reqs;
    vkGetBufferMemoryRequirements(device, buffer, &mem_reqs);

    auto alloc_r = pool.allocate(mem_reqs, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (!alloc_r)
    {
        vkDestroyBuffer(device, buffer, nullptr);
        NN_TRY_CHECK(alloc_r);
    }

    res = vkBindBufferMemory(device, buffer, alloc_r->memory, alloc_r->offset);
    if (res != VK_SUCCESS)
    {
        pool.free(*alloc_r);
        vkDestroyBuffer(device, buffer, nullptr);
        NN_FAIL("vkBindBufferMemory failed: " + std::to_string(res));
    }

    return make_gpu_buffer(device, buffer, *alloc_r, pool, std::move(backend_keepalive));
}

Result<GpuBuffer>
GpuBuffer::create_host_visible(std::uint64_t device_u, MemoryPool& pool, std::size_t byte_count,
                               std::uint32_t usage_u, std::shared_ptr<GpuBackend> backend_keepalive)
{
    VkDevice device = reinterpret_cast<VkDevice>(device_u);
    const VkBufferUsageFlags usage = static_cast<VkBufferUsageFlags>(usage_u);
    VkBufferCreateInfo buf_info{};
    buf_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buf_info.size = byte_count;
    buf_info.usage =
        usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    if (pool.device_address_alloc())
        buf_info.usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    buf_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer buffer = VK_NULL_HANDLE;
    VkResult res = vkCreateBuffer(device, &buf_info, nullptr, &buffer);
    if (res != VK_SUCCESS) NN_FAIL("vkCreateBuffer failed: " + std::to_string(res));
    VkMemoryRequirements mem_reqs;
    vkGetBufferMemoryRequirements(device, buffer, &mem_reqs);

    VkMemoryPropertyFlags preferred, fallback;
    if (pool.is_uma())
    {
        preferred = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        fallback = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    }
    else
    {
        preferred = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        fallback = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    }

    auto alloc_r = pool.allocate(mem_reqs, preferred, fallback);
    if (!alloc_r)
    {
        vkDestroyBuffer(device, buffer, nullptr);
        NN_TRY_CHECK(alloc_r);
    }

    res = vkBindBufferMemory(device, buffer, alloc_r->memory, alloc_r->offset);
    if (res != VK_SUCCESS)
    {
        pool.free(*alloc_r);
        vkDestroyBuffer(device, buffer, nullptr);
        NN_FAIL("vkBindBufferMemory failed: " + std::to_string(res));
    }

    return make_gpu_buffer(device, buffer, *alloc_r, pool, std::move(backend_keepalive));
}

// 张量 buffer 的标准 usage 标志：
//   STORAGE_BUFFER: 绑定到 shader 进行 GPU 计算
//   TRANSFER_DST:   允许 vkCmdFillBuffer(zero) / vkCmdCopyBuffer(insert_rows) 写入
//   TRANSFER_SRC:   允许 vkCmdCopyBuffer(clone / slice_rows) 读取
static constexpr VkBufferUsageFlags TENSOR_BUFFER_USAGE = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                          VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                                          VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

template <Precision P>
inline Result<GpuTensorT<P>> GpuTensorT<P>::from_matrix(const MatrixT<P>& cpu_mat,
                                                        GpuBackend& backend)
{
    if (cpu_mat.empty()) NN_FAIL("Empty matrix");
    // buffer 字节数 = 元素数 × sizeof(elem<P>)（§6.3），**4 B 对齐**（见下）
    // ⚠ 缓冲字节数 4 B 对齐（P3 验收门 D 实测修复）：f16 按 2 B/元素，奇数
    //   元素数时 packHalf2x16 类写点（cast.comp 的 D[pairs-1] 等）的最后一个
    //   uint 会越出缓冲末尾 1~2 B —— 越界 store 被驱动丢弃 → **末元素永不写入**
    //   （实测 (65,17) f16 目标 idx1104 恒缺失；新分配的零内存把它掩盖成
    //   "新进程首跑正确"，池复用后残留旧值 → 静默错值/NaN）。f32/f64 天然
    //   4 B 对齐、零行为变化。
    const std::size_t byte_count = (cpu_mat.size() * sizeof(elem<P>) + 3u) & ~std::size_t{3};
    auto buf_res =
        GpuBuffer::create_device_local(reinterpret_cast<std::uint64_t>(backend.device().device()), backend.alloc_pool(), byte_count,
                                       TENSOR_BUFFER_USAGE, backend.keepalive());
    NN_TRY_CHECK(buf_res);
    auto tensor = GpuTensorT<P>(std::make_shared<GpuBuffer>(std::move(*buf_res)), cpu_mat.rows(),
                                cpu_mat.cols());

    // 同精度跨设备 = 原始字节拷贝（§6.5）
    auto upload_res = backend.upload_blocking(tensor, cpu_mat.span());
    NN_TRY_CHECK(upload_res);
    return tensor;
}

template <Precision P>
inline Result<GpuTensorT<P>> GpuTensorT<P>::create_empty(std::size_t rows, std::size_t cols,
                                                         GpuBackend& backend)
{
    // buffer 字节数 = 元素数 × sizeof(elem<P>)（§6.3），**4 B 对齐**（见下）
    // ⚠ 缓冲字节数 4 B 对齐（P3 验收门 D 实测修复）：f16 按 2 B/元素，奇数
    //   元素数时 packHalf2x16 类写点（cast.comp 的 D[pairs-1] 等）的最后一个
    //   uint 会越出缓冲末尾 1~2 B —— 越界 store 被驱动丢弃 → **末元素永不写入**
    //   （实测 (65,17) f16 目标 idx1104 恒缺失；新分配的零内存把它掩盖成
    //   "新进程首跑正确"，池复用后残留旧值 → 静默错值/NaN）。f32/f64 天然
    //   4 B 对齐、零行为变化。
    // ⚠ 二次放大溢出防线：rows*cols 由 check_shape_ 拦下，但再乘 sizeof(elem<P>)
    //   仍可能回绕（元素数接近 size_t 上限时）→ 按回绕后的字节数分配即越界。
    //   判据 cnt*sizeof ≤ SIZE_MAX-3（对齐最多再加 3）。
    constexpr std::size_t k_elem_bytes = sizeof(elem<P>);
    constexpr std::size_t k_max_size = ~std::size_t{0};
    if (rows != 0 && cols > (k_max_size - 3) / k_elem_bytes / rows)
        NN_FAIL("GpuTensorT::create_empty: 字节数溢出（rows*cols*sizeof(elem) 超过 size_t 上限）");
    const std::size_t byte_count = (rows * cols * sizeof(elem<P>) + 3u) & ~std::size_t{3};
    // TRANSFER_DST_BIT: 允许 vkCmdFillBuffer (zero) 和 vkCmdCopyBuffer (insert_rows) 写入
    // TRANSFER_SRC_BIT: 允许 vkCmdCopyBuffer (clone/slice_rows) 读取
    auto buf_res =
        GpuBuffer::create_device_local(reinterpret_cast<std::uint64_t>(backend.device().device()), backend.alloc_pool(), byte_count,
                                       TENSOR_BUFFER_USAGE, backend.keepalive());
    NN_TRY_CHECK(buf_res);
    return GpuTensorT<P>(std::make_shared<GpuBuffer>(std::move(*buf_res)), rows, cols);
}

template <Precision P>
inline Result<GpuTensorT<P>>
GpuTensorT<P>::create_host_visible_empty(std::size_t rows, std::size_t cols, GpuBackend& backend)
{
    // Host Visible 存储（activation offload）：数据驻留主机可见内存（§6.6）。
    // 仅用于 vkCmdCopyBuffer 中转（TRANSFER_DST/SRC）。
    constexpr std::size_t k_elem_bytes = sizeof(elem<P>);
    constexpr std::size_t k_max_size = ~std::size_t{0};
    if (rows != 0 && cols > (k_max_size - 3) / k_elem_bytes / rows)
        NN_FAIL("GpuTensorT::create_host_visible_empty: 字节数溢出（rows*cols*sizeof(elem) 超过 size_t 上限）");
    const std::size_t byte_count = (rows * cols * sizeof(elem<P>) + 3u) & ~std::size_t{3};  // 4 B 对齐（同上）
    auto buf_res =
        GpuBuffer::create_host_visible(reinterpret_cast<std::uint64_t>(backend.device().device()), backend.alloc_pool(), byte_count,
                                       TENSOR_BUFFER_USAGE, backend.keepalive());
    NN_TRY_CHECK(buf_res);
    return GpuTensorT<P>(std::make_shared<GpuBuffer>(std::move(*buf_res)), rows, cols);
}

template <Precision P> inline Result<MatrixT<P>> GpuTensorT<P>::to_matrix(GpuBackend& backend) const
{
    if (!valid()) NN_FAIL("Invalid GpuTensor");
    MatrixT<P> cpu_mat(rows_, cols_);
    auto dl_res = backend.download_blocking(*this, cpu_mat.span());
    NN_TRY_CHECK(dl_res);
    return cpu_mat;
}

} // namespace nn

#endif // NN_HAS_VULKAN
