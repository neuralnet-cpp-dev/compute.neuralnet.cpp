// ═══════════════════════════════════════════════════════════════════════════
//  compute_engine.hpp — 抽象层（MIT License）
//
//  这是引擎的**公共 C++ 接口**：消费方只 include 本头 + 链接引擎 .so
//  （libnn-cpu.so / libnn-gpu.so，LGPL），即可获得完整引擎能力。
//
//  许可边界：
//    · 本文件 = MIT —— 只有接口声明 + 值类型（无任何引擎实现代码）。
//    · 引擎 .so = LGPL —— 真正的 CpuEngine/GpuEngine 实现编在可替换的 .so 里，
//      消费方动态链接（不把 LGPL 代码内联进自身二进制，且可替换 .so），
//      因此抽象层与消费方不被 LGPL 传染。
//
//  跨 .so 边界协议（C++ 直出，区别于 C ABI）：
//    · 引擎句柄：std::unique_ptr<ComputeEngine>，工厂 create_*_engine() 由 .so 导出。
//    · 张量：Tensor 是轻量值类型（opaque 句柄 + 缓存维度），真实存储由引擎托管。
//      引擎持有全部存活张量，引擎析构即连带回收；销毁单个张量用 destroy_tensor。
//      Tensor 拷贝 = 别名（视图），不是深拷贝；同一底层张量只销毁一次。
//    · 错误：last_error() 返回 .so 内缓冲的只读文本（下一次调用前有效，线程本地）。
//    · STL 仅 std::unique_ptr<ComputeEngine> 与 std::string（describe 文本）跨界；
//      二者要求消费方与 .so 用同编译器 / 同标准库 / 同 -fno-exceptions 编译。
// ═══════════════════════════════════════════════════════════════════════════
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

namespace neuralnet
{

// 精度（映射引擎 nn::Precision 的核心取值：F16 / F32）
enum class Precision : std::uint8_t
{
    F16 = 0,
    F32 = 1,
};

// ── Tensor：跨边界的张量值类型 ─────────────────────────────────────────
// 持有引擎侧真实张量的不透明句柄 + 缓存维度。形状不可变，故缓存不失效。
// 拷贝 = 别名；所有权归引擎（engine 析构连带回收 / destroy_tensor 单个释放）。
class Tensor
{
public:
    Tensor() = default;

    bool   valid() const { return handle_ != nullptr; }
    std::size_t rows() const { return rows_; }
    std::size_t cols() const { return cols_; }
    Precision   precision() const { return prec_; }

    // 引擎侧不透明句柄（.so 适配器用它取真实张量；普通消费方忽略，勿解引用）。
    void* native_handle() const { return handle_; }

    // 引擎实现侧构造（.so 用它回填句柄 + 维度）。普通消费方不应直接调用。
    explicit Tensor(void* handle, std::size_t rows, std::size_t cols, Precision prec)
        : handle_(handle), rows_(rows), cols_(cols), prec_(prec)
    {
    }

private:
    void*       handle_ = nullptr;
    std::size_t rows_ = 0;
    std::size_t cols_ = 0;
    Precision   prec_ = Precision::F32;
};

// ── ComputeEngine：抽象引擎接口（纯虚；实现编在引擎 .so）──────────────
class ComputeEngine
{
public:
    virtual ~ComputeEngine() = default;

    // 稳定的单行描述文本（.so 内缓冲，下一次调用前有效）。
    virtual const char* describe() const = 0;

    // ── 张量生命周期（张量归引擎托管）────────────────────────────
    virtual Tensor create_tensor(std::size_t rows, std::size_t cols, Precision prec) = 0;
    virtual Tensor from_matrix(const float* data, std::size_t rows, std::size_t cols,
                               Precision prec) = 0;
    // 下载到 row-major f32 缓冲。返回 false = 失败（last_error）。
    // out == nullptr 时只回填 out_n（查形状），不写数据。
    virtual bool to_matrix(const Tensor& t, float* out, std::size_t cap, std::size_t* out_n) = 0;
    virtual void destroy_tensor(Tensor& t) = 0;

    // ── 原语（返回新建张量，归引擎托管；失败返回 invalid Tensor）──────
    virtual Tensor matmul(const Tensor& A, const Tensor& B, bool transA, bool transB) = 0;
    virtual Tensor row_reduce_sum(const Tensor& A) = 0;
    virtual Tensor col_reduce_sum(const Tensor& A) = 0;
    virtual Tensor col_reduce_max(const Tensor& A) = 0;
    virtual Tensor transpose(const Tensor& A) = 0;
    virtual Tensor clone(const Tensor& A) = 0;
    // 原地操作。返回 false = 失败（last_error）。
    virtual bool add_inplace(Tensor& A, const Tensor& B) = 0;
    virtual bool scale_inplace(Tensor& A, float s) = 0;
    virtual bool zero(Tensor& A) = 0;

    // ── 批处理录制窗口（GPU 录制到 command buffer；CPU 为 no-op）──────
    virtual bool begin_batch() = 0;
    virtual bool end_batch() = 0;

    // ── 表达式求值（AOT/JIT 融合 shader）────────────────────────────
    // spec = 序列化的 ExprSpec 字节（编码协议见引擎侧 kernel_cache.hpp）；
    // inputs = 输入张量数组；prec_sig bit16 = 输出 f16。返回新建张量。
    virtual Tensor eval_expr(const std::uint8_t* spec, std::size_t len, const Tensor* inputs,
                             std::size_t nin, std::uint32_t prec_sig, std::size_t rows,
                             std::size_t cols) = 0;

    // 最近一次错误的只读文本（.so 内缓冲，下一次调用前有效；无错误返回 nullptr）。
    virtual const char* last_error() const = 0;
};

// ── 工厂（由引擎 .so 导出实现）───────────────────────────────────────
// create_cpu_engine()：libnn-cpu.so。确定性、无外部依赖。
std::unique_ptr<ComputeEngine> create_cpu_engine();
// create_gpu_engine(device)：libnn-gpu.so。device = 设备名子串（空 = 自动，
// 同 NN_VULKAN_DEVICE）；无可用 Vulkan 设备返回 nullptr（last_error 说明），
// 消费方可回退 CPU。
std::unique_ptr<ComputeEngine> create_gpu_engine(const char* device);

// 最近一次引擎级错误的只读文本（线程本地；**create_*_engine 失败返回 nullptr 时
// 也可用本函数查询**）。下一次调用前有效；无错误返回 nullptr。
const char* last_error();

} // namespace neuralnet
