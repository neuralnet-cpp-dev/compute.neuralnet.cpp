// ═══════════════════════════════════════════════════════════════════════════
//  gpu_engine_so.cpp — libnn-gpu.so 的 C++ 导出实现（LGPL）
//
//  与 cpu_engine_so.cpp 同构：把真实 nn::GpuEngine + Vulkan backend + 原语
//  shader + kernelgen 编进独立 .so，以 MIT 抽象层 neuralnet::ComputeEngine
//  接口直出 C++。本 .so 独立可替换 → LGPL 承诺兑现。
// ═══════════════════════════════════════════════════════════════════════════

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>

#include <neuralnet.cpp/compute_engine.hpp>   // MIT 抽象层（本 .so 实现它）
#include <neuralnet.cpp/nn.hpp>                // 真实引擎（LGPL，编入本 .so）
#include <neuralnet.cpp/kernel_cache.hpp>      // decode_expr_spec（eval_expr 边界载荷）

#ifdef NN_HAS_VULKAN

namespace
{

thread_local std::string g_err;

inline void set_err(const std::string& msg) { g_err = msg; }

inline neuralnet::Precision to_mit(nn::Precision p)
{
    return p == nn::Precision::F16 ? neuralnet::Precision::F16 : neuralnet::Precision::F32;
}
inline nn::Precision to_real(neuralnet::Precision p)
{
    return p == neuralnet::Precision::F16 ? nn::Precision::F16 : nn::Precision::F32;
}

// GPU 引擎适配器：实现 MIT 抽象接口，转发到真实 nn::GpuEngine。
class GpuEngineAdapter : public neuralnet::ComputeEngine
{
public:
    explicit GpuEngineAdapter(std::shared_ptr<nn::GpuBackend> backend)
        : eng_(nn::GpuEngine::create(*backend))
    {
    }
    ~GpuEngineAdapter() override = default;

    bool ok() const { return eng_ != nullptr; }

    const char* describe() const override
    {
        desc_cache_ = eng_->describe().text();
        return desc_cache_.c_str();
    }

    neuralnet::Tensor create_tensor(std::size_t rows, std::size_t cols,
                                    neuralnet::Precision prec) override
    {
        auto r = eng_->try_create_tensor(rows, cols, to_real(prec));
        return finish(std::move(r));
    }

    neuralnet::Tensor from_matrix(const float* data, std::size_t rows, std::size_t cols,
                                  neuralnet::Precision prec) override
    {
        nn::Matrix m(rows, cols);
        if (data)
            std::memcpy(m.span().data(), data, rows * cols * sizeof(float));
        else
            std::memset(m.span().data(), 0, rows * cols * sizeof(float));
        auto r = eng_->from_matrix(m, to_real(prec));
        return finish(std::move(r));
    }

    bool to_matrix(const neuralnet::Tensor& t, float* out, std::size_t cap,
                   std::size_t* out_n) override
    {
        if (!t.valid()) { set_err("invalid tensor"); return false; }
        auto r = eng_->to_matrix(real(t));
        if (!r) { set_err(r.error().message); return false; }
        const nn::Matrix& m = *r;
        const std::size_t n = m.rows() * m.cols();
        if (out_n) *out_n = n;
        if (out == nullptr) return true;
        if (cap < n) { set_err("to_matrix 缓冲容量不足"); return false; }
        std::memcpy(out, m.span().data(), n * sizeof(float));
        return true;
    }

    void destroy_tensor(neuralnet::Tensor& t) override
    {
        if (!t.valid()) return;
        tensors_.erase(t.native_handle());
        t = neuralnet::Tensor{};
    }

    neuralnet::Tensor matmul(const neuralnet::Tensor& A, const neuralnet::Tensor& B, bool transA,
                             bool transB) override
    {
        if (!A.valid() || !B.valid()) { set_err("invalid tensor"); return {}; }
        auto r = eng_->matmul(real(A), real(B), transA, transB, nn::Precision::F32);
        return finish(std::move(r));
    }
    neuralnet::Tensor row_reduce_sum(const neuralnet::Tensor& A) override
    {
        if (!A.valid()) { set_err("invalid tensor"); return {}; }
        return finish(eng_->row_reduce_sum(real(A)));
    }
    neuralnet::Tensor col_reduce_sum(const neuralnet::Tensor& A) override
    {
        if (!A.valid()) { set_err("invalid tensor"); return {}; }
        return finish(eng_->col_reduce_sum(real(A)));
    }
    neuralnet::Tensor col_reduce_max(const neuralnet::Tensor& A) override
    {
        if (!A.valid()) { set_err("invalid tensor"); return {}; }
        return finish(eng_->col_reduce_max(real(A)));
    }
    neuralnet::Tensor transpose(const neuralnet::Tensor& A) override
    {
        if (!A.valid()) { set_err("invalid tensor"); return {}; }
        return finish(eng_->transpose(real(A)));
    }
    neuralnet::Tensor clone(const neuralnet::Tensor& A) override
    {
        if (!A.valid()) { set_err("invalid tensor"); return {}; }
        return finish(eng_->clone(real(A)));
    }

    bool add_inplace(neuralnet::Tensor& A, const neuralnet::Tensor& B) override
    {
        if (!A.valid() || !B.valid()) { set_err("invalid tensor"); return false; }
        auto r = eng_->add_inplace(real(A), real(B));
        if (!r) { set_err(r.error().message); return false; }
        return true;
    }
    bool scale_inplace(neuralnet::Tensor& A, float s) override
    {
        if (!A.valid()) { set_err("invalid tensor"); return false; }
        auto r = eng_->scale_inplace(real(A), s);
        if (!r) { set_err(r.error().message); return false; }
        return true;
    }
    bool zero(neuralnet::Tensor& A) override
    {
        if (!A.valid()) { set_err("invalid tensor"); return false; }
        auto r = eng_->zero(real(A));
        if (!r) { set_err(r.error().message); return false; }
        return true;
    }

    bool begin_batch() override
    {
        auto r = eng_->begin_batch();
        if (!r) { set_err(r.error().message); return false; }
        return true;
    }
    bool end_batch() override
    {
        auto r = eng_->end_batch();
        if (!r) { set_err(r.error().message); return false; }
        return true;
    }

    neuralnet::Tensor eval_expr(const std::uint8_t* spec, std::size_t len, const neuralnet::Tensor* inputs,
                                std::size_t nin, std::uint32_t prec_sig, std::size_t rows,
                                std::size_t cols) override
    {
        if (!spec || (!inputs && nin > 0)) { set_err("null 参数"); return {}; }
        auto spec_r = nn::fused::decode_expr_spec(spec, len);
        if (!spec_r) { set_err(spec_r.error().message); return {}; }
        std::vector<nn::Tensor> ins;
        ins.reserve(nin);
        for (std::size_t i = 0; i < nin; ++i) ins.push_back(real(inputs[i]));
        auto r = eng_->eval_expr(*spec_r, ins, rows, cols,
                                 (prec_sig & 0x10000u) ? nn::Precision::F16 : nn::Precision::F32);
        return finish(std::move(r));
    }

    const char* last_error() const override
    {
        return g_err.empty() ? nullptr : g_err.c_str();
    }

private:
    static nn::Tensor& real(const neuralnet::Tensor& t)
    {
        return *static_cast<nn::Tensor*>(t.native_handle());
    }

    neuralnet::Tensor finish(nn::Result<nn::Tensor>&& r)
    {
        if (!r) { set_err(r.error().message); return {}; }
        auto* p = new nn::Tensor(std::move(*r));
        void* key = p;
        tensors_[key] = std::unique_ptr<nn::Tensor>(p);
        return neuralnet::Tensor(key, p->rows(), p->cols(), to_mit(p->precision()));
    }

    std::shared_ptr<nn::ComputeEngine> eng_;
    mutable std::string desc_cache_;
    std::unordered_map<void*, std::unique_ptr<nn::Tensor>> tensors_;
};

} // namespace

namespace neuralnet
{

std::unique_ptr<ComputeEngine> create_gpu_engine(const char* device)
{
    auto backend = nn::GpuBackend::shared();
    std::string sel = device ? device : "";
    auto r = backend->initialize(sel);
    if (!r)
    {
        g_err = "GPU 初始化失败：" + r.error().message;
        return nullptr;
    }
    auto* a = new GpuEngineAdapter(backend);
    if (!a->ok()) { delete a; return nullptr; }
    return std::unique_ptr<ComputeEngine>(a);
}

const char* last_error()
{
    return g_err.empty() ? nullptr : g_err.c_str();
}

} // namespace neuralnet

#else // !NN_HAS_VULKAN

namespace neuralnet
{
std::unique_ptr<ComputeEngine> create_gpu_engine(const char* device)
{
    (void)device;
    return nullptr;   // 本构建无 Vulkan
}

const char* last_error()
{
    return "本构建无 Vulkan（NN_ENABLE_GPU 构建时未启用）";
}
} // namespace neuralnet

#endif
