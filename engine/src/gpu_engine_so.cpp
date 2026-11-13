// ═══════════════════════════════════════════════════════════════════════════
//  gpu_engine_so.cpp — libnn-gpu.so（LGPL）
//
//  与 cpu_engine_so.cpp 同构：真实 nn::GpuEngine + Vulkan backend + 原语 shader
//  编进本 .so，经 MIT 抽象层接口 C++ 直出。GPU 初始化失败（无可用设备）返回
//  nullptr，消费方可回退 CPU。
// ═══════════════════════════════════════════════════════════════════════════

#include <memory>
#include <string>

#include <neuralnet.cpp/compute_engine.hpp>     // MIT 抽象层接口（本 .so 实现它）
#include <neuralnet.cpp/compute_gpu_engine.hpp> // 真实 GpuEngine 实现（编入本 .so）

// ── 跨模块剖析器挂钩（core_profile.hpp 的状态分裂修复）──────────────────────
// 同 cpu_engine_so.cpp：宿主按模块句柄 GetProcAddress 本符号（不经链接器）。
extern "C" void nn_prof_set_hooks(void* state_fn, void* stack_fn)
{
    ::nn::prof::detail::set_host_hooks(state_fn, stack_fn);
}

#ifdef NN_HAS_VULKAN

namespace
{
thread_local std::string g_err;
}

namespace nn
{

std::shared_ptr<ComputeEngine> create_gpu_engine(const char* device)
{
    auto backend = GpuBackend::shared();
    auto r = backend->initialize(device ? device : "");
    if (!r)
    {
        g_err = "GPU 初始化失败：" + r.error().message;
        return nullptr;
    }
    auto e = GpuEngine::create(*backend);   // shared_ptr<GpuEngine> → shared_ptr<ComputeEngine>
    if (!e) { g_err = "GPU 引擎创建失败"; return nullptr; }
    return e;
}

const char* last_error()
{
    return g_err.empty() ? nullptr : g_err.c_str();
}

} // namespace nn

#else // !NN_HAS_VULKAN

namespace nn
{
std::shared_ptr<ComputeEngine> create_gpu_engine(const char* device)
{
    (void)device;
    return nullptr;   // 本构建无 Vulkan
}
const char* last_error() { return "本构建无 Vulkan（NN_ENABLE_GPU 构建时未启用）"; }
} // namespace nn

#endif
