// ═══════════════════════════════════════════════════════════════════════════
//  cpu_engine_so.cpp — libnn-cpu.so（LGPL）
//
//  真实引擎 nn::CpuEngine（header-only 实现，LGPL）编译进本 .so，经 MIT 抽象层
//  接口头（compute_engine.hpp：完整 ComputeEngine 接口 + Tensor 值类型跨界）
//  C++ 直出。消费方只 include 抽象层头 + 链接本 .so；.so 可替换 → LGPL 承诺兑现。
//
//  本 TU 不做任何句柄映射/序列化——Tensor 是真实值类型（布局在 MIT 头，两侧同
//  编译器/旗标编译 → ABI 一致），虚方法分派直接落到 .so 内的 CpuEngine 实现。
// ═══════════════════════════════════════════════════════════════════════════

#include <memory>
#include <string>

#include <neuralnet.cpp/compute_engine.hpp>     // MIT 抽象层接口（本 .so 实现它）
#include <neuralnet.cpp/compute_cpu_engine.hpp> // 真实 CpuEngine 实现（编入本 .so）

namespace
{
thread_local std::string g_err;
}

namespace nn
{

std::shared_ptr<ComputeEngine> create_cpu_engine()
{
    auto e = CpuEngine::create();   // shared_ptr<CpuEngine> → shared_ptr<ComputeEngine>
    if (!e) { g_err = "CPU 引擎创建失败"; return nullptr; }
    return e;
}

const char* last_error()
{
    return g_err.empty() ? nullptr : g_err.c_str();
}

} // namespace nn

// ── 跨模块剖析器挂钩（core_profile.hpp 的状态分裂修复）──────────────────────
// 宿主（主程序/测试 EXE）在**首次**进入剖析路径时按模块句柄 GetProcAddress
// 本符号并调用——不经链接器直接引用（两引擎同名导出，import 符号会撞）。
extern "C" void nn_prof_set_hooks(void* state_fn, void* stack_fn)
{
    ::nn::prof::detail::set_host_hooks(state_fn, stack_fn);
}
