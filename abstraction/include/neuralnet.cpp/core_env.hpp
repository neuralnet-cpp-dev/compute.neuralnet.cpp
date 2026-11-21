#pragma once

// ── core_env.hpp — 探针门层（raw 读取已迁 exstl::env）──────────────────────
// 平台分支（MSVC _dupenv_s vs getenv）唯一真源 = <exstl/env.hpp>。
// 本头只保留**探针门**包装：`NN_ENABLE_PROBES=OFF` 时 probe_* 恒 false /
// 空串且**不求值**读取——默认构建里诊断开关零分支、零字符串、零系统调用。
//
// 行为开关（NN_VULKAN_DEVICE / NN_POOL_* 等）直接用 exstl::env::flag/str/integer，
// 调用点自行 `static const bool on = ...` 缓存。
// ─────────────────────────────────────────────────────────────────────────

#include <string>

#include <exstl/env.hpp>

#include <neuralnet.cpp/core_config.hpp>   // NN_PROBES_ENABLED（探针编译门）

namespace nn::detail
{

// ── 探针门下的读取（OFF 时恒 false / 空串，**不求值**环境读取）─────────────

[[nodiscard]] inline bool probe_flag_raw(const char* name) noexcept
{
#if NN_PROBES_ENABLED
    return exstl::env::flag(name);
#else
    (void)name;
    return false;
#endif
}

[[nodiscard]] inline std::string probe_string_raw(const char* name) noexcept
{
#if NN_PROBES_ENABLED
    return exstl::env::str(name);
#else
    (void)name;
    return {};
#endif
}

} // namespace nn::detail
