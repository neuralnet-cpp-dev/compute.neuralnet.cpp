#pragma once

#include <charconv>
#include <cstddef>
#include <cstdio>   // fputs（错误打印）
#include <cstdlib>  // for strtof/strtod/strtold（浮点解析回退）+ abort/exit
                    // + Windows：_set_abort_behavior/_set_error_mode（abort 弹窗静音）
#include <expected>
#include <source_location>
#include <string>
#include <string_view>
#include <type_traits>

namespace nn
{

// ── 错误类型（C++23 std::expected）─────────────────────────────────────
// 所有公共 API 使用 Result<T> 返回错误，不抛异常。
struct Error
{ std::string message; };

template <typename T> using Result = std::expected<T, Error>;

// ── [[nodiscard]] 包装：让丢弃 Result 的调用在编译期告警 ──────────────
// 用法：auto r = nn::check_result(foo());  // 若丢弃 r 会告警
// 注：Result<T> 本身是 std::expected 的别名，无法在别名声明上加属性，
// 此处提供显式 nodiscard 包装用于关键路径。
template <typename T> [[nodiscard]] constexpr Result<T> check_result(Result<T> r) noexcept
{ return r; }

// ── 错误处理宏族的实现支撑（detail；调用方只用下面的宏）────────────────────
namespace detail
{

// 终止策略：铁律 #1 禁异常，这里只有 abort / exit 两条路，库头不出现 throw。
// ⚠ Windows：CRT 在 abort() 时弹**模态**"abort() has been called"对话框
//   （GUI 弹窗不进 stdout/stderr → 日志/审计全看不见，测试与脚本被卡死等人点；
//   实测 `error_macro_test` 的 abort 负例子进程即弹了此框）。错误
//   信息本函数已打印到 stderr，可见性不受影响 → 终止前把 CRT 运行时错误/abort
//   消息与 WER 上报全部改道/静音，保证 abort 恒为"打印后立即终止"。
[[noreturn]] inline void fail_abort(const std::string& msg)
{
    std::fputs(msg.c_str(), stderr);
    std::fputc('\n', stderr);
    std::fflush(stderr);
#if defined(_WIN32)
    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
    _set_error_mode(_OUT_TO_STDERR);
#endif
    std::abort();
}

[[noreturn]] inline void fail_exit(const std::string& msg, int code)
{
    std::fputs(msg.c_str(), stderr);
    std::fputc('\n', stderr);
    std::fflush(stderr);
    std::exit(code);
}

// 统一打印格式（单独成函数 = 可被单测直接覆盖）：
//     [nn] <语境>: <错误详情> (<表达式>) at <file>:<line>
// 详情位为空（bool 条件失败）时回落为 "check failed"。
[[nodiscard]] inline std::string check_message(const char* what, std::string_view ctx,
                                               std::string_view detail, std::source_location loc)
{
    std::string s = "[nn]";
    if (!ctx.empty())
    {
        s += ' ';
        s += ctx;
        s += ':';
    }
    s += ' ';
    s += detail.empty() ? std::string_view{"check failed"} : detail;
    s += " (";
    s += (what != nullptr && what[0] != '\0') ? what : "?";
    s += ") at ";
    s += loc.file_name();
    s += ':';
    s += std::to_string(loc.line());
    return s;
}

// 错误链路加语境（NN_TRY_MSG）：ctx 非空时前缀到原消息上，逐层可读。
[[nodiscard]] inline Error contextualize(Error e, std::string_view ctx)
{
    if (ctx.empty()) return e;
    std::string m(ctx);
    m += ": ";
    m += e.message;
    return Error{std::move(m)};
}

// ── 解包 + 终止（NN_UNWRAP）：Result<T> → T（失败 abort）────────────────────
// **精确 Result<T> 重载，不用泛型 T&&**：`std::expected` 自带 `operator bool`，
// 若形参是泛型 T&&，会与下面的 bool 重载形成部分排序歧义，clang 改走上下文
// 转换、把整个 Result 当返回值（实测编译期才暴露）。三重载按形参类型精确
// 划分：Result<T>（非 void）/ Result<void> / bool。
// 非 void 带 nodiscard（丢弃解出的值会告警）；void 形态不带（否则纯副作用
// 调用 NN_UNWRAP(f()); 会误告警）。按值接 + std::move 取值：结果可能持有不可
// 拷贝类型（如 std::unique_ptr），调用侧右值/左值都能绑。
template <typename T>
    requires (!std::is_void_v<T>)
[[nodiscard]] inline T check_value(Result<T>&& r, const char* what, std::string_view ctx,
                                   std::source_location loc)
{
    if (!r) fail_abort(check_message(what, ctx, r.error().message, loc));
    return std::move(r).value();
}

template <typename T>
    requires (!std::is_void_v<T>)
[[nodiscard]] inline T check_value(Result<T> const& r, const char* what, std::string_view ctx,
                                   std::source_location loc)
{
    if (!r) fail_abort(check_message(what, ctx, r.error().message, loc));
    return std::move(r).value();   // 只有失败才走到这；成功路径由 rvalue 重载接走
}

inline void check_value(Result<void> r, const char* what, std::string_view ctx,
                        std::source_location loc)
{
    if (!r) fail_abort(check_message(what, ctx, r.error().message, loc));
}

inline void check_value(bool ok, const char* what, std::string_view ctx, std::source_location loc)
{
    if (!ok) fail_abort(check_message(what, ctx, {}, loc));
}

// ── 解包 + std::exit(code)（NN_EXIT）：CLI/参数这类"用户输入错误"用它保住
//    退出码（内部不变量用 NN_UNWRAP 的 abort）。**恒不带 nodiscard** —— 最常见
//    用法是纯退出检查（结果就此丢弃），带 nodiscard 会误告警。
//    形参用转发引用 + if constexpr 取错误详情：既接 Result<T>（含 void 与不可
//    拷贝的 unique_ptr 载体），也接裸指针 / unique_ptr / any 上下文可判定类型
//    （原手写 `if (!tok) { …; return 1; }` 的 tok 就是这一类）。
template <typename T>
    requires (!std::is_same_v<std::decay_t<T>, bool>)
inline void check_exit(T&& r, const char* what, std::string_view ctx, int code,
                       std::source_location loc)
{
    if (!r)
    {
        if constexpr (requires { r.error().message; })
            fail_exit(check_message(what, ctx, r.error().message, loc), code);
        else fail_exit(check_message(what, ctx, {}, loc), code);
    }
}

inline void check_exit(bool ok, const char* what, std::string_view ctx, int code,
                       std::source_location loc)
{
    if (!ok) fail_exit(check_message(what, ctx, {}, loc), code);
}

} // namespace detail
} // namespace nn

// ── 错误处理宏族 → 迁至宏岛头 nn_error_macros.hpp（P2/D3，docs/internal/35）──
// C++ 模块不可导出宏；import 消费方与模块 GMF 一律文本包含宏岛头拿宏。
// ⚠ **必须在文件作用域 include**：岛头尾部还会拉入 core_assert.hpp，
//   若在 `namespace nn` 内 include，被包含头的 namespace nn 会变成 nn::nn
//   （实测 `no member named 'Error' in namespace 'nn::nn'` + fatal_at 找不到
//   fail_abort）。故先关闭本文件的 namespace nn，include 完再重开；
//   包含点仍保持「detail 检查函数定义之后、parse_number 使用 NN_FAIL 之前」
//   的语义位置（见岛头文件头注释的次序说明）。
#include "../nn_error_macros.hpp"

namespace nn
{

// ── 数字解析（std::from_chars，不抛异常） ────────────────────────────
// 替代 std::stoi/stod/stoul/stoull 等会抛异常的函数。
// 用法：auto v = nn::parse_number<int>("123");
//       if (!v) return std::unexpected(v.error());
template <typename T>
    requires std::is_arithmetic_v<T>
[[nodiscard]] inline Result<T> parse_number(std::string_view s) noexcept
{
    // 去除首尾空白（from_chars 不容忍前后空白）
    while (!s.empty() &&
           (s.front() == ' ' || s.front() == '\t' || s.front() == '\n' || s.front() == '\r'))
        s.remove_prefix(1);
    while (!s.empty() &&
           (s.back() == ' ' || s.back() == '\t' || s.back() == '\n' || s.back() == '\r'))
        s.remove_suffix(1);

    if (s.empty()) NN_FAIL("empty numeric string");
    T value{};
    if constexpr (std::is_floating_point_v<T>)
    {
        // 浮点：libc++ 部分版本缺少浮点 std::from_chars（__charconv/from_chars_integral.h
        // 只有整型版），会选中被删除的 bool 重载而编译失败。故用 C 的 strtoX
        // （全平台/全标准库都有、不抛异常）。s 已去除首尾空白，要求整串被解析。
        std::string tmp(s);  // strtoX 需要 null 结尾
        char* p = nullptr;
        if constexpr (std::is_same_v<T, float>) value = std::strtof(tmp.c_str(), &p);
        else if constexpr (std::is_same_v<T, double>) value = std::strtod(tmp.c_str(), &p);
        else value = std::strtold(tmp.c_str(), &p);
        if (p != tmp.c_str() + tmp.size()) NN_FAIL("invalid number: " + std::string(s));
    }
    else
    {
        // string_view::end() 即指向末尾后一位置的 const char*，无需手动指针算术
        auto [ptr, ec] = std::from_chars(s.data(), s.data() + s.size(), value);
        if (ec != std::errc{} || ptr != s.data() + s.size())
            NN_FAIL("invalid number: " + std::string(s));
    }
    return value;
}

} // namespace nn
