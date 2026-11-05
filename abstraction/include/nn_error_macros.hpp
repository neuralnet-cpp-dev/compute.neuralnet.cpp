#pragma once

// ── nn_error_macros — 宏岛头：全仓错误处理 / 断言宏的唯一真源 ─────────────
//
// 为什么独立成头（P2 / D3，docs/internal/35）：
//   C++ 标准规定**宏不可随模块导出**——`import nn.*` 的消费方看不到
//   NN_TRY / NN_UNWRAP / NN_ASSERT_* 等。宏段从 core_errors.hpp /
//   core_assert.hpp 迁到本头（不带 `neuralnet.cpp/` 前缀，避免被误当作
//   模块接口头），模块 GMF、include 消费方、src/ 的 import 消费方三侧
//   都靠**文本包含本头**拿到宏；core_errors.hpp / core_assert.hpp 原样
//   `#include` 本头 → 库内所有既有用法零改动。
//
// ⚠ 环形包含的合法次序（改动前必读）：
//   本头与 core_errors.hpp / core_assert.hpp 互相包含，靠「宏在前、
//   依赖在后 + 一律文件作用域」闭合，#pragma once 下三种入口都成立：
//     ① 入口 = 本头：先定义宏 → 尾部拉入 core_assert/core_errors；
//        被包含头回包含本头被 #pragma once 跳过，其后续正文（detail
//        检查函数、fatal_at、parse_number）正常推进——宏**定义**不依赖
//        这些函数，只有**展开点**需要，而展开点必经本头尾部的两个包含。
//     ② 入口 = core_errors：其在**文件作用域**（namespace nn 关闭之后）
//        包含本头 → 宏定义 → 尾部回包含跳过 → 重开 namespace nn 继续
//        parse_number（NN_FAIL 已可见）。
//     ③ 入口 = core_assert：→ 本头 → …（同 ② 走完）→ fatal_at 定义。
//   ⚠ 任何包含都**不得落在 namespace nn 内部**：被包含头的 namespace nn
//     会变成 nn::nn（实测 'Error' in 'nn::nn' / fail_abort 未声明）。
//   任何"把依赖包含挪到宏定义之前"的改动都会让入口 ① 在
//   core_errors::parse_number 处报 NN_FAIL 未定义——保持「宏在前」。
//
// 约束（继承铁律 #1）：终止策略只有 abort / exit，本头任何情况下不出现
// throw；宏体续行禁止 // 注释（铁律 #10，一律 /* */）。

// ── 错误处理宏族（铁律 #1 的全仓统一写法）───────────────────────────────
// `auto r = f(); if (!r) return std::unexpected(r.error());` 是本项目实施
// 铁律 #1 的主形态，在 L2 层重复约 500 次、占 Layer 代码近 9%。全仓统一为
// 下面三种语义（各自对应一种手写形态），外加一个错误源头语法糖：
//
//   ① 传播 —— 函数返回 Result<T> 时往上传：
//        NN_TRY(r, f())                声明 r 并检查 + return unexpected
//        NN_TRY_MSG(r, "读取配置", f()) 同上，失败时错误消息前缀 "读取配置: "
//        NN_TRY_CHECK(r)               检查已声明的 r
//
//   ② 解包继续 —— 任何上下文（main / void / 测试 helper），失败即终止：
//        NN_UNWRAP(f())                 Result<T>→T；Result<void> 与 bool 条件也可
//        NN_UNWRAP(f(), "上传批次")      带调用点语境
//      语义是 **unwrap**（成功解出值；失败即终止打印 message + file:line），
//      不是 assert。旧名 `NN_CHECK` / `NN_EASY_CHECK` 会诱导按断言直觉使用，
//      故更名；两者保留为等价别名（见下方），**新代码一律用 NN_UNWRAP**。
//      ⚠ 非 void 形态带 nodiscard：丢弃解出的值需显式 `(void)`。
//
//   ③ 解包退出 —— CLI/参数这类"用户输入错误"，保住退出码：
//        NN_EXIT(f(), 1)               失败打印后 std::exit(1)
//        NN_EXIT(f(), 1, "解析参数")
//
//   ④ 错误源头（可选糖）：NN_FAIL("msg") ≡ return std::unexpected(Error{"msg"})
//
// NN_TRY / NN_TRY_CHECK 仍是**纯语法压缩**：展开后与手写形态逐字等价（错误
// 消息、返回类型、语句结构都相同），不含任何控制流或所有权语义；展开为普通
// 语句（不用 do{}while(0)），因此在 if/else 里的悬垂 else 行为与原 if 一致。
// NN_TRY_MSG 仅多做一件事：把 ctx 前缀拼进 Error.message（ctx 为空则逐字等价）。
//
// 实现约束（改动前必读）：
//   · 终止策略只有 abort / exit —— 铁律 #1 禁异常，本头任何情况下不出现 throw；
//   · 变参分发走实参计数（NN_DETAIL_PICK2/3），不用 __VA_OPT__（MSVC 传统
//     预处理器不可靠），末尾哨兵保证 `...` 至少收到一个实参；
//   · ⚠ 表达式含**顶层逗号**（多参数模板列表、花括号初始化）会让实参计数出错
//     → 选中错误分支 → **编译期报错（响亮，不会静默错值）**，加一层括号即可；
//   · ⚠ 铁律 #10：宏续行反斜杠后面必须紧跟代码，绝不能出现 // 注释（宏体内
//     一律用 /* */，// 注释只放整个宏块上方）。
#define NN_TRY(decl, ...)      \
    auto decl = (__VA_ARGS__); \
    if (!decl) return std::unexpected(decl.error())

// ⚠ NN_TRY_CHECK **必须只求值一次** x：旧展开 `if (!(x)) return
//   std::unexpected((x).error())` 把 x 求值两遍，对**带副作用**的表达式是
//   正确性炸弹——x = seg.end() 这类"失败即已重置状态"的调用，第二遍会重新
//   进入并**返回成功**，随后对 success 的 std::expected 调 .error() = MSVC
//   `expected` 断言 / 读已析构的 Error 字符串 → 堆损坏（实测 f16_precision_test
//   0xC0000374）。改 if-init 绑定：单次求值、语义与文档承诺的手写形态逐字等价。
//   绑定用 auto&&（不引入拷贝：Result 载体可能是不可拷贝的 unique_ptr）。
#define NN_TRY_CHECK(x) \
    if (auto&& nn_try_r_ = (x); !nn_try_r_) return std::unexpected(nn_try_r_.error())

#define NN_TRY_MSG(decl, ctx, ...) \
    auto decl = (__VA_ARGS__);     \
    if (!decl) return std::unexpected(::nn::detail::contextualize(decl.error(), (ctx)))

#define NN_FAIL(msg) return std::unexpected(nn::Error{(msg)})

// ── 变参个数分发（1 个实参 → *_1；2 个实参 → *_MSG）─────────────────────
#define NN_DETAIL_PICK2(_1, _2, NAME, ...) NAME
#define NN_DETAIL_PICK3(_1, _2, _3, NAME, ...) NAME

#define NN_UNWRAP(...) \
    NN_DETAIL_PICK2(__VA_ARGS__, NN_UNWRAP_MSG, NN_UNWRAP_1, NN_PICK_END)(__VA_ARGS__)
#define NN_UNWRAP_1(x) ::nn::detail::check_value((x), #x, {}, std::source_location::current())
#define NN_UNWRAP_MSG(x, m) ::nn::detail::check_value((x), #x, (m), std::source_location::current())

// ── 向后兼容别名（等价展开，仅旧代码在用；新代码请用 NN_UNWRAP）─────────
#define NN_CHECK(...) NN_UNWRAP(__VA_ARGS__)
#define NN_EASY_CHECK(x) NN_UNWRAP_1(x)

#define NN_EXIT(...) NN_DETAIL_PICK3(__VA_ARGS__, NN_EXIT_MSG, NN_EXIT_2, NN_PICK_END)(__VA_ARGS__)
#define NN_EXIT_2(x, c) ::nn::detail::check_exit((x), #x, {}, (c), std::source_location::current())
#define NN_EXIT_MSG(x, c, m) \
    ::nn::detail::check_exit((x), #x, (m), (c), std::source_location::current())

// ── 调试断言宏（带位置信息）────────────────────────────────────────────
// **可选诊断**：用于内部不变量检查，生产环境（NDEBUG）为空展开。
// ⚠ 只在"失败后不再使用被校验的前提"时用它；否则请用 NN_ASSERT_ALWAYS。
#ifdef NDEBUG
#define NN_ASSERT(cond, msg) ((void)0)
#else
#define NN_ASSERT(cond, msg)                                                                    \
    do                                                                                          \
    {                                                                                           \
        if (!(cond))                                                                            \
        {                                                                                       \
            std::fprintf(stderr, "NN_ASSERT failed at %s:%d: %s\n", __FILE__, __LINE__, (msg)); \
            std::abort();                                                                       \
        }                                                                                       \
    } while (0)
#endif

// ── 前置条件校验宏（**Release 也终止**，RV-01）────────────────────────────
// 语义 = "断言后继续用该条件" 的正确形态：NDEBUG 下与 Debug 下**同形**
// （都打印 file:line 并终止），因此不存在"Release 静默 UB"的窗口。
// 判定口径：凡是断言点之后**仍会**解引用/索引被校验对象的，一律用这两个宏。
// 代价 = 一次整型比较，与 algebra_matrix.hpp::require_same_shape 的既有修法同价。
#define NN_ASSERT_ALWAYS(cond, msg)                                                         \
    do                                                                                      \
    {                                                                                       \
        if (!(cond)) ::nn::detail::fatal_at("NN_ASSERT_ALWAYS", __FILE__, __LINE__, (msg)); \
    } while (0)

// 无条件终止（无条件的编程错误：走到这里本身就是 bug）。
// msg 必须是**可打印的字符串**（char* / std::string::c_str()）；格式化场景
// 先 snprintf 到局部缓冲再传（避免把 std::string_view::data() 当 NUL 结尾串）。
#define NN_FATAL(msg) ::nn::detail::fatal_at("NN_FATAL", __FILE__, __LINE__, (msg))

// ── 运行时检查宏（始终启用）───────────────────────────────────────────
// 用于前置条件校验，返回 Error 而非 abort
#define NN_REQUIRE(cond, msg)                                      \
    do                                                             \
    {                                                              \
        if (!(cond)) { return std::unexpected(nn::Error{(msg)}); } \
    } while (0)

// ── 热点函数标记（提示编译器优化）─────────────────────────────────────
// 用于高频调用的前向/反向传播函数
#if defined(__GNUC__) || defined(__clang__)
#define NN_HOT_FN [[gnu::hot]] [[gnu::always_inline]]
#elif defined(_MSC_VER)
#define NN_HOT_FN __forceinline
#else
#define NN_HOT_FN inline
#endif

// ── 分支预测提示 ─────────────────────────────────────────────────────
// 用于罕见错误路径，帮助编译器优化指令流水线
#if defined(__GNUC__) || defined(__clang__)
#define NN_LIKELY(x) (__builtin_expect(!!(x), 1))
#define NN_UNLIKELY(x) (__builtin_expect(!!(x), 0))
#else
#define NN_LIKELY(x) (x)
#define NN_UNLIKELY(x) (x)
#endif

// ── 依赖引入（必须保持在**文件末尾**：宏在前、依赖在后）──────────────────
// 展开点需要的实现支撑：nn::Error / Result / detail::check_value /
// detail::check_exit / detail::contextualize / detail::fatal_at。
// include 放这里与文件头「环形包含的合法次序」说明绑定——挪到宏定义之前
// 会让「入口 = 本头」的路径在 core_errors::parse_number 处报 NN_FAIL 未定义。
#include "neuralnet.cpp/core_assert.hpp"
#include "neuralnet.cpp/core_errors.hpp"
