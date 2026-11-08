// ── threadpool_exception_test — 任务体异常闸门回归（审查报告-2026-11-09 #C1）──
//
// 背景：本仓默认 -fno-exceptions（任务体不可能抛，bad_alloc 直接 terminate），
// 但 core_threadpool.hpp 是 header-only，消费方可能在 -fexceptions 下编译它。
// 此时若任务体的异常穿过分块，会有两处后果：
//   ① 异常逃出工作线程 → std::terminate（整个进程死）；
//   ② latch 少减一次 → 等待方永久挂住，且队列中残留的任务持有已失效的栈引用。
// 本测试用 -fexceptions 编译（见 engine/CMakeLists.txt 对该目标的反向覆盖），
// 锁定以下契约：
//   1) 任务体抛异常后主线程必须返回（latch 一定归零），不得死锁/超时；
//   2) 异常不静默：发起并行区的线程收到重抛，线程池计数递增；
//   3) 重抛发生在全部分块结束之后（不留悬垂引用）：其余分块都已完成；
//   4) 归约路径不合并半成品：有块失败时整体重抛，绝不返回脏结果；
//   5) 96B 行内任务对象在异常路径下仍正常析构（无泄漏）；
//   6) 串行回落路径的异常语义与普通函数调用一致。

#include <neuralnet.cpp/core_threadpool.hpp>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{
int g_fail = 0;

#define CHECK(cond, msg)                                                  \
    do                                                                    \
    {                                                                     \
        if (!(cond))                                                      \
        {                                                                 \
            std::printf("[FAIL] %s:%d: %s\n", __FILE__, __LINE__, (msg)); \
            ++g_fail;                                                     \
        }                                                                 \
    } while (0)

// 看门狗：场景 5 s 未返回即判死锁并立即退出（ctest 不会卡死）。
// 被判定死锁的线程仍在等待，不能 join，故用 _Exit 绕过静态析构。
template <typename Fn> void run_guarded(const char* what, Fn&& fn)
{
    auto fut = std::async(std::launch::async, std::forward<Fn>(fn));
    if (fut.wait_for(std::chrono::seconds(5)) != std::future_status::ready)
    {
        std::printf("[FAIL] %s: 5 s 内未返回 —— latch 未归零 / 死锁\n", what);
        std::fflush(stdout);
        std::_Exit(2);
    }
    fut.get();
}

constexpr std::size_t kN = 200000;  // 远超 MIN_CHUNK，保证 n_chunks > 1

// ① 全部分块抛异常：主线程返回、异常可见、计数递增
void test_all_chunks_throw()
{
    nn::ThreadPool pool(4);
    std::vector<int> v(kN, 0);
    const std::size_t before = pool.task_exception_count();

    bool saw = false;
    try
    {
        pool.parallel_for_each(v.begin(), v.end(),
                               [](int&) { throw std::runtime_error("chunk-boom"); });
    }
    catch (const std::runtime_error& e)
    {
        saw = (std::string(e.what()) == "chunk-boom");
    }
    catch (...)
    {
    }

    CHECK(saw, "调用线程收到任务体异常（未被静默吞掉）");
    CHECK(pool.task_exception_count() > before, "线程池记录了任务异常次数");

    // 异常已被消费，同一线程池应完全可用（无残留错误状态）
    std::vector<int> w(kN, 0);
    pool.parallel_for_each(w.begin(), w.end(), [](int& x) { ++x; });
    CHECK(w.front() == 1 && w.back() == 1, "异常后同一线程池仍可正常并行");
}

// ② 只有调用者末块抛异常：证明重抛发生在其余分块全部完成之后
void test_throw_in_caller_chunk_after_others_done()
{
    nn::ThreadPool pool(4);
    std::vector<int> v(kN, 0);
    std::atomic<std::size_t> visited{0};

    bool threw = false;
    try
    {
        pool.parallel_for_each(
            v.begin(), v.end(),
            [&](int& x)
            {
                const auto idx = static_cast<std::size_t>(&x - v.data());
                if (idx == kN - 1) throw std::runtime_error("tail-boom");
                visited.fetch_add(1, std::memory_order_relaxed);
            });
    }
    catch (const std::runtime_error&)
    {
        threw = true;
    }
    catch (...)
    {
    }

    CHECK(threw, "末块异常被重抛给调用线程");
    CHECK(visited.load() == kN - 1, "其余分块全部完成（先 latch 归零、后重抛 → 无悬垂引用）");
}

// ③ 归约路径：有块失败时不得合并半成品
void test_reduce_does_not_return_partial_result()
{
    nn::ThreadPool pool(4);
    std::vector<double> v(kN, 0.0);

    bool threw = false;
    try
    {
        (void)pool.parallel_transform_reduce(
            v.begin(), v.end(), 0.0, [](double a, double b) { return a + b; },
            [](double) -> double { throw std::runtime_error("reduce-boom"); });
    }
    catch (const std::runtime_error&)
    {
        threw = true;
    }
    catch (...)
    {
    }

    CHECK(threw, "归约分块失败时整体重抛，绝不返回未写完的部分和");
}

// ④ 96B 行内任务对象：异常后仍正常析构（行内存储不漏析构）
void test_inline_task_destructor_on_exception()
{
    struct Tracked
    {
        double pad[10];  // 80 字节，连同指针共 88 → 落在 PoolTask 96B 行内区
        int* live;
        explicit Tracked(int* p) noexcept : pad{}, live(p) { ++*live; }
        Tracked(const Tracked& o) noexcept : pad{}, live(o.live) { ++*live; }
        Tracked(Tracked&& o) noexcept : pad{}, live(o.live) { ++*live; }
        ~Tracked() { --*live; }
    };

    int live = 0;
    {
        auto body = [t = Tracked(&live)]()
        {
            (void)t;
            throw std::runtime_error("task-boom");
        };
        nn::detail::PoolTask task = std::move(body);
        CHECK(live >= 1, "捕获体已就地构造在任务对象内");

        bool caught = false;
        try
        {
            task();
        }
        catch (const std::runtime_error&)
        {
            caught = true;
        }
        CHECK(caught, "PoolTask 本身不吞异常（截断点在池的分块层）");
    }
    CHECK(live == 0, "行内任务对象在异常后正常析构（无泄漏）");
}

// ⑤ 串行回落路径：异常语义与普通函数调用一致（不经过线程池）
void test_serial_fallback_propagates()
{
    nn::ThreadPool pool(4);
    std::vector<int> small(10, 0);

    bool caught = false;
    try
    {
        pool.parallel_for_each(small.begin(), small.end(),
                               [](int&) { throw std::logic_error("serial-boom"); });
    }
    catch (const std::logic_error&)
    {
        caught = true;
    }
    catch (...)
    {
    }
    CHECK(caught, "串行回落路径的异常直接抛给调用方");
}

}  // namespace

int main()
{
    run_guarded("all_chunks_throw", test_all_chunks_throw);
    run_guarded("throw_in_caller_chunk_after_others_done", test_throw_in_caller_chunk_after_others_done);
    run_guarded("reduce_does_not_return_partial_result", test_reduce_does_not_return_partial_result);
    run_guarded("inline_task_destructor_on_exception", test_inline_task_destructor_on_exception);
    run_guarded("serial_fallback_propagates", test_serial_fallback_propagates);

    if (g_fail == 0) std::printf("[PASS] threadpool_exception_test\n");
    return g_fail == 0 ? 0 : 1;
}
