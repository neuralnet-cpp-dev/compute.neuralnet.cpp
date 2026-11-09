// ── threadpool_nested_test — 嵌套并行的调用者 slot 隔离回归（审查报告-2026-11-09 #C2）──
//
// 背景：调用者分片与 work-stealing 都借用"调用者 slot"。改动前该 slot 硬编码为
// workers_.size()，于是**任何两个同时活跃的并行区**（外层 caller + 内层 caller、
// 或两个外部线程各自发起并行区）都会把各自的调用者分片放到同一条槽上：
//   · 依赖 worker_index() 做 per-slot 状态的调用方会出现同槽并发写（数据竞争）；
//   · 语义上"调用者 slot 独占"的契约被打破。
// 本测试锁定两条可观察契约：
//   1) 嵌套（外层并行体内再起内层并行）时，内外层调用者分片不得同时占用同一 slot；
//   2) 两个外部线程各自发起并行区时，两个调用者分片不得同时占用同一 slot。
// 判定方式：分片回调里对 worker_index() 做 per-slot 并发计数，进入时若该槽已被
// 其它线程占用即记为一次冲突。探针不依赖修复后的具体 slot 编号。

#include <neuralnet.cpp/core_threadpool.hpp>

#include <atomic>
#include <barrier>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <future>
#include <thread>
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

// 看门狗：场景 20 s 未返回即判死锁并立即退出（ctest 不会卡死）。
template <typename Fn> void run_guarded(const char* what, Fn&& fn)
{
    auto fut = std::async(std::launch::async, std::forward<Fn>(fn));
    if (fut.wait_for(std::chrono::seconds(20)) != std::future_status::ready)
    {
        std::printf("[FAIL] %s: 20 s 内未返回 —— 死锁\n", what);
        std::fflush(stdout);
        std::_Exit(2);
    }
    fut.get();
}

constexpr std::size_t kProbeSlots = 128;

// per-slot 并发探针：同一 slot 被两个线程同时持有 → collisions 递增。
struct SlotProbe
{
    SlotProbe()
    {
        for (auto& c : concurrent) c.store(0, std::memory_order_relaxed);
    }

    void enter(std::size_t slot) noexcept
    {
        if (slot >= kProbeSlots)
        {
            overflow.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        if (concurrent[slot].fetch_add(1, std::memory_order_acq_rel) > 0)
            collisions.fetch_add(1, std::memory_order_relaxed);
    }

    void leave(std::size_t slot) noexcept
    {
        if (slot < kProbeSlots) concurrent[slot].fetch_sub(1, std::memory_order_acq_rel);
    }

    std::atomic<int> concurrent[kProbeSlots];
    std::atomic<std::size_t> collisions{0};
    std::atomic<std::size_t> overflow{0};
};

// ── 场景 1：嵌套并行（外层调用者分片占住 slot，worker 侧起内层并行区）────────
void test_nested_parallel_caller_slot()
{
    std::printf("嵌套并行：内外层调用者 slot 隔离\n");
    nn::ThreadPool pool(4);
    const std::size_t caller_slot = pool.size();  // 外层调用者分片的 slot（约定）

    constexpr std::size_t kOuter = 3 * 1024;  // chunk_count = min(4, 3) = 3
    constexpr std::size_t kInner = 2 * 1024;  // chunk_count = min(4, 2) = 2

    std::vector<int> outer(kOuter, 0);
    std::vector<int> inner(kInner, 0);
    SlotProbe probe;
    std::atomic<std::size_t> outer_visits{0};
    std::atomic<std::size_t> inner_visits{0};
    std::atomic<std::size_t> inner_regions{0};
    std::atomic<int> inner_caller_seen{0};

    pool.parallel_for_each_indexed(
        outer.begin(), outer.end(),
        [&](int& v, std::size_t slot)
        {
            probe.enter(slot);

            if (slot == caller_slot)
            {
                // 外层调用者分片：持有 caller slot，等待 worker 侧内层调用者分片进入。
                // 修复前内层调用者分片同样领取 caller_slot → 探针必报冲突。
                const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
                while (inner_caller_seen.load(std::memory_order_acquire) == 0 &&
                       std::chrono::steady_clock::now() < deadline)
                    std::this_thread::yield();
            }
            else
            {
                // worker 侧外层分片：就地发起内层并行区
                inner_regions.fetch_add(1, std::memory_order_relaxed);
                const auto outer_tid = std::this_thread::get_id();
                pool.parallel_for_each_indexed(
                    inner.begin(), inner.end(),
                    [&](int& iv, std::size_t islot)
                    {
                        probe.enter(islot);
                        // 内层并行区的"调用者上下文" = 发起内层区的那个线程
                        if (std::this_thread::get_id() == outer_tid)
                            inner_caller_seen.fetch_add(1, std::memory_order_release);
                        ++iv;
                        inner_visits.fetch_add(1, std::memory_order_relaxed);
                        probe.leave(islot);
                    });
            }

            ++v;
            outer_visits.fetch_add(1, std::memory_order_relaxed);
            probe.leave(slot);
        });

    CHECK(inner_caller_seen.load() > 0, "内层并行区确实在 worker 侧发起（场景生效）");
    CHECK(probe.overflow.load() == 0, "slot 索引未越界");
    if (probe.collisions.load() != 0)
        std::printf("       collisions=%zu outer_visits=%zu inner_visits=%zu regions=%zu\n",
                    probe.collisions.load(), outer_visits.load(), inner_visits.load(),
                    inner_regions.load());
    CHECK(probe.collisions.load() == 0, "嵌套并行：内外层调用者 slot 无并发共用");
    CHECK(outer_visits.load() == kOuter, "外层分片全覆盖（每元素一次）");
    CHECK(inner_regions.load() * kInner == inner_visits.load(), "内层分片全覆盖（每元素一次）");
}

// ── 场景 2：两个外部线程各自发起并行区，调用者分片同时活跃 ─────────────────
void test_concurrent_external_callers()
{
    std::printf("并发外部调用者：两个并行区调用者 slot 隔离\n");
    nn::ThreadPool pool(4);

    constexpr std::size_t kN = 3 * 1024;  // chunk_count = min(4, 3) = 3
    std::vector<int> a(kN, 0);
    std::vector<int> b(kN, 0);
    SlotProbe probe;
    std::barrier sync{2};

    auto run_one = [&](std::vector<int>& v)
    {
        const auto my_id = std::this_thread::get_id();
        pool.parallel_for_each_indexed(
            v.begin(), v.end(),
            [&](int& value, std::size_t slot)
            {
                probe.enter(slot);
                if (std::this_thread::get_id() == my_id)
                {
                    // 两个外部线程的调用者分片在此对齐：同时持有各自的 slot。
                    // 修复前两者都是 workers_.size() → 探针必报冲突。
                    sync.arrive_and_wait();
                }
                ++value;
                probe.leave(slot);
            });
    };

    std::thread t(run_one, std::ref(a));
    run_one(b);
    t.join();

    bool full = true;
    for (std::size_t i = 0; i < kN; ++i)
        if (a[i] != 1 || b[i] != 1) full = false;

    CHECK(probe.overflow.load() == 0, "slot 索引未越界");
    CHECK(probe.collisions.load() == 0, "并发外部调用者 slot 无并发共用");
    CHECK(full, "两个并行区均全覆盖");
}

}  // namespace

int main()
{
    run_guarded("nested_parallel_caller_slot", test_nested_parallel_caller_slot);
    run_guarded("concurrent_external_callers", test_concurrent_external_callers);

    if (g_fail == 0)
    {
        std::printf("[PASS] threadpool_nested_test\n");
        return 0;
    }
    std::printf("[FAIL] threadpool_nested_test: %d failure(s)\n", g_fail);
    return 1;
}
