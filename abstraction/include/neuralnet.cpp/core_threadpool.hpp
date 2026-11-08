#pragma once

#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>
#include <atomic>
#include <exception>  // std::exception_ptr（任务异常记录，见 RegionState）
#include <cstddef>    // for std::max_align_t
#include <new>        // for placement new（任务就地构造）
#include <utility>    // for std::move / std::forward
#include <algorithm>  // for std::transform (serial fallback)
#include <numeric>    // for std::transform_reduce (serial fallback)
#include <iterator>   // for std::distance
#include <type_traits>

#include <neuralnet.cpp/core_assert.hpp>   // base 仓

// ── CPU pause 自旋原语：GCC/Clang 用 __builtin_ia32_pause，MSVC 用 _mm_pause ──
#if defined(_MSC_VER)
#include <immintrin.h>
#define NN_CPU_PAUSE() _mm_pause()
#else
#define NN_CPU_PAUSE() __builtin_ia32_pause()
#endif

namespace nn
{
namespace detail
{
// ── 轻量任务对象：类型擦除可调用体，捕获体就地存放（#P12）──────────────
// 动机：std::function 的小对象缓冲约 16 字节，而分块 lambda 的捕获
// （this + 迭代器 + &func + &latch …）普遍 40–64 字节 → **每个分块**
// 一次堆分配。这里把捕获体直接放进任务对象（96 字节行内存储），
// 只有超出行内容量/对齐的罕见捕获才退化为一次堆分配。
// 语义与 std::function<void()> 等价：拷贝语义不开放（队列只用移动）。
class PoolTask
{
public:
    static constexpr std::size_t kInlineCapacity = 96;

    PoolTask() noexcept = default;
    ~PoolTask() { reset(); }

    PoolTask(const PoolTask&) = delete;
    PoolTask& operator=(const PoolTask&) = delete;

    PoolTask(PoolTask&& other) noexcept { adopt(other); }
    PoolTask& operator=(PoolTask&& other) noexcept
    {
        if (this != &other)
        {
            reset();
            adopt(other);
        }
        return *this;
    }

    // 非 explicit：供 vector::emplace_back(lambda) 就地构造（零移动）
    template <typename F,
              typename = std::enable_if_t<!std::is_same_v<std::remove_cvref_t<F>, PoolTask>>>
    PoolTask(F&& f)
    {
        using Fn = std::remove_cvref_t<F>;
        static_assert(std::is_invocable_v<Fn&>, "ThreadPool 任务必须是可无参调用的对象");
        invoke_ = &invoke_impl<Fn>;
        if constexpr (sizeof(Fn) <= kInlineCapacity && alignof(Fn) <= alignof(std::max_align_t))
        {
            ::new (static_cast<void*>(storage_)) Fn(std::forward<F>(f));
            ctx_ = storage_;
            destroy_ = &destroy_impl<Fn>;
            relocate_ = &relocate_impl<Fn>;
        }
        else
        {
            ctx_ = static_cast<void*>(new Fn(std::forward<F>(f)));
            destroy_ = &delete_impl<Fn>;  // 堆回退：析构即 delete
        }
    }

    void operator()() { invoke_(ctx_); }
    explicit operator bool() const noexcept { return invoke_ != nullptr; }

private:
    template <typename Fn> static void invoke_impl(void* p) { (*static_cast<Fn*>(p))(); }
    template <typename Fn> static void destroy_impl(void* p) { static_cast<Fn*>(p)->~Fn(); }
    template <typename Fn> static void delete_impl(void* p) { delete static_cast<Fn*>(p); }
    template <typename Fn> static void relocate_impl(void* dst, void* src)
    {
        Fn* from = static_cast<Fn*>(src);
        ::new (dst) Fn(std::move(*from));
        from->~Fn();
    }

    void reset() noexcept
    {
        if (destroy_ != nullptr) destroy_(ctx_);
        invoke_ = nullptr;
        destroy_ = nullptr;
        relocate_ = nullptr;
        ctx_ = nullptr;
    }

    void adopt(PoolTask& other) noexcept
    {
        invoke_ = other.invoke_;
        destroy_ = other.destroy_;
        relocate_ = other.relocate_;
        if (other.ctx_ == other.storage_)  // 行内捕获体：搬移构造后析构源
        {
            relocate_(storage_, other.storage_);
            ctx_ = storage_;
        }
        else
        {
            ctx_ = other.ctx_;  // 堆捕获体：转移所有权
        }
        other.invoke_ = nullptr;
        other.destroy_ = nullptr;
        other.relocate_ = nullptr;
        other.ctx_ = nullptr;
    }

    alignas(std::max_align_t) unsigned char storage_[kInlineCapacity];
    void* ctx_ = nullptr;
    void (*invoke_)(void*) = nullptr;
    void (*destroy_)(void*) = nullptr;
    void (*relocate_)(void*, void*) = nullptr;  // 仅行内路径需要
};

// ── 并行区共享状态：分块完成计数 + 首个异常记录（#C1）──────────────────
// 放在发起并行区的调用方栈上：队列任务只引用它；只要 latch 先归零再离开
// 并行函数，引用就不会悬垂（异常路径同样由 finish_chunk 保证归零）。
struct RegionState
{
    explicit RegionState(int n_chunks) noexcept : latch(n_chunks) {}

    std::atomic<int> latch;
    std::atomic<bool> failed{false};
    std::exception_ptr error;  // 仅首个异常；异常构建下才会被写入
};

// ── 异常闸门：调用可调用体，异常一律吞下并转交 on_error（绝不外泄）──────
// -fno-exceptions 构建（全仓默认）下任务体不可能抛（bad_alloc 直接
// terminate），该分支编译为空壳，行为与改动前逐位一致；-fexceptions 构建
// 下任务体的异常若穿过工作线程会触发 std::terminate，因此必须在此截断。
template <typename Fn, typename OnError>
inline void invoke_nothrow(Fn&& fn, OnError&& on_error) noexcept
{
#if defined(__cpp_exceptions)
    try
    {
        fn();
    }
    catch (...)
    {
        on_error(std::current_exception());
    }
#else
    fn();
    (void)on_error;
#endif
}
}  // namespace detail

// ── 简易线程池（latch 零分配设计）──────────────────────────────────────
// 提交模型：1 次批量入队 + 1 个原子计数器 + 调用者参与处理，
//   每分块零次 shared_ptr<packaged_task> 堆分配、零次 future 同步开销。
//   任务用 detail::PoolTask（捕获体就地存放）+ 环形队列（见下方队列注释），
//   每分块入队/出队不产生堆分配。
class ThreadPool
{
private:
    std::vector<std::thread> workers_;

    // ── 任务队列：环形数组（摊还分配，稳态零分配）──────────────────
    // 不用 std::queue：MSVC STL 的 deque 对 >8 字节元素是 1 元素/块——
    // 每次 push 仍是一块堆分配。这里用容量按 2 倍增长的 vector + 队首
    // 下标（消费过半即就地压缩），扩容摊还 O(1)。
    std::vector<detail::PoolTask> tasks_;
    std::size_t queue_head_ = 0;  // 队首下标（其前的槽均已消费）

    // ── 队列基本操作（调用方必须已持 queue_mutex_）──────────────────
    [[nodiscard]] bool queue_empty_() const noexcept
    { return queue_head_ >= tasks_.size(); }

    // 出队一个任务。前提：持锁且 !queue_empty_()。
    // 返回值持有捕获体（行内或堆），执行完由 PoolTask 析构回收。
    [[nodiscard]] detail::PoolTask pop_task_()
    {
        detail::PoolTask task = std::move(tasks_[queue_head_]);
        ++queue_head_;
        if (queue_head_ == tasks_.size())
        {
            tasks_.clear();
            queue_head_ = 0;
        }
        else if (queue_head_ >= 64 && queue_head_ * 2 >= tasks_.size())
        {
            // 就地压缩：存活任务搬到前部（move 赋值，堆捕获体也随之转移）
            const std::size_t live = tasks_.size() - queue_head_;
            std::move(tasks_.begin() + static_cast<std::ptrdiff_t>(queue_head_), tasks_.end(),
                      tasks_.begin());
            tasks_.resize(live);
            queue_head_ = 0;
        }
        return task;
    }

    std::mutex queue_mutex_;
    std::condition_variable condition_;
    std::atomic<bool> stop_{false};

    // 累计被吞掉的任务体异常数（诊断用；异常本身会在发起并行区的调用线程
    // 重抛，见 rethrow_if_failed —— 这里只是"至少记录一次"的持久观测点）
    std::atomic<std::size_t> task_exceptions_{0};

public:
    // ── worker 线程索引（每个 worker 在构造时绑定唯一 ID） ──────
    // 用于 parallel_for_each 内部识别当前执行线程，支持 per-worker 状态
    // （如 BPE 训练的并行合并 delta 映射）。
    static thread_local std::size_t tl_worker_index;

    explicit ThreadPool(std::size_t num_threads = std::thread::hardware_concurrency())
    {
        if (num_threads == 0) num_threads = 1;
        workers_.reserve(num_threads);
        for (std::size_t i = 0; i < num_threads; ++i)
        {
            workers_.emplace_back(
                [this, i]
                {
                    tl_worker_index = i;
                    for (;;)
                    {
                        detail::PoolTask task;
                        {
                            std::unique_lock lock(queue_mutex_);
                            condition_.wait(lock,
                                            [this]
                                            {
                                                return stop_.load(std::memory_order_acquire) ||
                                                       !queue_empty_();
                                            });
                            if (stop_.load(std::memory_order_acquire) && queue_empty_()) return;
                            task = pop_task_();
                        }
                        task();
                    }
                });
        }
    }

    // ── 提交模型只有分块原语：本池是"零分配 latch"设计，不提供单任务
    //    submit（其每次调用需 make_shared<packaged_task> 堆分配）。
    //    需要 future 语义时应在调用方分块后用 parallel_* 系列原语。 ───

private:
    // ── 分块辅助：计算合理的分块数 ──────────────────────────────────
    // 每分块至少 1024 个元素（Scalar=float 时约 4 KB），可装入 L1 缓存。
    // 下限取 1024（而非更大值）：使 MNIST 小隐藏层（64×batch）也能
    // 分出多个分块、触发多核并行。
    [[nodiscard]] std::size_t chunk_count(std::size_t total) const noexcept
    {
        constexpr std::size_t MIN_CHUNK = 1024;
        const auto nw = workers_.size();
        if (nw <= 1 || total < MIN_CHUNK * 2) return 1;
        auto n = std::min(nw, total / MIN_CHUNK);
        return n < 1 ? 1 : n;
    }

    // ── 分片完成：递减 latch；最后一个完成者唤醒 cv 等待者 ──────────
    // 等待方（wait_for_latch 的无超时 wait / worker 主循环）都挂在
    // condition_ 上，latch 归零必须通知，否则等待者会永久睡眠。
    // 用 notify_all：等待者可能同时包含调用者与空闲 worker。
    //
    // ⚠ 必须持 queue_mutex_ 再 notify（否则丢唤醒 / lost wakeup）：
    // wait_for_latch 用 wait(lock, [&]{ return latch == 0; }) 等待，谓词
    // 在同一把锁下求值。若归零+notify 不持锁，就会出现
    //   等待者判谓词=false → 通知者置零并 notify（此刻无注册等待者，信号丢失）
    //   → 等待者真正进入 wait() → 永久阻塞
    // 空并行区压测（32 核）实测：不持锁
    // 数万次 region 内必死锁；持锁后 3×50000 次稳定通过。
    void finish_chunk(std::atomic<int>& latch) noexcept
    {
        if (latch.fetch_sub(1, std::memory_order_release) == 1)
        {
            std::lock_guard lock(queue_mutex_);
            condition_.notify_all();
        }
    }

    // ── 分块异常闸门（#C1）──────────────────────────────────────────
    // 每个分块——工作线程上的队列任务与调用者自己那段——都必须经由此处：
    //   ① latch 递减走 RAII：任务体抛异常时同样归零，等待方不会永久挂住；
    //   ② 异常在此截断并记录首个：绝不穿过工作线程（否则 std::terminate）；
    //   ③ 记录后继续执行其它分块，区域结束时由调用线程统一重抛。
    template <typename Body> void run_chunk(detail::RegionState& st, Body&& body) noexcept
    {
        struct LatchGuard
        {
            ThreadPool* pool;
            std::atomic<int>* latch;
            ~LatchGuard() { pool->finish_chunk(*latch); }
        } guard{this, &st.latch};

        detail::invoke_nothrow(std::forward<Body>(body),
                               [this, &st](std::exception_ptr e) noexcept
                               { record_chunk_error(st, std::move(e)); });
    }

    // 记录分块异常：只在失败路径调用（成功路径零原子写）。首个异常胜出，
    // 其余只计数；错误槽由区域状态持有，随并行函数返回而销毁。
    void record_chunk_error(detail::RegionState& st, std::exception_ptr e) noexcept
    {
        task_exceptions_.fetch_add(1, std::memory_order_relaxed);
        std::lock_guard lock(queue_mutex_);
        if (!st.failed.load(std::memory_order_relaxed))
        {
            st.failed.store(true, std::memory_order_relaxed);
            st.error = std::move(e);
        }
    }

    // ── 区域收尾：把任务层记录的异常在调用线程重抛（#C1）──────────────
    // 必须在 wait_for_latch 之后调用：此时所有分块均已结束，队列中不再有
    // 指向本栈帧（func / 区间 / 状态）的任务，展开不会留下悬垂引用。
    // 传递方式 = 标记为失败而非静默继续：与串行路径语义一致（串行时
    // 任务体的异常直接抛给调用方），调用方无需关心本次是否走了并行。
    void rethrow_if_failed(detail::RegionState& st)
    {
        if (!st.failed.load(std::memory_order_acquire)) return;
#if defined(__cpp_exceptions)
        std::exception_ptr e;
        {
            std::lock_guard lock(queue_mutex_);
            e = std::move(st.error);
            st.error = nullptr;
        }
        if (e) std::rethrow_exception(e);
#endif
    }

    // ── work-stealing 等待：调用者不空转，帮忙处理队列任务 ─────────
    // 等待分三阶段（依据实测：长时间自旋反复原子读取，CPU 占用率显示
    // 100% 但实际有效计算比例低，纯自旋/空转不可取）：
    //   1) 短自旋（16 次 pause）快速检测 latch 归零——典型情况无 yield 开销
    //   2) 自旋失败后 work-steal：尝试从队列取任务执行（参与计算）
    //   3) 队列为空时阻塞等待 condition_variable（无超时轮询），
    //      由 finish_chunk 在 latch 归零时通知唤醒
    // 线程索引约定：调用者线程帮忙执行偷来的任务时，临时将
    // tl_worker_index 置为 workers_.size()（调用者 slot），避免
    // parallel_for_each_indexed 的任务体与 worker 0 撞同一 slot；
    // 执行完恢复原值（嵌套并行时 worker 保留自己的索引）。
    void wait_for_latch(std::atomic<int>& latch)
    {
        // 阶段 1：短自旋（16 次 pause）——应对 latch 即将归零的快路径
        for (int spin = 0; spin < 16; ++spin)
        {
            if (latch.load(std::memory_order_acquire) == 0) return;
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
            NN_CPU_PAUSE();
#endif
        }

        // 阶段 2：work-stealing + cv 等待
        while (latch.load(std::memory_order_acquire) > 0)
        {
            detail::PoolTask task;
            {
                std::unique_lock lock(queue_mutex_);
                if (!queue_empty_()) task = pop_task_();
            }
            if (task)
            {
                // 借用"调用者 slot 索引"执行，执行完恢复原值
                const std::size_t saved = tl_worker_index;
                tl_worker_index = workers_.size();
                task();
                tl_worker_index = saved;

                // 执行完一个任务后回到阶段 1 短自旋
                for (int spin = 0; spin < 16; ++spin)
                {
                    if (latch.load(std::memory_order_acquire) == 0) return;
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
                    NN_CPU_PAUSE();
#endif
                }
            }
            else
            {
                // 队列为空：阻塞等待（无超时）；
                // 完成通知来自 finish_chunk，虚假唤醒由谓词兜底
                std::unique_lock lock(queue_mutex_);
                condition_.wait(lock,
                                [&latch]() { return latch.load(std::memory_order_acquire) == 0; });
            }
        }
    }

public:
    // ── 并行 for_each（latch + 调用者参与） ─────────────────────────
    // 特点：零 future 分配、一次加锁入队、调用者不空等
    template <typename Iterator, typename Func>
    void parallel_for_each(Iterator first, Iterator last, Func&& func)
    {
        const auto total = static_cast<std::size_t>(std::ranges::distance(first, last));
        if (total == 0) return;

        const auto n_chunks = chunk_count(total);
        if (n_chunks <= 1)
        {
            for (auto it = first; it != last; ++it)
                func(*it);
            return;
        }

        const std::size_t base = total / n_chunks;
        const std::size_t rem = total % n_chunks;

        // 原子计数器 + 首个异常记录（异常路径见 run_chunk）
        detail::RegionState st{static_cast<int>(n_chunks)};

        // 将前 n_chunks-1 个分块批量入队（仅一次加锁）；
        // 每入队一个任务 notify_one，避免高核机器上唤醒全部 worker
        {
            std::lock_guard lock(queue_mutex_);
            // 预留容量：入队中途扩容失败会留下"latch 永远等不到"的残局
            tasks_.reserve(tasks_.size() + (n_chunks - 1));
            std::size_t off = 0;
            for (std::size_t c = 0; c < n_chunks - 1; ++c)
            {
                const std::size_t len = base + (c < rem ? 1 : 0);
                auto beg = first;
                std::ranges::advance(beg, static_cast<std::ptrdiff_t>(off));
                auto end = beg;
                std::ranges::advance(end, static_cast<std::ptrdiff_t>(len));
                off += len;

                tasks_.emplace_back(
                    [this, beg, end, &func, &st]()
                    {
                        run_chunk(st,
                                  [&]
                                  {
                                      for (auto it = beg; it != end; ++it)
                                          func(*it);
                                  });
                    });
                condition_.notify_one();
            }
        }

        // 调用者处理最后一个分块（不经过队列，零分配）
        {
            const std::size_t c = n_chunks - 1;
            const std::size_t len = base + (c < rem ? 1 : 0);
            const std::size_t off = total - len;
            auto beg = first;
            std::ranges::advance(beg, static_cast<std::ptrdiff_t>(off));
            run_chunk(st,
                      [&]
                      {
                          for (auto it = beg; it != last; ++it)
                              func(*it);
                      });
        }

        // work-stealing 等待：调用者帮忙处理队列任务而非空转
        wait_for_latch(st.latch);
        rethrow_if_failed(st);
    }

    // ── 带 worker 索引的并行 for_each ────────────────────────────
    // 与 parallel_for_each 相同的分区和调度策略，但回调额外接收
    // worker_index 参数（worker 线程 = 0..size()-1，调用者线程 = size()）。
    // 用于需要 per-worker 状态（如 BPE 训练的 local delta）的场景，
    // 调用者可安全使用 index = n_threads 作为独立 slot，避免与 worker 0 冲突。
    template <typename Iterator, typename Func>
    void parallel_for_each_indexed(Iterator first, Iterator last, Func&& func)
    {
        const auto total = static_cast<std::size_t>(std::ranges::distance(first, last));
        if (total == 0) return;

        const auto n_chunks = chunk_count(total);
        if (n_chunks <= 1)
        {
            for (auto it = first; it != last; ++it)
                func(*it, 0);  // 单线程：index = 0
            return;
        }

        const std::size_t base = total / n_chunks;
        const std::size_t rem = total % n_chunks;

        detail::RegionState st{static_cast<int>(n_chunks)};

        {
            std::lock_guard lock(queue_mutex_);
            tasks_.reserve(tasks_.size() + (n_chunks - 1));
            std::size_t off = 0;
            for (std::size_t c = 0; c < n_chunks - 1; ++c)
            {
                const std::size_t len = base + (c < rem ? 1 : 0);
                auto beg = first;
                std::ranges::advance(beg, static_cast<std::ptrdiff_t>(off));
                auto end = beg;
                std::ranges::advance(end, static_cast<std::ptrdiff_t>(len));
                off += len;

                tasks_.emplace_back(
                    [this, beg, end, &func, &st]()
                    {
                        run_chunk(st,
                                  [&]
                                  {
                                      for (auto it = beg; it != end; ++it)
                                          func(*it, tl_worker_index);
                                  });
                    });
                condition_.notify_one();
            }
        }

        // 调用者处理最后一个分块，index = workers_.size()（不与任何 worker 冲突）
        {
            const std::size_t c = n_chunks - 1;
            const std::size_t len = base + (c < rem ? 1 : 0);
            const std::size_t off = total - len;
            auto beg = first;
            std::ranges::advance(beg, static_cast<std::ptrdiff_t>(off));
            run_chunk(st,
                      [&]
                      {
                          for (auto it = beg; it != last; ++it)
                              func(*it, workers_.size());
                      });
        }

        wait_for_latch(st.latch);
        rethrow_if_failed(st);
    }

    // ── 块级并行 for_each（供矩阵乘法/转置的分块循环使用）─────────
    // 与普通 parallel_for_each 不同：每个"元素"本身就是一次重量级计算
    // （如 64×64×K 的矩阵乘法分块），因此不适用 MIN_CHUNK 保护。
    // 直接按块数分给各线程，1 块 = 1 分片。
    template <typename Iterator, typename Func>
    void parallel_for_blocks(Iterator first, Iterator last, Func&& func)
    {
        const auto total = static_cast<std::size_t>(std::ranges::distance(first, last));
        if (total <= 1)
        {
            for (auto it = first; it != last; ++it)
                func(*it);
            return;
        }

        const auto n_chunks = std::min(workers_.size(), total);
        const std::size_t base = total / n_chunks;
        const std::size_t rem = total % n_chunks;
        detail::RegionState st{static_cast<int>(n_chunks)};

        {
            std::lock_guard lock(queue_mutex_);
            tasks_.reserve(tasks_.size() + (n_chunks - 1));
            std::size_t off = 0;
            for (std::size_t c = 0; c < n_chunks - 1; ++c)
            {
                const std::size_t len = base + (c < rem ? 1 : 0);
                auto beg = first;
                std::ranges::advance(beg, static_cast<std::ptrdiff_t>(off));
                auto end = beg;
                std::ranges::advance(end, static_cast<std::ptrdiff_t>(len));
                off += len;

                tasks_.emplace_back(
                    [this, beg, end, &func, &st]()
                    {
                        run_chunk(st,
                                  [&]
                                  {
                                      for (auto it = beg; it != end; ++it)
                                          func(*it);
                                  });
                    });
                condition_.notify_one();
            }
        }

        {
            const std::size_t c = n_chunks - 1;
            const std::size_t off = total - (base + (c < rem ? 1 : 0));
            auto beg = first;
            std::ranges::advance(beg, static_cast<std::ptrdiff_t>(off));
            run_chunk(st,
                      [&]
                      {
                          for (auto it = beg; it != last; ++it)
                              func(*it);
                      });
        }

        wait_for_latch(st.latch);
        rethrow_if_failed(st);
    }

    // ── 并行 for_samples（独立样本级并行）───────────────────────────
    // 与 for_each（按元素分块）不同：每个"样本"包含一个完整的计算子任务
    // （如矩阵乘法、前向传播），样本间完全独立、无数据竞争。
    // 用于 GPT/Transformer batch 维度的并行：1 样本 = 1 分片。
    template <typename Func> void parallel_for_samples(std::size_t num_samples, Func&& func)
    {
        if (num_samples <= 1)
        {
            for (std::size_t i = 0; i < num_samples; ++i)
                func(i);
            return;
        }

        const auto n_chunks = std::min(workers_.size(), num_samples);
        const std::size_t base = num_samples / n_chunks;
        const std::size_t rem = num_samples % n_chunks;
        detail::RegionState st{static_cast<int>(n_chunks)};

        {
            std::lock_guard lock(queue_mutex_);
            tasks_.reserve(tasks_.size() + (n_chunks - 1));
            std::size_t off = 0;
            for (std::size_t c = 0; c < n_chunks - 1; ++c)
            {
                const std::size_t len = base + (c < rem ? 1 : 0);
                const std::size_t start = off;
                const std::size_t end = off + len;
                off += len;

                tasks_.emplace_back(
                    [this, start, end, &func, &st]()
                    {
                        run_chunk(st,
                                  [&]
                                  {
                                      for (std::size_t i = start; i < end; ++i)
                                          func(i);
                                  });
                    });
                condition_.notify_one();
            }
        }

        // 调用者处理最后一个分片
        {
            const std::size_t c = n_chunks - 1;
            const std::size_t start = num_samples - (base + (c < rem ? 1 : 0));
            run_chunk(st,
                      [&]
                      {
                          for (std::size_t i = start; i < num_samples; ++i)
                              func(i);
                      });
        }

        wait_for_latch(st.latch);
        rethrow_if_failed(st);
    }

    // ── 并行 transform（一元）───────────────────────────────────────
    template <typename InputIt, typename OutputIt, typename UnaryOp>
    void parallel_transform(InputIt first, InputIt last, OutputIt d_first, UnaryOp&& op)
    {
        const auto total = static_cast<std::size_t>(std::ranges::distance(first, last));
        if (total == 0) return;

        const auto n_chunks = chunk_count(total);
        if (n_chunks <= 1)
        {
            std::transform(first, last, d_first, std::forward<UnaryOp>(op));
            return;
        }

        const std::size_t base = total / n_chunks;
        const std::size_t rem = total % n_chunks;
        detail::RegionState st{static_cast<int>(n_chunks)};

        {
            std::lock_guard lock(queue_mutex_);
            tasks_.reserve(tasks_.size() + (n_chunks - 1));
            std::size_t off = 0;
            for (std::size_t c = 0; c < n_chunks - 1; ++c)
            {
                const std::size_t len = base + (c < rem ? 1 : 0);
                auto in_beg = first;
                auto out_beg = d_first;
                std::ranges::advance(in_beg, static_cast<std::ptrdiff_t>(off));
                std::ranges::advance(out_beg, static_cast<std::ptrdiff_t>(off));
                auto in_end = in_beg;
                std::ranges::advance(in_end, static_cast<std::ptrdiff_t>(len));
                off += len;

                tasks_.emplace_back(
                    [this, in_beg, in_end, out_beg, &op, &st]()
                    {
                        run_chunk(st,
                                  [&]
                                  {
                                      auto in = in_beg;
                                      auto out = out_beg;
                                      for (; in != in_end; ++in, ++out)
                                          *out = op(*in);
                                  });
                    });
                condition_.notify_one();
            }
        }

        {
            const std::size_t c = n_chunks - 1;
            const std::size_t off = total - (base + (c < rem ? 1 : 0));
            auto in_beg = first;
            auto out_beg = d_first;
            std::ranges::advance(in_beg, static_cast<std::ptrdiff_t>(off));
            std::ranges::advance(out_beg, static_cast<std::ptrdiff_t>(off));
            run_chunk(st,
                      [&]
                      {
                          for (; in_beg != last; ++in_beg, ++out_beg)
                              *out_beg = op(*in_beg);
                      });
        }

        wait_for_latch(st.latch);
        rethrow_if_failed(st);
    }

    // ── 并行 transform（二元）───────────────────────────────────────
    template <typename InputIt1, typename InputIt2, typename OutputIt, typename BinaryOp>
    void parallel_transform(InputIt1 first1, InputIt1 last1, InputIt2 first2, OutputIt d_first,
                            BinaryOp&& op)
    {
        const auto total = static_cast<std::size_t>(std::ranges::distance(first1, last1));
        if (total == 0) return;

        const auto n_chunks = chunk_count(total);
        if (n_chunks <= 1)
        {
            std::transform(first1, last1, first2, d_first, std::forward<BinaryOp>(op));
            return;
        }

        const std::size_t base = total / n_chunks;
        const std::size_t rem = total % n_chunks;
        detail::RegionState st{static_cast<int>(n_chunks)};

        {
            std::lock_guard lock(queue_mutex_);
            tasks_.reserve(tasks_.size() + (n_chunks - 1));
            std::size_t off = 0;
            for (std::size_t c = 0; c < n_chunks - 1; ++c)
            {
                const std::size_t len = base + (c < rem ? 1 : 0);
                auto i1 = first1;
                auto i2 = first2;
                auto o = d_first;
                std::ranges::advance(i1, static_cast<std::ptrdiff_t>(off));
                std::ranges::advance(i2, static_cast<std::ptrdiff_t>(off));
                std::ranges::advance(o, static_cast<std::ptrdiff_t>(off));
                auto i1_end = i1;
                std::ranges::advance(i1_end, static_cast<std::ptrdiff_t>(len));
                off += len;

                tasks_.emplace_back(
                    [this, i1, i1_end, i2, o, &op, &st]()
                    {
                        run_chunk(st,
                                  [&]
                                  {
                                      auto it1 = i1;
                                      auto it2 = i2;
                                      auto out = o;
                                      for (; it1 != i1_end; ++it1, ++it2, ++out)
                                          *out = op(*it1, *it2);
                                  });
                    });
                condition_.notify_one();
            }
        }

        {
            const std::size_t c = n_chunks - 1;
            const std::size_t off = total - (base + (c < rem ? 1 : 0));
            auto i1 = first1;
            auto i2 = first2;
            auto o = d_first;
            std::ranges::advance(i1, static_cast<std::ptrdiff_t>(off));
            std::ranges::advance(i2, static_cast<std::ptrdiff_t>(off));
            std::ranges::advance(o, static_cast<std::ptrdiff_t>(off));
            run_chunk(st,
                      [&]
                      {
                          for (; i1 != last1; ++i1, ++i2, ++o)
                              *o = op(*i1, *i2);
                      });
        }

        wait_for_latch(st.latch);
        rethrow_if_failed(st);
    }

    // ── 归约分块（确定性契约，铁律 8）──────────────────────────────────
    // 分段边界必须**只由 total 决定**，与 worker 数 / 机器核数无关：
    // 浮点加法非结合律使求和顺序直接决定字节结果，若分段依赖
    // workers_.size()，同一输入在 1-worker 与 N-worker 下会走不同折叠
    // 结构 → 字节不一致（跨机也不一致）。因此 1-worker 与 N-worker、
    // 任何机器都走完全相同的分段，部分和按**固定块下标**存放与合并，
    // 线程只决定"谁算哪块"，不参与边界与合并顺序。
    //
    // 借鉴 ATen::parallel_reduce（torch/include/ATen/Parallel-inl.h）：
    // 其部分和按 results[tid] 存放、边界由 get_num_threads() 决定 →
    // 跨线程数不可复现（PyTorch 亦声明 CPU 归约不保证跨线程数一致）；
    // 本库铁律 8 要求逐字节一致，故与 ATen 反向：边界固定、不用
    // per-thread 部分和下标。
    //
    // 分段语义（串/并行共用的唯一定义）：
    //   块 0 以 init 为种子，块 c>0 以块内首元素为种子（init 恰好计入
    //   一次——若每块各加 init、合并时再加一次，init≠0 时结果错误）；
    //   块内从左到右折叠；结果按块下标升序合并。
    //   n_chunks==1 时退化为纯左折叠，与 n < PARALLEL_THRESHOLD 的
    //   串行路径（core_config::transform_reduce）逐字节一致。
    // 块大小下限保证 n_chunks≥2 时每块至少 MIN_CHUNK 个元素（非空，
    // c>0 块取首元素作种子安全）；MAX_CHUNKS 限制任务/partials 数量，
    // 任务开销 ∝ 块数，实际调用方（text_train 梯度统计，log 间隔一次）
    // 对任务数不敏感。
    [[nodiscard]] static std::size_t reduce_chunk_count(std::size_t total) noexcept
    {
        constexpr std::size_t MIN_CHUNK = 16384;   // 元素数（64KB f32）
        constexpr std::size_t MAX_CHUNKS = 512;
        if (total < MIN_CHUNK * 2) return 1;
        return std::min(MAX_CHUNKS, total / MIN_CHUNK);
    }

    // ── 并行 transform_reduce（一元 transform）──────────────────────
    template <typename InputIt, typename T, typename BinaryOp, typename UnaryOp>
    T parallel_transform_reduce(InputIt first, InputIt last, T init, BinaryOp&& reduce_op,
                                UnaryOp&& transform_op)
    {
        const auto total = static_cast<std::size_t>(std::ranges::distance(first, last));
        if (total == 0) return init;

        const auto n_chunks = reduce_chunk_count(total);
        if (n_chunks == 1)
        {
            // 纯左折叠（= 单块语义）。显式循环而非 std::transform_reduce：
            // 后者归约顺序标准未定义，不能作确定性基准。
            T local = init;
            for (auto it = first; it != last; ++it)
                local = reduce_op(local, transform_op(*it));
            return local;
        }

        const std::size_t base = total / n_chunks;
        const std::size_t rem = total % n_chunks;
        detail::RegionState st{static_cast<int>(n_chunks)};

        // 部分和按固定块下标存放（与调度无关 → 确定性）
        std::vector<T> partials(n_chunks);

        {
            std::lock_guard lock(queue_mutex_);
            tasks_.reserve(tasks_.size() + (n_chunks - 1));
            std::size_t off = 0;
            for (std::size_t c = 0; c < n_chunks - 1; ++c)
            {
                const std::size_t len = base + (c < rem ? 1 : 0);
                auto beg = first;
                std::ranges::advance(beg, static_cast<std::ptrdiff_t>(off));
                auto end = beg;
                std::ranges::advance(end, static_cast<std::ptrdiff_t>(len));
                off += len;

                if (c == 0)
                {
                    // 块 0：以 init 为种子（init 在整个归约恰好计入一次）
                    tasks_.emplace_back(
                        [this, beg, end, &reduce_op, &transform_op, &partials, &st, init]()
                        {
                            run_chunk(st,
                                      [&]
                                      {
                                          T local = init;
                                          for (auto it = beg; it != end; ++it)
                                              local = reduce_op(local, transform_op(*it));
                                          partials[0] = std::move(local);
                                      });
                        });
                }
                else
                {
                    // 块 c>0：以块内首元素为种子（len ≥ MIN_CHUNK ≥ 1 保证非空）
                    tasks_.emplace_back(
                        [this, beg, end, &reduce_op, &transform_op, &partials, &st, c]()
                        {
                            run_chunk(st,
                                      [&]
                                      {
                                          auto it = beg;
                                          T local = transform_op(*it);
                                          for (++it; it != end; ++it)
                                              local = reduce_op(local, transform_op(*it));
                                          partials[c] = std::move(local);
                                      });
                        });
                }
                condition_.notify_one();
            }
        }

        {
            // 末段由调用者执行（n_chunks ≥ 2 ⇒ 末块下标 ≥ 1 ⇒ 首元素种子）
            const std::size_t c = n_chunks - 1;
            const std::size_t off = total - (base + (c < rem ? 1 : 0));
            auto beg = first;
            std::ranges::advance(beg, static_cast<std::ptrdiff_t>(off));
            run_chunk(st,
                      [&]
                      {
                          auto it = beg;
                          T local = transform_op(*it);
                          for (++it; it != last; ++it)
                              local = reduce_op(local, transform_op(*it));
                          partials[c] = std::move(local);
                      });
        }

        wait_for_latch(st.latch);
        // 失败的分块不会写自己的 partials 槽；必须先重抛，绝不合并半成品
        rethrow_if_failed(st);

        T result = std::move(partials[0]);
        for (std::size_t c = 1; c < n_chunks; ++c)
            result = reduce_op(result, partials[c]);
        return result;
    }

    // ── 并行 transform_reduce（二元输入范围）────────────────────────
    // 与一元版共享同一确定性契约（铁律 8）：分段边界只由 total 决定、
    // 块 0 以 init 为种子、其余块以首元素为种子、按块下标升序合并。
    template <typename InputIt1, typename InputIt2, typename T, typename BinaryOp, typename UnaryOp>
    T parallel_transform_reduce(InputIt1 first1, InputIt1 last1, InputIt2 first2, T init,
                                BinaryOp&& reduce_op, UnaryOp&& transform_op)
    {
        const auto total = static_cast<std::size_t>(std::ranges::distance(first1, last1));
        if (total == 0) return init;

        const std::size_t n_chunks = reduce_chunk_count(total);
        if (n_chunks == 1)
        {
            T local = init;
            auto i1 = first1;
            auto i2 = first2;
            for (; i1 != last1; ++i1, ++i2)
                local = reduce_op(local, transform_op(*i1, *i2));
            return local;
        }

        const std::size_t base = total / n_chunks;
        const std::size_t rem = total % n_chunks;
        detail::RegionState st{static_cast<int>(n_chunks)};
        // 部分和按固定块下标存放（边界 = f(total)，与调度无关）
        std::vector<T> partials(n_chunks);

        {
            std::lock_guard lock(queue_mutex_);
            tasks_.reserve(tasks_.size() + (n_chunks - 1));
            std::size_t off = 0;
            for (std::size_t c = 0; c < n_chunks - 1; ++c)
            {
                const std::size_t len = base + (c < rem ? 1 : 0);
                auto i1 = first1;
                auto i2 = first2;
                std::ranges::advance(i1, static_cast<std::ptrdiff_t>(off));
                std::ranges::advance(i2, static_cast<std::ptrdiff_t>(off));
                auto i1_end = i1;
                std::ranges::advance(i1_end, static_cast<std::ptrdiff_t>(len));
                off += len;

                if (c == 0)
                {
                    // 块 0：以 init 为种子（init 在整个归约恰好计入一次）
                    tasks_.emplace_back(
                        [this, i1, i1_end, i2, &reduce_op, &transform_op, &partials, &st, init]()
                        {
                            run_chunk(st,
                                      [&]
                                      {
                                          T local = init;
                                          auto it1 = i1;
                                          auto it2 = i2;
                                          for (; it1 != i1_end; ++it1, ++it2)
                                              local = reduce_op(local, transform_op(*it1, *it2));
                                          partials[0] = std::move(local);
                                      });
                        });
                }
                else
                {
                    // 块 c>0：以块内首元素为种子（len ≥ MIN_CHUNK ≥ 1 保证非空）
                    tasks_.emplace_back(
                        [this, i1, i1_end, i2, &reduce_op, &transform_op, &partials, &st, c]()
                        {
                            run_chunk(st,
                                      [&]
                                      {
                                          auto it1 = i1;
                                          auto it2 = i2;
                                          T local = transform_op(*it1, *it2);
                                          for (++it1, ++it2; it1 != i1_end; ++it1, ++it2)
                                              local = reduce_op(local, transform_op(*it1, *it2));
                                          partials[c] = std::move(local);
                                      });
                        });
                }
                condition_.notify_one();
            }
        }

        {
            // 末段由调用者执行（n_chunks ≥ 2 ⇒ 末块下标 ≥ 1 ⇒ 首元素种子）
            const std::size_t c = n_chunks - 1;
            const std::size_t off = total - (base + (c < rem ? 1 : 0));
            auto i1 = first1;
            auto i2 = first2;
            std::ranges::advance(i1, static_cast<std::ptrdiff_t>(off));
            std::ranges::advance(i2, static_cast<std::ptrdiff_t>(off));
            run_chunk(st,
                      [&]
                      {
                          T local = transform_op(*i1, *i2);
                          for (++i1, ++i2; i1 != last1; ++i1, ++i2)
                              local = reduce_op(local, transform_op(*i1, *i2));
                          partials[c] = std::move(local);
                      });
        }

        wait_for_latch(st.latch);
        rethrow_if_failed(st);

        T result = std::move(partials[0]);
        for (std::size_t c = 1; c < n_chunks; ++c)
            result = reduce_op(result, partials[c]);
        return result;
    }

    // ── 析构 ────────────────────────────────────────────────────────
    // ⚠ 与 finish_chunk 同一条丢唤醒规则（见其注释）：置 stop_ 必须
    // 持 queue_mutex_ 再 notify。否则 worker 可能刚判完谓词 stop_=false、
    // 尚未真正进入 wait()，本次 notify 信号丢失且之后再无人通知 →
    // worker 永久睡眠 → 下方 join 永久阻塞（退出期死锁）。
    ~ThreadPool()
    {
        {
            std::lock_guard lock(queue_mutex_);
            stop_.store(true, std::memory_order_release);
            condition_.notify_all();
        }
        for (auto& worker : workers_)
        {
            if (worker.joinable()) worker.join();
        }
    }

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    ThreadPool(ThreadPool&&) = delete;
    ThreadPool& operator=(ThreadPool&&) = delete;

    [[nodiscard]] std::size_t size() const noexcept { return workers_.size(); }

    // 累计被截断并记录的任务体异常数（#C1 诊断口）。
    // 异常本身会在发起并行区的调用线程重抛，控制流无需依赖该计数；
    // 它用于"至少记录一次、不静默吞掉"的观测（日志/探针/断言）。
    [[nodiscard]] std::size_t task_exception_count() const noexcept
    { return task_exceptions_.load(std::memory_order_relaxed); }

    // 返回当前线程的 worker 索引（0..size()-1）。
    // 非 worker 线程默认返回 0；当调用者线程在 wait_for_latch 中帮忙
    // 执行偷来的任务、或处理 parallel_for_each_indexed 的调用者分片时，
    // 返回 size()（调用者 slot，不与任何 worker 冲突）。
    [[nodiscard]] static std::size_t worker_index() noexcept { return tl_worker_index; }
};

// thread_local 定义（inline，ODR-safe）
inline thread_local std::size_t ThreadPool::tl_worker_index = 0;

// ── 全局线程池单例 ─────────────────────────────────────────────────────
inline ThreadPool& global_thread_pool()
{
    static ThreadPool pool{std::thread::hardware_concurrency()};
    return pool;
}

} // namespace nn
