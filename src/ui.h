#ifndef UI_H

template <class T> struct ScalarRange;
struct ProgramState;
struct Task;
struct TaskFuture;
template<size_t thread_count = 4> class WorkQueue;

using TaskResult = std::variant<bool, int, real, void*>;

static void glfw_error_callback(int error, const char* description);
static void glfw_framebuffer_size_callback(GLFWwindow *window, int width, int height);
inline float get_s_elapsed(u64 c1, u64 c2, int timer_freq);
inline float get_ms_elapsed(u64 c1, u64 c2, int timer_freq);
inline float get_s_to_now(u64 c1, int perf_freq);
inline float get_ms_to_now(u64 c1, int perf_freq);
void sleep(double ms);

template <class T>
struct ScalarRange
{
   T min;
   T max;
   T step;
   int step_count;

   void clamp(T clamp_min, T clamp_max)
   {
      min = std::clamp(min, clamp_min, clamp_max);
      max = std::clamp(max, min, clamp_max);
   }
};

struct ProgramState
{
   /* TODO: Move all state variables here */
   // Memory
   // Work queue 
   // Event queue
};

bool check_future(const std::future<TaskResult>& f)
{
   //std::future_status status = f.wait_for(0s);
   //return (status == std::future_status::ready) || (status == std::future_status::timeout);
   
   // NOTE: Timeout not checked
   return (f.wait_for(0s) == std::future_status::ready);
}

struct Task
{
   template<class... Args>
   Task(int task_ID, Args&&... args) : ID(task_ID), task(std::forward<Args>(args)...) { }

   void operator()() { task(); }

   int ID;
   std::packaged_task<TaskResult()> task;
};

struct TaskFuture
{
   TaskFuture() = default;
   TaskFuture(Task& task, int _task_ID) : future(task.task.get_future()), task_ID(_task_ID) { }

   template<class T>
   constexpr T get() { return std::get<T>(future.get()); }

   bool is_ready() const { return check_future(future); }

   int get_ID() const
   {
      if(!is_ready())
      {
         return task_ID;
      }

      return -1;
   }

   std::future<TaskResult> future;
   int task_ID;
};

// NOTE: The public functions (and gen_task_ID) of WorkQueue are not thread-safe
template<size_t thread_count>
class WorkQueue
{
   static inline u32 task_counter = 1;

   template<size_t C>
   friend void worker_proc(WorkQueue<C>* const q, const int worker_ID);

public:
   WorkQueue();
   
   template <class... Args>
   TaskFuture push(std::invocable<Args...> auto f, Args&&... args);

   template <class T, class... Args>
   TaskFuture push(T&& t, auto (T::*f)(Args...), Args&&... args);

   TaskFuture push(Task& task);

   void cancel_task(const TaskFuture& f);

private:
   std::deque<Task> tasks;
   std::mutex tasks_mutex;
   std::counting_semaphore<std::numeric_limits<std::ptrdiff_t>::max()> pending{0};

   std::array<std::jthread, thread_count> workers;
   std::array<int, thread_count> worker_statuses;

   bool pop(int worker_ID);

   int gen_task_ID() { return task_counter++; }

   //std::jthread gen_worker(int ID) { return std::jthread(worker_proc, ID, this); }
};

#define UI_H
#endif
