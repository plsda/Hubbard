#ifndef UI_H

#ifdef HUBBARD_USE_CUDA
   #ifdef HUBBARD_DEBUG
      #define BUILD_TYPE_STR "Debug build with CUDA"
   #else
      #define BUILD_TYPE_STR "Release build with CUDA"
   #endif
#else
   #ifdef HUBBARD_DEBUG
      #define BUILD_TYPE_STR "Debug build"
   #else
      #define BUILD_TYPE_STR "Release build"
   #endif
#endif

template <class T> struct ScalarRange;
struct ProgramState;
struct Task;
struct TaskFuture;
template<size_t thread_count = 4> class WorkQueue;
class ProgramState;

using TaskResult = std::variant<bool, int, real, void*>;

static void glfw_error_callback(int error, const char* description);
static void glfw_framebuffer_size_callback(GLFWwindow *window, int width, int height);
inline double get_s_elapsed(u64 c1, u64 c2, int timer_freq);
inline double get_ms_elapsed(u64 c1, u64 c2, int timer_freq);
inline double get_s_to_now(u64 c1, int perf_freq);
inline double get_ms_to_now(u64 c1, int perf_freq);
void sleep(double ms);

void format_memory(char* result, size_t max_count, size_t bytes)
{
   double mem = double(bytes)/double(1024*1024);
   const char* mem_format = "%.3f MiB";
   if(mem > 999.0)
   { 
      mem /= double(1024);
      mem_format = "%.3f GiB";
      if(mem > 999.0)
      { 
         mem /= double(1024);
         mem_format = "%.3f TiB";
      }
   }

   if(mem < 0.001)
   {
      mem_format = "<0.001 MiB";
      sprintf_s(result, max_count, mem_format);
   }
   else
   {
      sprintf_s(result, max_count, mem_format, mem);
   }
}

bool check_future(const std::future<TaskResult>& f)
{
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
   TaskFuture(std::future<TaskResult>&& _future, int _task_ID) : future(std::move(_future)), task_ID(_task_ID) {}
   TaskFuture(Task& task, int _task_ID) : future(task.task.get_future()), task_ID(_task_ID) {}

   template<class T>
   constexpr T get() { return std::get<T>(future.get()); }

   bool is_valid() const { return future.valid(); }
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
   TaskFuture push(T& t, auto (T::*f)(Args...), Args&&... args);

   TaskFuture push(Task& task);

   void cancel_task(TaskFuture& f);

private:
   std::deque<Task> tasks;
   std::mutex tasks_mutex;
   std::counting_semaphore<std::numeric_limits<std::ptrdiff_t>::max()> pending{0};

   std::array<std::jthread, thread_count> workers;
   std::array<int, thread_count> worker_statuses{};

   bool pop(int worker_ID);

   int gen_task_ID() { return task_counter++; }

   //std::jthread gen_worker(int ID) { return std::jthread(worker_proc, ID, this); }
};

class ProgramState
{
public:
   ProgramState(const char* window_name, int window_w, int window_h, ArenaAllocator& _allocator,
                HubbardComputeDevice& _cdev, ErrorStream& errors, ImVec4 _clear_color = ImVec4(0.0f, 0.0f, 0.0f, 1.0f));

   ~ProgramState();

   bool is_running();
   void handle_events();
   void render_UI();

private:
   GLFWwindow* window;
   ImGuiWindowFlags window_flags;
   ImVec4 clear_color;
   u64 timer_freq;

   float in_T = 1.0f;
   float in_U = 1.0f;
   int in_Ns = 5;
   int in_N_up = 2;
   int in_N_dn = 2;
   HubbardParams params;
   HubbardSizes params_sz;
   IntArgs int_args;

   float x_step = 0.1;
   float T_range[3] = {0.1, 1, x_step};
   float U_range[3] = {0, 1, x_step};
   int Ns_range[3] = {1, 1, 1};
   bool force_halffilling = false;

   bool computing = false;
   char counter_buf[256];
   char result_E0_buf[256];
   char result_compute_time_buf[256];
   char params_memory_buf[64];

   u64 compute_start_counter = 0;
   u64 compute_end_counter = 0;
   bool show_halffilling = false;
   bool show_noninteracting = false;
   bool show_atomic_limit = false;
   bool show_dimer = false;
   bool plot_mode = false;
   bool plot_done = true;
   bool plot_T_range = true;
   int plot_x_step_idx = 0;

   double last_compute_elapsed_s = 0;

   WorkQueue<4> work_queue;
   ArenaAllocator& allocator;
   TaskFuture compute_result;
   HubbardModel model;

   const int MAX_PLOT_PTS = 500;
   std::vector<real> plot_E_vals;
   std::vector<real> plot_x_vals;

   std::vector<real> halffilling_E_vals;
   std::vector<real> dimer_E_vals;
   std::vector<real> nonint_E_vals;
   std::vector<real> atomic_E_vals;

   std::vector<const char*> prof_labels;
   std::vector<float> prof_total;
   std::vector<float> prof_mean;
   std::vector<float> prof_min;
   std::vector<float> prof_max;
   std::vector<int> prof_count;
   std::vector<double> prof_y_ticks;
};

#define UI_H
#endif
