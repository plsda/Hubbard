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

struct ProgramState;
struct Task;
struct TaskFuture;
template<size_t thread_count = 4> class WorkQueue;
struct PlotElement;
class ProgramState;

using TaskResult = std::variant<std::monostate, bool, int, real, void*, std::vector<real>::iterator>;

enum class PLOT_QUANTITY
{
   COMPUTED_E,
   DIMER_E,
   NONINT_E,
   ATOMIC_E,
   HF_E,

   COUNT
};

enum class PLOT_TYPE
{
   LINE,
   SCATTER,
};

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
   bool is_pending() const { return is_valid() && !is_ready(); }

#if 0
   TaskFuture& operator=(TaskFuture&& other)
   {
      if(is_valid())
      {
         future.get();
      }
      future = std::move(other.future);
      task_ID = other.task_ID;

      return *this;
   }
   TaskFuture& operator=(const TaskFuture& other) = delete;
#endif

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
   friend void worker_proc(std::stop_token s, WorkQueue<C>* const q, const int worker_ID);

public:
   WorkQueue();
   ~WorkQueue()
   {
      for(auto& w : workers)
      {
         w.request_stop();
      }
      for(auto& w : workers)
      {
         pending.release();
      }
   }
   
   template<class... Args>
   TaskFuture push(std::invocable<Args...> auto f, Args&&... args);

   template<class... Args>
   TaskFuture push_valcap(std::invocable<Args...> auto f, Args&&... args);

   template<class T, class... Args>
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

struct PlotElement
{
   PLOT_QUANTITY value_type;
   PLOT_TYPE type;
   const char* legend;
   bool show;
   bool enabled;
   ImPlotMarker marker_style;
   const std::vector<float>* x;
   std::vector<float> y;
   TaskFuture comp_status;
};

static const std::vector<float> ZERO_FVEC = {0};

struct ProgramState
{
   explicit ProgramState(const char* window_name, int window_w, int window_h, ArenaAllocator& _allocator,
                         HubbardComputeDevice& _cdev, ErrorStream& errors, ImVec4 _clear_color = ImVec4(0.0f, 0.0f, 0.0f, 1.0f));

   ~ProgramState();

   bool is_running();
   void handle_events();
   void render_UI();

   ProgramState& operator=(const ProgramState&) = delete;
   ProgramState& operator=(ProgramState&&) = delete;

   ProgramState* _this;
   GLFWwindow* window;
   ImGuiWindowFlags window_flags;
   ImVec4 clear_color;
   u64 timer_freq;

   float in_T = 1.0f;
   float in_U = 1.0f;
   int in_Ns = 5;
   int in_N_up = 2;
   int in_N_dn = 2;
   int N_dn_min = 0, N_dn_max = 0;
   int N_up_min = 0, N_up_max = 0;
   HubbardParams params;
   HubbardSizes params_sz;
   IntArgs int_args;

   ScalarRange<float> T_range;
   ScalarRange<float> U_range;
   bool force_halffilling = false;

   bool compute = false;
   char counter_buf[256];
   char result_E0_buf[256];
   char result_compute_time_buf[256];
   char params_memory_buf[64];

   u64 compute_start_counter = 0;
   u64 compute_end_counter = 0;
   bool plot_mode = false;
   bool plot_T_range = true;
   bool param_input = true;
   bool params_changed = true;
   bool force_clear_profiling_results = false;

   double last_compute_elapsed_s = 0;

   WorkQueue<4> work_queue;
   ArenaAllocator& allocator;
   TaskFuture* compute_result;
   HubbardModel model;

   std::vector<const char*> prof_labels;
   std::vector<float> prof_total;
   std::vector<float> prof_mean;
   std::vector<float> prof_min;
   std::vector<float> prof_max;
   std::vector<int> prof_count;
   std::vector<double> prof_y_ticks;
   std::vector<double> prof_percentage;

   std::vector<real> plot_x_vals;
   std::vector<PlotElement> plot_elements;
};


#define UI_H
#endif
