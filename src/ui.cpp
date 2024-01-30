
static void glfw_error_callback(int error, const char* description)
{
    std::cerr << "GLFW error " << error << ": " << description << std::endl;
}

static void glfw_framebuffer_size_callback(GLFWwindow *window, int width, int height)
{
   glViewport(0, 0, width, height);
}

inline float get_s_elapsed(u64 c1, u64 c2, int timer_freq)
{
    return (double(c2 - c1) / double(timer_freq));
}

inline float get_ms_elapsed(u64 c1, u64 c2, int timer_freq)
{
    return get_s_elapsed(c1, c2, timer_freq)*1000.0;
}

inline float get_s_to_now(u64 c1, int perf_freq)
{
   return (double(glfwGetTimerValue() - c1) / double(perf_freq));
}

inline float get_ms_to_now(u64 c1, int perf_freq)
{
   return get_s_to_now(c1, perf_freq)*1000.0;
}

void sleep(double ms)
{
#ifdef _WIN32
    Sleep(DWORD(ms));
#else
    usleep(useconds_t(ms*1000.0));
#endif
}

template<size_t thread_count>
void worker_proc(WorkQueue<thread_count>* const q, const int worker_ID)
{
   for(;;)
   {
      if(!q->pop(worker_ID))
      {
         q->pending.acquire();
      }
   }
}

template<size_t thread_count>
WorkQueue<thread_count>::WorkQueue() : workers{make_enumerated_carray<std::jthread, thread_count>(worker_proc<thread_count>, this)} { }

template<size_t thread_count> template<class... Args>
TaskFuture WorkQueue<thread_count>::push(std::invocable<Args...> auto f, Args&&... args)
{
   static_assert(sizeof(f(args...)) <= sizeof(TaskResult));
   int task_ID = gen_task_ID();
   tasks.emplace_back(task_ID, [&]() { return TaskResult{f(args...)}; });
   TaskFuture result(tasks.back(), task_ID);
   
   pending.release();
   return result;
}

template<size_t thread_count> template <class T, class... Args>
TaskFuture WorkQueue<thread_count>::push(T&& t, auto (T::*f)(Args...), Args&&... args)
{
   static_assert(sizeof((t.*f)(args...)) <= sizeof(TaskResult));
   int task_ID = gen_task_ID();
   tasks.emplace_back(task_ID, [&]() { return TaskResult{(t.*f)(args...)}; });
   TaskFuture result(tasks.back(), task_ID);
   
   pending.release();
   return result;
}

template<size_t thread_count>
TaskFuture WorkQueue<thread_count>::push(Task& task)
{
   int task_ID = gen_task_ID();
   tasks.push_back({.ID = task_ID, .task = task});
   TaskFuture result(tasks.back(), task_ID);

   pending.release();
   return result;
}

template<size_t thread_count>
void WorkQueue<thread_count>::cancel_task(const TaskFuture& f)
{
   if(!f.is_ready())
   {
      tasks_mutex.lock();

      int task_ID = f.task_ID;
      int worker_ID = 0;
      bool task_is_active = false; // Not pending or finished, i.e. being worked on by one of the workers
      for(; worker_ID < thread_count && !task_is_active; worker_ID++)
      { 
         task_is_active = (worker_statuses[worker_ID] == task_ID);
      }

      if(task_is_active)
      {
         // If the task is being worked on, stop the thread that has the target task and update worker_statuses
         // (doesn't matter if the worker just finishes the task and clears its status to 0, terminate the thread anyway)
         if(worker_statuses[worker_ID] == task_ID)
         {
            std::jthread& worker = workers[worker_ID];
            if(worker.request_stop())
            {
               worker.detach();
               worker_statuses[worker_ID] = 0;
               worker = std::jthread(worker_proc<thread_count>, this, worker_ID);
            }
         }
      }
      else if(!f.is_ready())
      {
         // If pending, remove from the task queue
         tasks.erase(std::find_if(tasks.begin(), tasks.end(), [task_ID](auto& t) { return t.ID == task_ID; }));
      }

      tasks_mutex.unlock();
   }
}

template<size_t thread_count>
bool WorkQueue<thread_count>::pop(int worker_ID)
{
   bool acquired_task = false;
   if(tasks.size() > 0 && tasks_mutex.try_lock())
   {
      auto task = std::move(tasks.back());
      tasks.pop_back();
      worker_statuses[worker_ID] = task.ID;
      tasks_mutex.unlock();
      task();
      worker_statuses[worker_ID] = 0;
      acquired_task = true;
   }

   return acquired_task;
}
