
static void glfw_error_callback(int error, const char* description)
{
    std::cerr << "GLFW error " << error << ": " << description << std::endl;
}

static void glfw_framebuffer_size_callback(GLFWwindow *window, int width, int height)
{
   glViewport(0, 0, width, height);
}

inline double get_s_elapsed(u64 c1, u64 c2, int timer_freq)
{
    return (double(c2 - c1) / double(timer_freq));
}

inline double get_ms_elapsed(u64 c1, u64 c2, int timer_freq)
{
    return get_s_elapsed(c1, c2, timer_freq)*1000.0;
}

inline double get_s_to_now(u64 c1, int perf_freq)
{
   return (double(glfwGetTimerValue() - c1) / double(perf_freq));
}

inline double get_ms_to_now(u64 c1, int perf_freq)
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
TaskFuture WorkQueue<thread_count>::push(T& t, auto (T::*f)(Args...), Args&&... args)
{
   static_assert(sizeof((t.*f)(args...)) <= sizeof(TaskResult));
   int task_ID = gen_task_ID();
   tasks.emplace_back(task_ID, [&t, f, args...]() { return TaskResult{(t.*f)(args...)}; });
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
void WorkQueue<thread_count>::cancel_task(TaskFuture& f)
{
   if(f.is_valid() && !f.is_ready())
   {
      tasks_mutex.lock();

      int task_ID = f.task_ID;
      int worker_ID = -1;
      bool task_is_active = false; // Not pending or finished, i.e. being worked on by one of the workers
      while(!task_is_active && (++worker_ID < thread_count))
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
      f = {}; // Invalidate f
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


ProgramState::ProgramState(const char* window_name, int window_w, int window_h, ArenaAllocator& _allocator,
                           HubbardComputeDevice& _cdev, ErrorStream& errors, ImVec4 _clear_color) :
   clear_color(_clear_color), allocator(_allocator), model(_cdev, _allocator)
{
   glfwSetErrorCallback(glfw_error_callback);
   if(!glfwInit())
   {
      errors << "glfInit failed.";
      return;
   }

   glfwWindowHint(GLFW_DOUBLEBUFFER, 1);
   const char* glsl_version = "#version 130";
   glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
   glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

   window = glfwCreateWindow(window_w, window_h, window_name, NULL, NULL);
   if(!window)
   {
      errors << "glfCreateWindow failed.";
      return;
   }
   glfwMakeContextCurrent(window);
   glfwSwapInterval(1);
   glfwSetFramebufferSizeCallback(window, glfw_framebuffer_size_callback);

   IMGUI_CHECKVERSION();
   ImGui::CreateContext();
   ImPlot::CreateContext();
   ImGuiIO& io = ImGui::GetIO();
   io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
   ImGui::StyleColorsDark();

   ImGui_ImplGlfw_InitForOpenGL(window, true);
   ImGui_ImplOpenGL3_Init(glsl_version);

   window_flags = ImGuiWindowFlags_NoResize |
      ImGuiWindowFlags_NoMove | 
      ImGuiWindowFlags_NoCollapse | 
      ImGuiWindowFlags_NoNavFocus | 
      ImGuiWindowFlags_NoBringToFrontOnFocus | 
      ImGuiWindowFlags_NoSavedSettings;

   ImGuiStyle& style = ImGui::GetStyle();
   style.Alpha = 1.0f;
   style.WindowBorderSize = 1.0f;

   ImGui::PushStyleColor(ImGuiCol_TitleBgActive , {0.0f, 0.0f, 0.0f, 0.0f}); 
   ImGui::PushStyleColor(ImGuiCol_TitleBgCollapsed, {0.0f, 0.0f, 0.0f, 0.0f}); 

   timer_freq = glfwGetTimerFrequency();

   params = {in_T, in_U, in_Ns, in_N_up, in_N_dn};
   params_sz = hubbard_memory_requirements(params);
   int_args = {.lower = 1e-8, .upper = 100, .abs_tol = 1e-8, .rel_tol = 1e-6, .min_steps = 2, .max_steps = 60};

   memcpy(counter_buf, "0 s", 4);
   sprintf_s(result_E0_buf, "E0: ");
   sprintf_s(result_compute_time_buf, "Compute time: ");
   format_memory(params_memory_buf, ARRAY_SIZE(params_memory_buf), params_sz.workspace_size);

   plot_E_vals.reserve(MAX_PLOT_PTS);
   plot_x_vals.reserve(MAX_PLOT_PTS);
   halffilling_E_vals.reserve(MAX_PLOT_PTS);
   dimer_E_vals.reserve(MAX_PLOT_PTS);
   nonint_E_vals.reserve(MAX_PLOT_PTS);
   atomic_E_vals.reserve(MAX_PLOT_PTS);
}

ProgramState::~ProgramState()
{
   ImGui::PopStyleColor();
   ImGui::PopStyleColor();

   ImGui_ImplOpenGL3_Shutdown();
   ImGui_ImplGlfw_Shutdown();
   ImPlot::DestroyContext();
   ImGui::DestroyContext();
   glfwDestroyWindow(window);
   glfwTerminate();
}

bool ProgramState::is_running()
{
   return !glfwWindowShouldClose(window);
}

void ProgramState::handle_events()
{
   glfwPollEvents();
}

void ProgramState::render_UI()
{
   // TODO: Move to handle events
   if(computing && compute_result.is_valid() && compute_result.is_ready())
   {
      compute_end_counter = glfwGetTimerValue();
      last_compute_elapsed_s = get_s_elapsed(compute_start_counter, compute_end_counter, timer_freq);
      computing = false;
      real E0 = compute_result.get<real>();
      sprintf_s(result_E0_buf, "E0: %f", E0);
      sprintf_s(result_compute_time_buf, "Compute time: %llu s", u64(last_compute_elapsed_s));

      if(plot_mode)
      {
         plot_E_vals.push_back(E0/params.Ns);
      }
      else
      {
         prof_labels.resize(0);
         prof_total.resize(0);
         prof_mean.resize(0);
         prof_min.resize(0);
         prof_max.resize(0);
         prof_count.resize(0);
         prof_y_ticks.resize(0);

         int trace_idx = 1;
         for(auto a : TimedScope::trace_stats)
         {
            prof_labels.push_back(a.first.data());
            const auto& stats = a.second;
            prof_total.push_back(stats.total);
            prof_mean.push_back(stats.mean());
            prof_min.push_back(stats.min);
            prof_max.push_back(stats.max);
            prof_count.push_back(stats.count);
            prof_y_ticks.push_back(trace_idx++);
         }
         TimedScope::clear();
      }
   }

   int display_w, display_h;
   glfwGetFramebufferSize(window, &display_w, &display_h);
   ImVec2 plot_win_pos       = {0.0f*display_w, 0.0f*display_h};
   ImVec2 plot_win_size      = {0.8f*display_w, 0.8f*display_h};

   ImVec2 settings_win_pos   = {0.8f*display_w, 0.0f*display_h};
   ImVec2 settings_win_size  = {0.2f*display_w - 1, 0.8f*display_h};

   ImVec2 profiling_win_pos  = {0.0f*display_w, 0.8f*display_h};
   ImVec2 profiling_win_size = {1.0f*display_w - 1, 0.2f*display_h};

   ImGui_ImplOpenGL3_NewFrame();
   ImGui_ImplGlfw_NewFrame();
   ImGui::NewFrame();

   ImGui::SetNextWindowPos(plot_win_pos);
   ImGui::SetNextWindowSize(plot_win_size);
   ImGui::Begin("Plotting", 0, window_flags);
   ImPlot::SetNextAxesToFit();

   ImPlot::PushStyleVar(ImPlotStyleVar_FitPadding, ImVec2(0.1f, 0.1f));
   if(plot_mode && ImPlot::BeginPlot("##plot1", ImVec2(-1, -1)))
   {
      ImPlot::SetupAxes(plot_T_range ? "T" : "U", "E0/Ns");
      //ImPlot::SetupAxes("U/T", "E0/Ns");
      ImPlot::SetNextLineStyle(IMPLOT_AUTO_COL, 1.5f);
      ImPlot::SetNextMarkerStyle(ImPlotMarker_Square);

      ImPlot::PlotLine("Computed", plot_x_vals.data(), plot_E_vals.data(), plot_E_vals.size());

      if(show_halffilling && is_halffilling(params))
      {
         assert(plot_x_vals.size() >= halffilling_E_vals.size());
         ImPlot::PlotLine("Lieb & Wu", plot_x_vals.data(), halffilling_E_vals.data(), halffilling_E_vals.size());
      }
      if(show_noninteracting)
      {
         assert(plot_x_vals.size() >= nonint_E_vals.size());
         ImPlot::SetNextLineStyle(IMPLOT_AUTO_COL, 1.5f);
         ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 5.0f, ImVec4(0, 0, 0, 0), IMPLOT_AUTO);
         ImPlot::PlotScatter("U = 0, ground truth", plot_x_vals.data(), nonint_E_vals.data(), nonint_E_vals.size());
      }
      if(show_atomic_limit)
      {
         assert(plot_x_vals.size() >= atomic_E_vals.size());
         ImPlot::SetNextLineStyle(IMPLOT_AUTO_COL, 1.5f);
         ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 5.0f, ImVec4(0, 0, 0, 0), IMPLOT_AUTO);
         ImPlot::PlotScatter("T = 0, ground truth", plot_x_vals.data(), atomic_E_vals.data(), atomic_E_vals.size());
      }
      if(show_dimer)
      {
         assert(plot_x_vals.size() >= dimer_E_vals.size());
         ImPlot::SetNextLineStyle(IMPLOT_AUTO_COL, 1.5f);
         ImPlot::SetNextMarkerStyle(ImPlotMarker_Cross);//, 2.0f);
         ImPlot::PlotLine("Dimer, ground truth", plot_x_vals.data(), dimer_E_vals.data(), dimer_E_vals.size());
      }

      ImPlot::EndPlot();
   }
   ImPlot::PopStyleVar();
   ImGui::End();

   ImGui::SetNextWindowPos(settings_win_pos);
   ImGui::SetNextWindowSize(settings_win_size);
   ImGui::Begin("Settings", 0, window_flags);

   ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 3.0f);
   ImGui::PushStyleVar(ImGuiStyleVar_Alpha, computing ? 0.5 : 1.0);

   bool should_compute = false;
   if(ImGui::Button("Compute"))
   {
      if(plot_mode)
      {
         plot_done = false;
         plot_x_step_idx = 0;
         plot_E_vals.resize(0);
         plot_x_vals.resize(0);

         halffilling_E_vals.resize(0);
         dimer_E_vals.resize(0);
         nonint_E_vals.resize(0);
         atomic_E_vals.resize(0);

         if(plot_T_range)
         { 
            real T_min = T_range[0];
            params.T = T_min; 
            in_T = T_min; 

            real cur_T = T_range[0];
            int i = 0;
            while(cur_T < T_range[1])
            {
               plot_x_vals.push_back(cur_T);
               cur_T = T_range[0] + (++i)*T_range[2];
            }
            plot_x_vals.push_back(T_range[1]);
         }
         else
         {
            real U_min = U_range[0];
            params.U = U_min; 
            in_U = U_min;

            real cur_U = U_range[0];
            int i = 0;
            while(cur_U < U_range[1])
            {
               plot_x_vals.push_back(cur_U);
               cur_U = U_range[0] + (++i)*U_range[2];
            }
            plot_x_vals.push_back(U_range[1]);
         }
      }
      else
      {
         should_compute = true;
      }
   }

   if(plot_mode && !plot_done)
   {
      should_compute = true;
   }

   if(should_compute && !computing)
   {
      compute_start_counter = glfwGetTimerValue();

      // TODO: Move to handle events
      if(plot_mode && !plot_done)
      {
         if(plot_T_range)
         {
            real T_min = T_range[0];
            real T_max = T_range[1];
            real dT = T_range[2];
            if(params.T >= T_max || plot_x_step_idx >= MAX_PLOT_PTS)
            {
               plot_done = true;
               plot_x_step_idx = 0;
            }
            else
            {
               params.T = T_min + plot_x_step_idx*dT;
               if(params.T >= T_max) { params.T = T_max; }

               in_T = params.T;
               plot_x_step_idx++;
            }

         }
         else
         {
            real U_min = U_range[0];
            real U_max = U_range[1];
            real dU = U_range[2];
            if(params.U >= U_max || plot_x_step_idx >= MAX_PLOT_PTS)
            {
               plot_done = true;
               plot_x_step_idx = 0;
            }
            else
            {
               params.U = U_min + plot_x_step_idx*dU;
               if(params.U >= U_max) { params.U = U_max; }

               in_U = params.U;
               plot_x_step_idx++;
            }
         }

         if(!plot_done)
         {
            if(show_halffilling && is_halffilling(params))
            {
               halffilling_E_vals.push_back(halffilled_E_per_N(params.T, params.U, int_args));
            }
            if(show_dimer)
            {
               dimer_E_vals.push_back(dimer_E0(params, BCS::PERIODIC)/params.Ns);
            }
            if(show_noninteracting)
            {
               nonint_E_vals.push_back(noninteracting_E0(params, BCS::PERIODIC)/params.Ns);
            }
            if(show_atomic_limit)
            {
               atomic_E_vals.push_back(atomic_E0(params)/params.Ns);
            }
         }
      }

      if(!plot_done || !plot_mode)
      {
         model.set_params(params);
         compute_result = work_queue.push(model, &HubbardModel::E0);
         computing = true;
      }
      else
      {
         computing = false;
      }
   }
   ImGui::PopStyleVar();

   ImGui::SameLine();
   if(ImGui::Button("Cancel"))
   {
      assert(!"Not implemented!"); // The actual tasks pushed to the work queue don't handle thread termination currently

      computing = false;
      compute_end_counter = glfwGetTimerValue();

      // TODO: Move to handle_events
      work_queue.cancel_task(compute_result);
      memcpy(counter_buf, "0 s", 4);
   }
   ImGui::PopStyleVar();

   if(computing)
   {
      sprintf_s(counter_buf, "%llu s", u64(get_s_to_now(compute_start_counter, timer_freq)));
   }
   ImGui::Text(counter_buf);

   if(ImGui::CollapsingHeader("Parameters"))
   {
      ImGui::InputFloat("T", &in_T);
      ImGui::InputFloat("U", &in_U);
      ImGui::SliderInt("Ns", &in_Ns, 1, MAX_SITE_COUNT);
      int N_dn_min, N_dn_max;
      int N_up_min, N_up_max;
      if(force_halffilling)
      {
         in_Ns += in_Ns & 1;
         in_N_up = int(in_Ns/2);
         in_N_dn = in_N_up;
         N_dn_min = in_N_up;
         N_dn_max = in_N_up;
         N_up_min = in_N_up;
         N_up_max = in_N_up;
      }
      else
      {
         N_dn_min = (in_N_dn == 0) ? 1 : 0;
         N_dn_max = in_Ns;
         N_up_min = (in_N_up == 0) ? 1 : 0;
         N_up_max = in_Ns;
      }
      ImGui::SliderInt("N_up", &in_N_up, N_dn_min, N_dn_max);
      ImGui::SliderInt("N_dn", &in_N_dn, N_up_min, N_up_max);
      if(in_N_up > in_Ns) { in_N_up = in_Ns; }
      if(in_N_dn > in_Ns) { in_N_dn = in_Ns; }

      HubbardParams new_params = {in_T, in_U, in_Ns, in_N_up, in_N_dn};
      bool params_changed = (params != new_params);
      params = new_params;

      ImGui::PushStyleVar(ImGuiStyleVar_Alpha, plot_T_range ? 1.0 : 0.5);
      if(ImGui::InputFloat3("T range", T_range))
      {
         if(T_range[2] < 0) { T_range[2] = 0; };
         plot_T_range = true;
      }
      ImGui::PopStyleVar();

      ImGui::PushStyleVar(ImGuiStyleVar_Alpha, plot_T_range ? 0.5 : 1.0);
      if(ImGui::InputFloat3("U range", U_range))
      {
         if(U_range[2] < 0) { U_range[2] = 0; };
         plot_T_range = false;
      }
      ImGui::PopStyleVar();

      ImGui::InputInt3("Ns range", Ns_range);
      Ns_range[0] = std::clamp(Ns_range[0], 1, MAX_SITE_COUNT);
      Ns_range[1] = std::clamp(Ns_range[1], Ns_range[0], MAX_SITE_COUNT);

      ImGui::Checkbox("Half-filling", &force_halffilling);

      if(params_changed)
      {
         params_sz = hubbard_memory_requirements(params);
         format_memory(params_memory_buf, ARRAY_SIZE(params_memory_buf), params_sz.workspace_size);
      }
      ImGui::Text("Min memory (excl. VRAM): "); // Excludes memory allocated in compute.lib
      ImGui::SameLine();
      ImGui::Text(params_memory_buf);
   }

   if(ImGui::CollapsingHeader("Plot"))
   {
      ImGui::MenuItem("E0/Ns", 0, &plot_mode);
   }
   if(ImGui::CollapsingHeader("Benchmark results"))
   {
      ImGui::Checkbox("Half-filling, asymptotic", &show_halffilling);
      ImGui::Checkbox("Non-interacting (U = 0)", &show_noninteracting);
      ImGui::Checkbox("Atomic limit (T = 0)", &show_atomic_limit);
      ImGui::Checkbox("Dimer", &show_dimer);
   }
   if(ImGui::CollapsingHeader("Results"))
   {
      ImGui::Text(result_E0_buf);
      ImGui::Text(result_compute_time_buf);
   }
   ImGui::Text(BUILD_TYPE_STR);

   ImGui::End();

   ImGui::SetNextWindowPos(profiling_win_pos);
   ImGui::SetNextWindowSize(profiling_win_size);
   ImGui::Begin("Profiling", 0, window_flags);
   if(!plot_mode)
   {
#ifdef HUBBARD_DEBUG
      if(prof_mean.size() > 0 && ImPlot::BeginPlot("##profplot", ImVec2(-1, -1)))
      {
         ImPlot::SetupLegend(ImPlotLocation_SouthEast);
         ImPlot::SetupAxisFormat(ImAxis_X1, "%g ms");

         ImPlot::SetupAxisTicks(ImAxis_Y1, prof_y_ticks.data(), prof_y_ticks.size(), prof_labels.data(), false);

         ImPlot::PlotBars("Total", prof_total.data(), prof_total.size(), 0.5, 1, ImPlotBarsFlags_Horizontal);
         ImPlot::PushStyleVar(ImPlotStyleVar_FillAlpha, 0.5f);
         ImPlot::PushStyleVar(ImPlotStyleVar_LineWeight, 2.5f);
         ImPlot::PlotBars("Max", prof_max.data(), prof_max.size(), 0.5, 1, ImPlotBarsFlags_Horizontal);
         ImPlot::PlotBars("Mean", prof_mean.data(), prof_mean.size(), 0.5, 1, ImPlotBarsFlags_Horizontal);
         ImPlot::PlotBars("Min", prof_min.data(), prof_min.size(), 0.5, 1, ImPlotBarsFlags_Horizontal);

         for(int i = 0; i < prof_total.size(); i++)
         {
            ImPlot::Annotation(prof_total[i], i + 1, ImVec4(0, 0, 0, 0), ImVec2(0, -5), true, "%d, %.2f%%", prof_count[i], prof_total[i]/(10.0*last_compute_elapsed_s));
         }

         ImPlot::PopStyleVar();
         ImPlot::PopStyleVar();

         ImPlot::EndPlot();
      }
#else
      ImPlot::Text("Use debug build to access profiling.");
#endif
   }
   ImGui::End();

   ImGui::Render();
   glClearColor(clear_color.x*clear_color.w, clear_color.y*clear_color.w, clear_color.z*clear_color.w, clear_color.w);
   glClear(GL_COLOR_BUFFER_BIT);
   ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

   glfwSwapBuffers(window);
}
