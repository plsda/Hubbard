
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
void worker_proc(std::stop_token s, WorkQueue<thread_count>* const q, const int worker_ID)
{
   while(!s.stop_requested())
   {
      q->pending.acquire();
      q->pop(worker_ID);
   }
}

template<size_t thread_count>
WorkQueue<thread_count>::WorkQueue() : workers{make_enumerated_carray<std::jthread, thread_count>(worker_proc<thread_count>, this)} { }

template<size_t thread_count> template<class... Args>
TaskFuture WorkQueue<thread_count>::push(std::invocable<Args...> auto f, Args&&... args)
{
   // NOTE: Always return a value from task function
   static_assert(sizeof(f(args...)) <= sizeof(TaskResult));

   tasks_mutex.lock();
   int task_ID = gen_task_ID();
   // NOTE: Capturing by reference
   tasks.emplace_back(task_ID, [&]() { return TaskResult{f(args...)}; });
   TaskFuture result(tasks.back(), task_ID);
   tasks_mutex.unlock();
   
   pending.release();
   return result;
}

template<size_t thread_count> template<class... Args>
TaskFuture WorkQueue<thread_count>::push_valcap(std::invocable<Args...> auto f, Args&&... args)
{
   // NOTE: Always return a value from task function
   static_assert(sizeof(f(args...)) <= sizeof(TaskResult));

   tasks_mutex.lock();
   int task_ID = gen_task_ID();
   // NOTE: Capturing by value
   tasks.emplace_back(task_ID, [=]() { return TaskResult{f(args...)}; });
   TaskFuture result(tasks.back(), task_ID);
   tasks_mutex.unlock();
   
   pending.release();
   return result;
}

template<size_t thread_count> template <class T, class... Args>
TaskFuture WorkQueue<thread_count>::push(T& t, auto (T::*f)(Args...), Args&&... args)
{
   static_assert(sizeof((t.*f)(args...)) <= sizeof(TaskResult));

   tasks_mutex.lock();
   int task_ID = gen_task_ID();
   tasks.emplace_back(task_ID, [&t, f, args...]() { return TaskResult{(t.*f)(args...)}; });
   TaskFuture result(tasks.back(), task_ID);
   tasks_mutex.unlock();
   
   pending.release();
   return result;
}

template<size_t thread_count>
TaskFuture WorkQueue<thread_count>::push(Task& task)
{
   tasks_mutex.lock();
   int task_ID = gen_task_ID();
   tasks.push_back({.ID = task_ID, .task = task});
   TaskFuture result(tasks.back(), task_ID);
   tasks_mutex.unlock();

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
   tasks_mutex.lock(); 
   if(tasks.size() > 0)
   {
      auto task = std::move(tasks.back());
      tasks.pop_back();
      worker_statuses[worker_ID] = task.ID;
      tasks_mutex.unlock();
      task();
      worker_statuses[worker_ID] = 0;
   }
   else
   {
      tasks_mutex.unlock();
   }

   return true;
}

ProgramState::ProgramState(const char* window_name, int window_w, int window_h, ArenaAllocator& _allocator,
                           HubbardComputeDevice& _cdev, ErrorStream& errors, ImVec4 _clear_color) :
   clear_color(_clear_color), allocator(_allocator), model(_cdev, _allocator), cdev(_cdev), _this(this)
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

   const GLFWvidmode* mode = glfwGetVideoMode(glfwGetPrimaryMonitor());
   window = glfwCreateWindow(std::min(window_w, mode->width), std::min(window_h, mode->height), window_name, NULL, NULL);
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
   T_range = {.interval = {0.1f, 1.0f}, .dim = 10};
   U_range = {.interval = {0.0f, 1.0f}, .dim = 10};
   int_args = {.lower = 1e-8, .upper = 100, .abs_tol = 1e-8, .rel_tol = 1e-6, .min_steps = 2, .max_steps = 60};

   memcpy(counter_buf, "0 s", 4);
   sprintf_s(result_E0_buf, "E0: ");
   sprintf_s(result_compute_time_buf, "Compute time: ");
   format_memory(params_memory_buf, ARRAY_SIZE(params_memory_buf), params_sz.workspace_size);

   plot_elements.reserve(size_t(PLOT_QUANTITY::COUNT));
   plot_elements.push_back({.value_type = PLOT_QUANTITY::COMPUTED_E, .type = PLOT_TYPE::LINE,    .legend = "Computed",            .show = false, .enabled = false, .marker_style = ImPlotMarker_Square, .x = &plot_x_vals, .y = {}, .comp_status = {}});
   plot_elements.push_back({.value_type = PLOT_QUANTITY::DIMER_E,    .type = PLOT_TYPE::LINE,    .legend = "Dimer, ground truth", .show = false, .enabled = false, .marker_style = ImPlotMarker_Square, .x = &plot_x_vals, .y = {}, .comp_status = {}});
   plot_elements.push_back({.value_type = PLOT_QUANTITY::NONINT_E,   .type = PLOT_TYPE::SCATTER, .legend = "U = 0, ground truth", .show = false, .enabled = false, .marker_style = ImPlotMarker_Circle, .x = &plot_x_vals, .y = {}, .comp_status = {}});
   plot_elements.push_back({.value_type = PLOT_QUANTITY::ATOMIC_E,   .type = PLOT_TYPE::SCATTER, .legend = "T = 0, ground truth", .show = false, .enabled = false, .marker_style = ImPlotMarker_Circle, .x = &plot_x_vals, .y = {}, .comp_status = {}});
   plot_elements.push_back({.value_type = PLOT_QUANTITY::HF_E,       .type = PLOT_TYPE::LINE,    .legend = "Half-filling, asymptotic (Lieb & Wu)", .show = false, .enabled = false, .marker_style = ImPlotMarker_Square, .x = &plot_x_vals, .y = {}, .comp_status = {}});
   compute_result = &plot_elements[0].comp_status; // plot_elements should not be resized after this without updating compute_result pointer!
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

template<class... Args>
int plot_work_proc(ProgramState* s, ScalarRange<real> domain, int eidx, bool plot_T_range, real(*E)(const HubbardParams&, Args...), Args... args)
{
   HubbardParams params = s->params;
   std::vector<real>& result = s->plot_elements[eidx].y;
   real& x_param = plot_T_range ? params.T : params.U;
   for(real val : domain)
   {
      x_param = val;
      result.push_back(E(params, args...)/params.Ns);
   }
   return 0;
}

void ProgramState::handle_events()
{
   glfwPollEvents();

   if(in_N_up > in_Ns) { in_N_up = in_Ns; }
   if(in_N_dn > in_Ns) { in_N_dn = in_Ns; }

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

   if(param_input)
   {
      HubbardParams new_params = {in_T, in_U, in_Ns, in_N_up, in_N_dn};
      params_changed = (params != new_params);
      params = new_params;

      if(params_changed)
      {
         params_sz = hubbard_memory_requirements(params);
         ComputeMemoryReqs reqs = cdev.get_memory_requirements(params_sz);
         reqs.total_host_memory_sz += params_sz.workspace_size;

         size_t total_mem_sz = reqs.total_host_memory_sz + reqs.total_device_memory_sz;
         format_memory(params_memory_buf, ARRAY_SIZE(params_memory_buf), total_mem_sz);

         force_clear_profiling_results = true; 
      }
   }

   auto& plotting_range = plot_T_range ? T_range : U_range;

   auto plotting_T_range = T_range;
   auto plotting_U_range = U_range;
   if(plot_T_range)
   {
      plotting_U_range.min = params.U;
      plotting_U_range.max = params.U;
      plotting_U_range.dim = 1;
   }
   else
   {
      plotting_T_range.min = params.T;
      plotting_T_range.max = params.T;
      plotting_T_range.dim = 1;
   }

   if(compute)
   {
      if(plot_mode)
      {
         plot_x_vals.resize(0);
         plot_x_vals.reserve(plotting_range.dim);
         std::copy(plotting_range.begin(), plotting_range.end(), std::back_inserter(plot_x_vals));

         int eidx = 0;
         for(PlotElement& e : plot_elements)
         {
            // TODO: Even if comp_status is pending, might want to let the element through. Reassigning e.comp_result will block until the result is ready, which might be desired behavior
            if(e.enabled && !e.comp_status.is_pending())
            {
               assert(!e.comp_status.is_valid() || e.comp_status.is_ready());
               e.y.resize(0);

               switch(e.value_type)
               {
                  case PLOT_QUANTITY::COMPUTED_E: 
                  {
                     e.y.reserve(plotting_range.dim);
                     e.comp_status = work_queue.push_valcap([](ProgramState* s, ScalarRange<real> plotting_range, int eidx, bool plot_T_range)
                                                            {
                                                               HubbardParams params = s->params;
                                                               std::vector<real>& result = s->plot_elements[eidx].y;
                                                               real& x_param = plot_T_range ? params.T : params.U;
                                                               for(real val : plotting_range)
                                                               {
                                                                  s->compute_start_counter = glfwGetTimerValue();
                                                                  x_param = val;
                                                                  result.push_back(s->model.set_params(params).E0()/real(params.Ns));
                                                                  s->compute_end_counter = glfwGetTimerValue();
                                                                  s->last_compute_elapsed_s = get_s_elapsed(s->compute_start_counter, s->compute_end_counter, s->timer_freq);
                                                               }
                                                               return 0;
                                                            }, _this, plotting_range, eidx, plot_T_range);
                     e.show = true;
                  } break;

                  case PLOT_QUANTITY::DIMER_E: 
                  {
                     e.show = false;
                     if(params.Ns == 2)
                     {
                        assert(!e.comp_status.is_pending());
                        e.y.reserve(plotting_range.dim);

                        if(e.comp_status.is_valid()) { e.comp_status.get<int>(); }
                        e.comp_status = work_queue.push_valcap(plot_work_proc<BCS>, _this, plotting_range, eidx, plot_T_range, 
                                                               dimer_E0, BCS::PERIODIC);
                        e.show = true;
                     }
                  } break;

                  case PLOT_QUANTITY::NONINT_E:
                  {
                     e.y.reserve(plotting_T_range.dim);
                     e.show = false;
                     if((plot_T_range && params.U == 0.0) || (!plot_T_range && plotting_U_range.includes(0.0)))
                     {
                        if(e.comp_status.is_valid()) { e.comp_status.get<int>(); }

                        e.comp_status = work_queue.push_valcap(plot_work_proc<BCS>, _this, plotting_T_range, eidx, true,
                                                               noninteracting_E0, BCS::PERIODIC);
                        if(plot_T_range)
                        {
                           e.x = &plot_x_vals;
                        }
                        else
                        {
                           e.x = &ZERO_FVEC;
                        }
                        e.show = true;
                     }
                  } break;

                  case PLOT_QUANTITY::ATOMIC_E: 
                  {
                     e.y.reserve(plotting_U_range.dim);
                     e.show = false;
                     if((!plot_T_range && params.T == 0.0) || (plot_T_range && plotting_T_range.includes(0.0)))
                     {
                        if(e.comp_status.is_valid()) { e.comp_status.get<int>(); }
                        e.comp_status = work_queue.push_valcap(plot_work_proc<>, _this, plotting_U_range, eidx, false,
                                                               atomic_E0);
                        if(plot_T_range)
                        {
                           e.x = &ZERO_FVEC;
                        }
                        else
                        {
                           e.x = &plot_x_vals;
                        }
                        e.show = true;
                     }
                  } break;

                  case PLOT_QUANTITY::HF_E:
                  {
                     e.show = false;
                     if(is_halffilling(params))
                     {
                        e.y.reserve(plotting_range.dim);

                        if(e.comp_status.is_valid()) { e.comp_status.get<int>(); }
                        e.comp_status = work_queue.push_valcap(plot_work_proc<IntArgs>, _this, plotting_range, eidx, plot_T_range,
                                                               halffilled_E, int_args);
                        e.show = true;
                     }
                  } break;

                  default: { assert(!"Unsupported plot quantity."); }
               }
            }

            eidx++;
         }
      }
      else
      {
         assert(!compute_result->is_valid() || compute_result->is_ready());
         compute_start_counter = glfwGetTimerValue();
         model.set_params(params);
         *compute_result = work_queue.push(model, &HubbardModel::E0); 
      }

      compute = false;
   }

   if(compute_result->is_valid() && compute_result->is_ready())
   {
      if(!plot_mode || force_clear_profiling_results)
      {
         prof_labels.resize(0);
         prof_total.resize(0);
         prof_mean.resize(0);
         prof_min.resize(0);
         prof_max.resize(0);
         prof_count.resize(0);
         prof_y_ticks.resize(0);
         prof_percentage.resize(0);

         force_clear_profiling_results = false;
      }

      if(!plot_mode)
      {
         compute_end_counter = glfwGetTimerValue(); 
         last_compute_elapsed_s = get_s_elapsed(compute_start_counter, compute_end_counter, timer_freq);

         real E0 = compute_result->get<real>();
         sprintf_s(result_E0_buf, "E0: %f", E0);
         sprintf_s(result_compute_time_buf, "Compute time: %llu s", u64(last_compute_elapsed_s));

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
            prof_percentage.push_back(stats.total/(10.0f*last_compute_elapsed_s));
         }
      }

      TimedScope::clear();
      *compute_result = {};
   }

   if(compute_result->is_pending())
   {
      sprintf_s(counter_buf, "%llu s", u64(get_s_to_now(compute_start_counter, timer_freq)));
   }

   params_changed = false;
}

void ProgramState::render_UI()
{
   int display_w, display_h;
   glfwGetFramebufferSize(window, &display_w, &display_h);
   const ImVec2 plot_win_pos       = {0.0f*display_w, 0.0f*display_h};
   const ImVec2 plot_win_size      = {0.8f*display_w, 0.8f*display_h};

   const ImVec2 settings_win_pos   = {0.8f*display_w, 0.0f*display_h};
   const ImVec2 settings_win_size  = {0.2f*display_w - 1, 0.8f*display_h};

   const ImVec2 profiling_win_pos  = {0.0f*display_w, 0.8f*display_h};
   const ImVec2 profiling_win_size = {1.0f*display_w - 1, 0.2f*display_h};

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

      for(const PlotElement& e : plot_elements)
      {
         if(e.show && e.enabled)
         {
            assert(plot_x_vals.size() >= e.y.size());
            ImPlot::SetNextLineStyle(IMPLOT_AUTO_COL, 1.5f);
            ImPlot::SetNextMarkerStyle(e.marker_style);
            switch(e.type)
            {
               case PLOT_TYPE::LINE: { ImPlot::PlotLine(e.legend, e.x->data(), e.y.data(), e.y.size()); } break;
               case PLOT_TYPE::SCATTER: { ImPlot::PlotScatter(e.legend, e.x->data(), e.y.data(), e.y.size()); } break;
               default: { assert(!"Unsupported plot type."); }
            } 
         }
      }

      ImPlot::EndPlot();
   }
   ImPlot::PopStyleVar();
   ImGui::End();

   ImGui::SetNextWindowPos(settings_win_pos);
   ImGui::SetNextWindowSize(settings_win_size);
   ImGui::Begin("Settings", 0, window_flags);

   ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 3.0f);
   ImGui::PushStyleVar(ImGuiStyleVar_Alpha, compute_result->is_pending() ? 0.5 : 1.0);

   if(ImGui::Button("Compute") && !compute_result->is_pending())
   {
      compute = true;
   }

   ImGui::PopStyleVar();

   ImGui::SameLine();
   if(ImGui::Button("Cancel"))
   {
      assert(!"Not implemented!"); // The actual tasks pushed to the work queue don't handle thread termination currently
   }
   ImGui::PopStyleVar();

   ImGui::Text(counter_buf);

   if(ImGui::CollapsingHeader("Parameters"))
   {
      param_input = ImGui::InputFloat("T", &in_T) |
                    ImGui::InputFloat("U", &in_U) |
                    ImGui::SliderInt("Ns", &in_Ns, 1, MAX_SITE_COUNT) |
                    ImGui::SliderInt("N_up", &in_N_up, N_dn_min, N_dn_max) |
                    ImGui::SliderInt("N_dn", &in_N_dn, N_up_min, N_up_max);

      ImGui::PushStyleVar(ImGuiStyleVar_Alpha, plot_T_range ? 1.0 : 0.5);
      int& in_plot_point_count = T_range.dim;
      if(ImGui::InputFloat2("T range", T_range.interval))
      { 
         plot_T_range = true; 
      }
      ImGui::PopStyleVar();

      ImGui::PushStyleVar(ImGuiStyleVar_Alpha, plot_T_range ? 0.5 : 1.0);
      if(ImGui::InputFloat2("U range", U_range.interval))
      { 
         in_plot_point_count = U_range.dim;
         plot_T_range = false;
      }
      ImGui::PopStyleVar();

      ImGui::InputInt("points", &in_plot_point_count);
      in_plot_point_count = std::max(0, in_plot_point_count);

      //ImGui::InputInt3("Ns range", Ns_range);
      //Ns_range[0] = std::clamp(Ns_range[0], 1, MAX_SITE_COUNT);
      //Ns_range[1] = std::clamp(Ns_range[1], Ns_range[0], MAX_SITE_COUNT);

      param_input = param_input | ImGui::Checkbox("Half-filling", &force_halffilling);

      ImGui::Text("Min memory (incl. VRAM): ");
      ImGui::SameLine();
      ImGui::Text(params_memory_buf);
   }

   if(ImGui::CollapsingHeader("Plot"))
   {
      ImGui::MenuItem("E0/Ns", 0, &plot_mode);
   }
   if(ImGui::CollapsingHeader("Benchmark results"))
   {
      for(PlotElement& e : plot_elements)
      {
         ImGui::Checkbox(e.legend, &e.enabled);
      }
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

      if(last_compute_elapsed_s > 0.0)
      {
         for(int i = 0; i < prof_total.size(); i++)
         {
            ImPlot::Annotation(prof_total[i], i + 1, ImVec4(0, 0, 0, 0), ImVec2(0, -5), true, "%d, %.2f%%", prof_count[i], prof_percentage[i]);
         }
      }

      ImPlot::PopStyleVar();
      ImPlot::PopStyleVar();

      ImPlot::EndPlot();
   }
#else
   ImPlot::Text("Use debug build to access profiling.");
#endif

   ImGui::End();

   ImGui::Render();
   glClearColor(clear_color.x*clear_color.w, clear_color.y*clear_color.w, clear_color.z*clear_color.w, clear_color.w);
   glClear(GL_COLOR_BUFFER_BIT);
   ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

   glfwSwapBuffers(window);
}
