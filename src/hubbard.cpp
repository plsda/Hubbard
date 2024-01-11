#include "hubbard.h"
#include "utils.cpp"
#include "basis.cpp"
#include "solver.cpp"

#include <algorithm>
#include <thread>
#include <future>
#include <chrono>
using namespace std::chrono_literals;

const int GUI_FRAMERATE = 30;
const double MS_PER_FRAME = 1000.0/double(GUI_FRAMERATE);

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

#ifdef _WIN32
#include <Windows.h>
#include <timeapi.h>
#else
#include <unistd.h>
#endif

void sleep(double ms)
{
#ifdef _WIN32
    Sleep(DWORD(ms));
#else
    usleep(useconds_t(ms*1000.0));
#endif
}

bool check_future(std::future<real>& f)
{
   // NOTE: Timeout not checked
   return (f.wait_for(0s) == std::future_status::ready);
}

bool is_halffilling(const HubbardParams& params)
{
   return ((params.Ns == 2*params.N_up) && (params.Ns == 2*params.N_down));
}

int main()
{
#ifdef _WIN32
   bool sleepable = (timeBeginPeriod(1) == TIMERR_NOERROR);
#endif

   float in_T = 1.0f;
   float in_U = 1.0f;
   int in_Ns = 5;
   int in_N_up = 2;
   int in_N_dn = 2;
   HubbardParams params(in_T, in_U, in_Ns, in_N_up, in_N_dn);
   IntArgs int_args = {.lower = 1e-8, .upper = 100, .abs_tol = 1e-8, .rel_tol = 1e-6, .min_steps = 2, .max_steps = 60};

   ErrorStream errors;
   HubbardComputeDevice compute_device(&errors);
   if(errors.has_errors)
   {
      std::cerr << errors;
      return -1;
   }

   float x_step = 0.1;
   float T_range[3] = {0, 1, x_step};
   float U_range[3] = {0, 1, x_step};
   int Ns_range[3] = {1, 1, 1};
   bool force_halffilling = false;

   bool computing = false;
   char counter_buf[256];
   char result_E0_buf[256];
   char result_compute_time_buf[256];
   memcpy(counter_buf, "0 s", 4);
   sprintf_s(result_E0_buf, "E0: ");
   sprintf_s(result_compute_time_buf, "Compute time: ");

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

   u64 last_compute_elapsed_s = 0;
   std::jthread compute_thread;
   std::future<real> compute_result;

   const int MAX_PLOT_PTS = 500;
   std::vector<real> plot_E_vals;
   std::vector<real> plot_x_vals;

   std::vector<real> halffilling_E_vals;
   std::vector<real> dimer_E_vals;
   std::vector<real> nonint_E_vals;
   std::vector<real> atomic_E_vals;

   plot_E_vals.reserve(MAX_PLOT_PTS);
   plot_x_vals.reserve(MAX_PLOT_PTS);
   halffilling_E_vals.reserve(MAX_PLOT_PTS);
   dimer_E_vals.reserve(MAX_PLOT_PTS);
   nonint_E_vals.reserve(MAX_PLOT_PTS);
   atomic_E_vals.reserve(MAX_PLOT_PTS);

   glfwSetErrorCallback(glfw_error_callback);
   if(!glfwInit())
   {
      return -1;
   }

   glfwWindowHint(GLFW_DOUBLEBUFFER, 1);
   const char* glsl_version = "#version 130";
   glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
   glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

   GLFWwindow* window = glfwCreateWindow(1280, 720, "Hubbard", NULL, NULL);
   if(!window)
   {
      return -1;
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

   bool show_demo_window = true;
   bool show_another_window = false;
   ImVec4 clear_color = ImVec4(0.0f, 0.0f, 0.0f, 1.0f);


   u64 timer_freq = glfwGetTimerFrequency();
   u64 end_counter = 0;
   u64 last_counter = glfwGetTimerValue();

   ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoResize |
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
   while (!glfwWindowShouldClose(window))
   {
      glfwPollEvents();

      if(computing && check_future(compute_result))
      {
         compute_end_counter = glfwGetTimerValue();
         last_compute_elapsed_s = get_s_elapsed(compute_start_counter, compute_end_counter, timer_freq);
         computing = false;
         real E0 = compute_result.get();
         sprintf_s(result_E0_buf, "E0: %f", E0);
         sprintf_s(result_compute_time_buf, "Compute time: %llu s", last_compute_elapsed_s);

         if(plot_mode)
         {
            plot_x_vals.push_back(plot_T_range ? params.T : params.U);
            plot_E_vals.push_back(E0/params.Ns);
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

      // TODO: Disable input (except "Cancel") when computing
      ImGui::SetNextWindowPos(plot_win_pos);
      ImGui::SetNextWindowSize(plot_win_size);
      ImGui::Begin("Plotting", 0, window_flags);
      ImPlot::SetNextAxesToFit();

      ImPlot::PushStyleVar(ImPlotStyleVar_FitPadding, ImVec2(0.1f, 0.1f));
      if(plot_mode && ImPlot::BeginPlot("##plot1", ImVec2(-1, -1)))
      {
         ImPlot::SetupAxes(plot_T_range ? "T" : "U", "E0/Ns");
         ImPlot::SetNextLineStyle(IMPLOT_AUTO_COL, 1.5f);
         ImPlot::SetNextMarkerStyle(ImPlotMarker_Square);

         ImPlot::PlotLine("##line1", plot_x_vals.data(), plot_E_vals.data(), plot_E_vals.size());

         if(show_halffilling && is_halffilling(params))
         {
            assert(plot_x_vals.size() == halffilling_E_vals.size());
            ImPlot::PlotLine("Lieb & Wu", plot_x_vals.data(), halffilling_E_vals.data(), halffilling_E_vals.size());
         }
         if(show_noninteracting)
         {
            ImPlot::SetNextLineStyle(IMPLOT_AUTO_COL, 1.5f);
            ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 5.0f, ImVec4(0, 0, 0, 0), IMPLOT_AUTO);
            real nonint_x_vals[] = {0};
            real nonint_E_vals[] = {noninteracting_E0(params, BCS::PERIODIC)/params.Ns};
            ImPlot::PlotScatter("U = 0, ground truth", nonint_x_vals, nonint_E_vals, 1);
         }
         if(show_atomic_limit)
         {
            ImPlot::SetNextLineStyle(IMPLOT_AUTO_COL, 1.5f);
            ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 5.0f, ImVec4(0, 0, 0, 0), IMPLOT_AUTO);
            real atomic_x_vals[] = {0};
            real atomic_E_vals[] = {atomic_E0(params)/params.Ns};
            ImPlot::PlotScatter("T = 0, ground truth", atomic_x_vals, atomic_E_vals, 1);
         }
         if(show_dimer)
         {
            assert(plot_x_vals.size() == dimer_E_vals.size());
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
            }
            else
            {
               real U_min = U_range[0];
               params.U = U_min; 
               in_U = U_min;
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
         computing = true;
         compute_start_counter = glfwGetTimerValue();

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
            }
         }


         std::packaged_task<real(HubbardComputeDevice&, HubbardParams)> compute_task(&__kfm_basis_compute_E0);
         compute_result = compute_task.get_future();

         compute_thread = std::jthread(std::move(compute_task), std::ref(compute_device), params);
      }
      ImGui::PopStyleVar();

      ImGui::SameLine();
      if(ImGui::Button("Cancel"))
      {
         computing = false;
         compute_end_counter = glfwGetTimerValue();
         compute_thread.request_stop(); 
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

         params = {in_T, in_U, in_Ns, in_N_up, in_N_dn};

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
      ImGui::End();


      ImGui::Render();
      glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w);
      glClear(GL_COLOR_BUFFER_BIT);
      ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

      glfwSwapBuffers(window);

      uint64_t end_counter = glfwGetTimerValue();
      double elapsed = get_ms_elapsed(last_counter, end_counter, timer_freq);
      if(elapsed < MS_PER_FRAME && sleepable)
      {
         sleep(MS_PER_FRAME - elapsed);
      }

      std::cout << "FPS: " << 1.0/get_ms_to_now(last_counter, timer_freq) * 1000.0 << std::endl;

      last_counter = glfwGetTimerValue();
   }
   ImGui::PopStyleColor();
   ImGui::PopStyleColor();

   ImGui_ImplOpenGL3_Shutdown();
   ImGui_ImplGlfw_Shutdown();
   ImPlot::DestroyContext();
   ImGui::DestroyContext();
   glfwDestroyWindow(window);
   glfwTerminate();

   return 0;
}
