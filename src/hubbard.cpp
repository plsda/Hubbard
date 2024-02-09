#include "hubbard.h"
#include "utils.cpp"
#include "basis.cpp"
#include "allocator.cpp"
#include "solver.cpp"
#include "profiler.cpp"
#include "ui.cpp"

const int GUI_FRAMERATE = 30;
const double MS_PER_FRAME = 1000.0/GUI_FRAMERATE;

int main()
{
#ifdef _WIN32
   bool sleepable = (timeBeginPeriod(1) == TIMERR_NOERROR);
#endif

   ErrorStream errors;
   size_t host_workspace_init_size = 50*1024*1024;
   size_t device_workspace_init_size = 50*1024*1024;
   HubbardComputeDevice cdev(host_workspace_init_size, device_workspace_init_size, &errors);
   if(errors.has_errors)
   {
      std::cerr << errors;
      return -1;
   }

   ArenaAllocator allocator(1*1024*1024);

   ProgramState state("Hubbard", 1280, 720, allocator, cdev, errors); // NOTE: Also initializes glfw
   if(errors.has_errors)
   {
      std::cerr << errors;
      return -1;
   }
   u64 timer_freq = glfwGetTimerFrequency();
   u64 end_counter = 0;
   u64 last_counter = glfwGetTimerValue();

   while(state.is_running())
   {
      state.handle_events();
      state.render_UI();

      uint64_t end_counter = glfwGetTimerValue();
      double elapsed = get_ms_elapsed(last_counter, end_counter, timer_freq);
      if(elapsed < MS_PER_FRAME && sleepable)
      {
         sleep(MS_PER_FRAME - elapsed);
      }

      std::cout << "FPS: " << 1.0/get_ms_to_now(last_counter, timer_freq) * 1000.0 << std::endl;
      last_counter = glfwGetTimerValue();
   }

   return 0;
}
