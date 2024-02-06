#ifndef PROFILER_H

/* A simple, but intrusive, profiling system */

using timept = std::chrono::time_point<std::chrono::steady_clock>;
using timedt = std::chrono::duration<double>;
using time_ms = std::chrono::milliseconds;
using time_s = std::chrono::seconds;
using record_idx_type = u8;
const size_t MAX_RECORD_COUNT = std::numeric_limits<record_idx_type>::max() + 1;
const size_t TRACE_COUNT_INIT = 64;

struct ProfilingStats
{
   void accumulate(timedt pt);
   float mean() const { return total/float(count); }

   float min = std::numeric_limits<float>::max();
   float max = 0;
   float total = 0;
   int count = 0;
};

class TimedScope
{
public:
   TimedScope(std::string_view ID);
   ~TimedScope();

   static inline std::unordered_map<std::string_view, ProfilingStats> trace_stats{TRACE_COUNT_INIT};

   static ProfilingStats& get_stats(std::string_view ID)
   {
      return trace_stats[ID];
   }
   static void clear()
   {
      trace_stats.clear();
   }
   static int trace_count()
   {
      return trace_stats.size();
   }

private:
   std::string_view ID;
   timept start;
};

#if(defined(HUBBARD_DEBUG) && !defined(HUBBARD_TEST))
#define TIMED_SCOPE_ID__(count, func, line) func "__" #line "__" #count
#define TIMED_SCOPE_ID(count, func, line) TIMED_SCOPE_ID__(count, func, line)
#define TIME_SCOPE_(suffix, ID) TimedScope timed_##suffix(ID);
#define TIME_SCOPE_AUTO() TIME_SCOPE_(__COUNTER__, TIMED_SCOPE_ID(__COUNTER__, __FUNCTION__, __LINE__))

#define TIME_SCOPE(name) TIME_SCOPE_(__COUNTER__, name)
#else
#define TIME_SCOPE_AUTO()
#define TIME_SCOPE(name)
#endif

#define PROFILER_H
#endif
