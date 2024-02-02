
void ProfilingStats::accumulate(timedt pt)
{
   float t = std::chrono::duration<float, std::milli>(pt).count();
   if(t < min)
   { 
      min = t; 
   }
   else if(t > max)
   { 
      max = t; 
   }

   total += t;
   count++;
}

TimedScope::TimedScope(std::string_view ID)
{
   this->ID = ID;
   start = std::chrono::steady_clock::now();
}

TimedScope::~TimedScope()
{
   timept end = std::chrono::steady_clock::now();
   timedt elapsed(end - start);

   trace_stats[ID].accumulate(elapsed);
}
