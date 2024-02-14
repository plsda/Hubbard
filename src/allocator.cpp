
namespace allocation
{
   void* nop_alloc(size_t) { return (void*)1; }
   void  nop_free(void*) { }
   void* nop_realloc(void*, size_t) { return (void*)1; }
   void* forbidden_realloc(void*, size_t) { assert(!"Unwanted realloc!"); return (void*)0; }

   alignas(GlobalSentinel::Checkpoint) u8 GlobalSentinel::_sentinel[sizeof(GlobalSentinel::Checkpoint)];
   
   GlobalSentinel::StaticCtor::StaticCtor()
   { 
      new(&GlobalSentinel::_sentinel) Checkpoint; 
   }
};
