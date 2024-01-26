
alignas(ArenaCheckpoint) u8 GlobalSentinel::_sentinel[sizeof(ArenaCheckpoint)];

GlobalSentinel::StaticCtor::StaticCtor()
{ 
   new (&GlobalSentinel::_sentinel) ArenaCheckpoint; 
}

ArenaCheckpoint* GlobalSentinel::sentinel()
{ 
   return (ArenaCheckpoint*)_sentinel; 
}


ArenaAllocator::ArenaAllocator(size_type size) : total_size(size), remaining_size(size)
{
   memory = std::make_shared<value_type[]>(size);
}

void ArenaAllocator::clear()
{
   remaining_size = total_size;
}

// Remove the checkpoint, but don't necessarily change allocator state/release memory
void clear_checkpoint(ArenaCheckpoint& cpt) 
{
   cpt.~ArenaCheckpoint();
}

template <class T>
T* ArenaAllocator::allocate(size_type count, size_type alignment)
{
   size_type requested_size = sizeof(T)*count;

   assert(total_size >= remaining_size);
   void* result = (void*)(memory.get() + total_size - remaining_size);
   if(std::align(alignment, requested_size, result, remaining_size))
   {
      remaining_size -= requested_size;
   }
   else
   {
      assert(!"Not enough memory in arena allocator!");
      result = nullptr;
   }

   return (T*)result;
}

// Pointer arithmetic below may lead to undefined behavior if in_p is not a pointer obtained from this allocator.
template <class T>
void ArenaAllocator::deallocate(T* in_p, size_type count) 
{
   pointer p = (pointer)in_p;
   assert((p >= memory.get()) && ((p - memory.get()) == sizeof(T)*count));
   deallocate(p);
}

template <class T>
void ArenaAllocator::deallocate(T* in_p)
{
   if(in_p != nullptr)
   {
      pointer p = (pointer)in_p;
      if(memory.get() <= p && p <= (memory.get() + total_size))
      {
         remaining_size = p - memory.get();
      }
      else
      {
         assert(!"Invalid pointer passed to deallocate!");
      }
   }
}

template <class T>
ArenaAllocator::size_type ArenaAllocator::get_aligned_size(size_type count)
{
   size_type requested_size = sizeof(T)*count;
   void* ptr = (void*)(memory.get() + total_size - remaining_size);
   size_type size1 = std::numeric_limits<size_type>::max();
   size_type size2 = size1;
   std::align(std::alignment_of_v<T>, requested_size, ptr, size1);

   return requested_size + (size2 - size1);
}

// TODO: Finish implementing these (for reserving an amount of memory for a checkpoint when don't know size beforehand but can point the beginning and end of allocations)
ArenaCheckpoint ArenaAllocator::begin_provision()
{
   return ArenaCheckpoint(*this);
}
void ArenaAllocator::end_provision(ArenaCheckpoint& cpt)
{
   cpt.size = used_size() - cpt.used_size;
}


ArenaCheckpoint::ArenaCheckpoint(ArenaAllocator& _alloc, size_type _size) : alloc(&_alloc), size(_size)
{
   used_size = alloc->used_size();
   prev = alloc->last_checkpoint;
   prev->next = this;
   next = 0;
   alloc->last_checkpoint = this;
}

ArenaCheckpoint::~ArenaCheckpoint()
{
   if(alloc) // NOTE: Checking if alloc is zeroed and if it is, the checkpoint was most likely already freed so ignoring the checkpoint. 
             //       However, might want to do something else here (warning/error)
   {
      if(alloc->last_checkpoint == this)
      {
         assert((next == 0) && used_size <= alloc->used_size());

         alloc->remaining_size = alloc->total_size - used_size;
         prev->next = 0;
         alloc->last_checkpoint = prev;
         alloc = 0;
         prev = 0;
      }
      else
      {
         assert(next->used_size >= used_size);
         // Merge with the next checkpoint so that restoring it will also restore this checkpoint
         next->used_size = used_size;
         next->size += size;

         next->prev = prev;
         prev->next = next;
      }
   }
}
