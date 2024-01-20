
ArenaAllocator::ArenaAllocator(size_type size) : total_size(size), remaining_size(size)
{
   memory = std::make_shared<value_type[]>(size);
}

void ArenaAllocator::clear()
{
   remaining_size = total_size;
}

template <class T>
T* ArenaAllocator::allocate(size_type count)
{
   size_type requested_size = sizeof(T)*count;

   assert(total_size >= remaining_size);
   void* result = (void*)(memory.get() + total_size - remaining_size);
   if(std::align(std::alignment_of_v<T>, requested_size, result, remaining_size))
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
   pointer p = (value_type)in_p;
   assert((p >= memory) && ((p - memory.get()) == sizeof(T)*count));
   deallocate(p);
}

template <class T>
void ArenaAllocator::deallocate(T* in_p)
{
   if(in_p != nullptr)
   {
      pointer p = (value_type)in_p;
      if(memory <= p && p <= (memory.get() + total_size))
      {
         remaining_size = ((pointer)p) - memory.get();
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
