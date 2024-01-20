#ifndef ALLOCATOR_H

class ArenaAllocator
{
public:
   using value_type = u8;
   using pointer = u8*;
   using size_type = size_t;

   ArenaAllocator(size_type size);
   void clear();
   template <class T> T* allocate(size_type count);
   // NOTE: Explicit temporary memory blocks derived from the arena allocator might be better than using these deallocation functions.
   template <class T> void deallocate(T* in_p, size_type count); // in_p has to be a pointer obtained from allocate()
   template <class T> void deallocate(T* in_p);
   template <class T> size_type get_aligned_size(size_type count);
   size_type max_size() const { return total_size; };
   size_type used_size() const { return total_size - remaining_size; };

   /*
   ArenaScope begin()
   {
   }
   void end()
   {
   }
   class ArenaScope
   {
   };
   */
private:
   size_type total_size;
   size_type remaining_size;
   std::shared_ptr<value_type[]> memory;
};

template <class T>
class StdArenaAllocatorWrapper
{
public:
   using value_type = T;
   using pointer = T*;
   using size_type = ArenaAllocator::size_type;

   StdArenaAllocatorWrapper(ArenaAllocator& _allocator, size_t _max_size = std::numeric_limits<size_t>::max()) :
      allocator(_allocator), max_size(_max_size), allocated_size(0), allocated_count(0) {}

   pointer allocate(size_type count)
   {
      size_type requested_size = allocator.get_aligned_size<T>(count);
      assert((allocated_size + requested_size) <= max_size); 
      T* result = allocator.allocate<T>(count);

      allocated_size += requested_size;
      allocated_count += count;

      return result;
   }

   void deallocate(pointer p, size_type count)
   {
      assert((count == allocated_count) && "std container doing partial deallocation on arena allocator");
      allocator.deallocate(p, count);
      allocated_size = 0;
      allocated_count = 0;
   }
private:
   ArenaAllocator& allocator;
   size_type max_size;
   size_type allocated_size;
   size_type allocated_count;
};

#define ALLOCATOR_H
#endif
