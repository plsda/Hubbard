#ifndef ALLOCATOR_H

class ArenaAllocator;
template <class T> class StdArenaAllocatorWrapper;

using DetArena  = StdArenaAllocatorWrapper<Det>;
using IntArena  = StdArenaAllocatorWrapper<int>;
using RealArena = StdArenaAllocatorWrapper<real>;
#if(__cplusplus >= 202002L)
using SpanArena = StdArenaAllocatorWrapper<std::span<Det>>;
#endif

class ArenaAllocator;
struct ArenaCheckpoint;
template <class T> class StdArenaAllocatorWrapper;

// Prevents the global ArenaCheckpoint sentinel from being destructed
class GlobalSentinel
{
public:
   GlobalSentinel() = delete;
   static ArenaCheckpoint* sentinel();
   struct StaticCtor { StaticCtor(); };

private:
   friend class StaticCtor;
   static StaticCtor sctor;
   static u8 _sentinel[];
};

// NOTE: ArenaAllocator is not thread-safe
class ArenaAllocator
{
public:
   using value_type = u8;
   using pointer = u8*;
   using size_type = size_t;

   ArenaAllocator() = default;
   ArenaAllocator(size_type size);

   void clear();
   //void reallocate(size_type new_size);
   template <class T> T* allocate(size_type count, size_type alignment = std::alignment_of_v<T>);
   // NOTE: Explicit temporary memory blocks derived from the arena allocator is probably better than using these deallocation functions.
   template <class T> void deallocate(T* in_p, size_type count); // in_p has to be a pointer obtained from allocate()
   template <class T> void deallocate(T* in_p);

   template <class T> size_type get_aligned_size(size_type count);
   size_type max_size() const { return total_size; };
   size_type used_size() const { return total_size - remaining_size; };
   size_type unused_size() const { return remaining_size; };

   ArenaCheckpoint begin_provision();
   void end_provision(ArenaCheckpoint& cpt);

   bool operator==(const ArenaAllocator& other) const noexcept { return (memory == other.memory); }

private:
   friend class ArenaCheckpoint;
   size_type total_size;
   size_type remaining_size;
   std::shared_ptr<value_type[]> memory;
   ArenaCheckpoint* last_checkpoint = GlobalSentinel::sentinel();
};

struct ArenaCheckpoint
{
   using size_type = ArenaAllocator::size_type;
   
   size_type used_size;
   // TODO: 'size' is currently unused
   size_type size; // Optionally store size for the checkpoint, so that can separate allocations from each other in the case that
                   // don't create/forgot to create a checkpoint for allocations after done allocating memory for an unrelated task
   ArenaAllocator* alloc;
   ArenaCheckpoint* prev;
   ArenaCheckpoint* next; 

   ArenaCheckpoint() : used_size(0), size(0), alloc(0), prev(0), next(0) { }
   ArenaCheckpoint(ArenaAllocator& _alloc, size_type _size = 0);
   ~ArenaCheckpoint();
};

template <class T>
struct StdArenaAllocatorWrapper
{
   using value_type = T;
   using pointer = T*;
   using size_type = ArenaAllocator::size_type;

   ArenaAllocator* allocator;
   size_type max_size;
   size_type allocated_size;
   size_type allocated_count;

   StdArenaAllocatorWrapper() : allocator(0), max_size(0), allocated_size(0), allocated_count(0) { }

   template <class U>
   StdArenaAllocatorWrapper(const StdArenaAllocatorWrapper<U>& other) noexcept
   {
      allocator = other.allocator;
      // NOTE: These will be out of sync with the values of the other allocator after construction and may not be useful any longer.
      //       If the other allocator allocates more memory afterwards, this allocator will have wrong size limit etc.
      max_size = other.max_size - other.allocated_size;
      allocated_size = 0;
      allocated_count = 0;
   } 

   StdArenaAllocatorWrapper(ArenaAllocator* _allocator, size_t max_count = 0, size_t pad_bytes = std::numeric_limits<size_t>::max()) :
      allocator(_allocator), max_size(sizeof(T)*max_count + pad_bytes), allocated_size(0), allocated_count(0) { }

   pointer allocate(size_type count)
   {
      size_type requested_size = allocator->get_aligned_size<T>(count);
      assert((allocated_size + requested_size) <= max_size); 
      T* result = allocator->allocate<T>(count);

      allocated_size += requested_size;
      allocated_count += count;

      return result;
   }
   void deallocate(pointer p, size_type count)
   {
      // Do nothing
   }

   bool operator==(const StdArenaAllocatorWrapper& other) const noexcept { return (*allocator == *other.allocator); }
   bool operator!=(const StdArenaAllocatorWrapper& other) const noexcept { return !(*this == other); }
};

#define ALLOCATOR_H
#endif
