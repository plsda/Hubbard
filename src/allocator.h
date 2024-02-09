#ifndef ALLOCATOR_H

namespace allocation
{
   template<auto _Alloc = std::malloc, auto _Free = std::free, auto _Realloc = std::realloc> class ArenaAllocator;
   template<auto _Alloc = std::malloc, auto _Free = std::free, auto _Realloc = std::realloc> struct ArenaCheckpoint;
   template<class T> class StdArenaAllocatorWrapper;
   
   // Prevents the global ArenaCheckpoint sentinel from being destructed
   class GlobalSentinel
   {
   public:
      using Checkpoint = ArenaCheckpoint<>;
   
      GlobalSentinel() = delete;
      template<class T> static T* sentinel() { return (T*)_sentinel; }
      struct StaticCtor { StaticCtor(); };
   
   private:
      friend class StaticCtor;
      static StaticCtor sctor;
      static u8 _sentinel[];
   };
   
   // NOTE: ArenaAllocator is not thread-safe
   template<auto _Alloc, auto _Free, auto _Realloc>
   class ArenaAllocator
   {
   public:
      using Checkpoint = ArenaCheckpoint<_Alloc, _Free, _Realloc>;
      using value_type = u8;
      using pointer = u8*;
      using size_type = size_t;
   
   
      ArenaAllocator() = default;
      ArenaAllocator(size_type size) : total_size(size), remaining_size(size), memory((pointer)_Alloc(size)) { }
      ~ArenaAllocator()
      {
         _Free(memory);
         memory = nullptr;
      }
   
      ArenaAllocator& operator=(const ArenaAllocator&) = delete;
      ArenaAllocator& operator=(ArenaAllocator&&) = delete;

      void clear()
      {
         remaining_size = total_size;
      }
      
      template <class T>
      T* allocate(size_type count, size_type alignment = std::alignment_of_v<T>)
      {
         size_type requested_size = sizeof(T)*count;
      
         assert(total_size >= remaining_size);
         void* result = (void*)(memory + total_size - remaining_size);
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
      void deallocate(T* in_p, size_type count) 
      {
         pointer p = (pointer)in_p;
         assert((p >= memory) && ((p - memory) == sizeof(T)*count));
         deallocate(p);
      }
      
      template <class T>
      void deallocate(T* in_p)
      {
         if(in_p != nullptr)
         {
            pointer p = (pointer)in_p;
            if(memory <= p && p < (memory + total_size))
            {
               remaining_size = p - memory;
            }
            else
            {
               assert(!"Invalid pointer passed to deallocate!");
            }
         }
      }

      template <class T>
      size_type get_aligned_size(size_type count)
      {
         size_type requested_size = sizeof(T)*count;
         void* ptr = (void*)(memory + total_size - remaining_size);
         size_type size1 = std::numeric_limits<size_type>::max();
         size_type size2 = size1;
         std::align(std::alignment_of_v<T>, requested_size, ptr, size1);
      
         return requested_size + (size2 - size1);
      }
      
      // TODO: Finish implementing these (for reserving an amount of memory for a checkpoint when don't know size beforehand but can point the beginning and end of allocations)
      Checkpoint begin_provision()
      {
         return Checkpoint(*this);
      }
      void end_provision(Checkpoint& cpt)
      {
         cpt.size = used_size() - cpt.used_size;
      }
      
      void reallocate(size_type new_size, bool clear = false)
      {
         // NOTE: Reallocation invalidates all previously obtained pointers to this arena. Here, reallocation mostly makes sense if the arena is empty 
         //       or consists of only one "block" of (scalar) values (or want to clear the arena anyways). Although malloc should automatically return
         //       an address with suitable alignment for any scalar type, reallocation doesn't necessarily preserve the alignment of all previous allocations.
         
         assert(clear || used_size() == 0);

         size_type used_sz = 0;
         if(new_size == 0)
         {
            _Free(memory);
            memory = nullptr;
         }
         else
         {
            if(clear)
            {
               _Free(memory);
               memory = (u8*)_Alloc(new_size);
               if(!memory)
               {
                  new_size = 0;
                  assert(!"Reallocation failed.");
               }
            }
            else
            {
               used_sz = std::min(used_size(), new_size);
               void* new_memory = (pointer)_Realloc((void*)memory, new_size); // NOTE: This copies even the unused portion of the old memory

               void* temp = new_memory;
               assert(std::align(alignof(std::max_align_t), new_size, temp, new_size));
               assert(temp == new_memory);

               if(!new_memory)
               {
                  _Free(memory);
                  new_size = 0;
                  used_sz = 0;
                  assert(!"Reallocation failed.");
               }

               memory = (u8*)new_memory;
            }
         }

         total_size = new_size;
         remaining_size = total_size - used_sz;
      }
      void reserve(size_type required_remaining, bool clear = false)
      {
         if(remaining_size < required_remaining)
         {
            reallocate(used_size() + required_remaining, clear);
         }
      }
   
      size_type max_size() const { return total_size; };
      size_type used_size() const { return total_size - remaining_size; };
      size_type unused_size() const { return remaining_size; };
    
      bool operator==(const ArenaAllocator<_Alloc, _Free>& other) const noexcept { return (memory == other.memory); }
   
   private:
      friend class Checkpoint;
      size_type total_size;
      size_type remaining_size;
      value_type* memory;
      Checkpoint* last_checkpoint = GlobalSentinel::sentinel<Checkpoint>();
   };
   
   template<auto _Alloc, auto _Free, auto _Realloc>
   struct ArenaCheckpoint
   {
      using Allocator = ArenaAllocator<_Alloc, _Free, _Realloc>;
      using size_type = typename Allocator::size_type;
      
      ArenaCheckpoint() : used_size(0), size(0), alloc(0), prev(0), next(0) { }
      ArenaCheckpoint(Allocator& _alloc, size_type _size = 0) : alloc(&_alloc), size(_size)
      {
         used_size = alloc->used_size();
         prev = alloc->last_checkpoint;
         prev->next = this;
         next = 0;
         alloc->last_checkpoint = this;
      }
      
      ~ArenaCheckpoint()
      {
         if(alloc) // NOTE: If alloc is zeroed, the checkpoint was most likely already freed so ignore the checkpoint. 
                   //       However, might want to do something else here (warning/error).
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
               // Merge with the next checkpoint so that restoring it will also restore this checkpoint.
               next->used_size = used_size;
               next->size += size;
      
               next->prev = prev;
               prev->next = next;
            }
         }
      }
   
      size_type used_size;
      // TODO: 'size' is currently unused
      size_type size; // Optionally store size for the checkpoint, so that can separate allocations from each other in the case that
                      // don't create/forgot to create a checkpoint for allocations after done allocating memory for an unrelated task
      Allocator* alloc;
      ArenaCheckpoint* prev;
      ArenaCheckpoint* next; 
   };
   
   template<class T>
   struct StdArenaAllocatorWrapper
   {
#define _Alloc std::malloc
#define _Free std::free
#define _Realloc std::realloc
      using Allocator = ArenaAllocator<_Alloc, _Free, _Realloc>;
      using value_type = T;
      using pointer = T*;
      using size_type = typename Allocator::size_type;

      StdArenaAllocatorWrapper() : allocator(0), max_size(0), allocated_size(0), allocated_count(0) { }
   
      template<class U>
      StdArenaAllocatorWrapper(const StdArenaAllocatorWrapper<U>& other) noexcept
      {
         allocator = other.allocator;
         // NOTE: These will be out of sync with the values of the other allocator after construction and may not be useful any longer
         //       If the other allocator allocates more memory afterwards, this allocator will have wrong size limit etc
         max_size = other.max_size - other.allocated_size;
         allocated_size = 0;
         allocated_count = 0;
      } 
   
      StdArenaAllocatorWrapper(Allocator* _allocator, size_t max_count = 0, size_t pad_bytes = std::numeric_limits<size_t>::max()) :
         allocator(_allocator), max_size(sizeof(T)*max_count + pad_bytes), allocated_size(0), allocated_count(0) { }
   
      pointer allocate(size_type count)
      {
         size_type requested_size = allocator->template get_aligned_size<T>(count);
         assert((allocated_size + requested_size) <= max_size); 
         T* result = allocator->template allocate<T>(count);
   
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
   
      Allocator* allocator;
      size_type max_size;
      size_type allocated_size;
      size_type allocated_count;
#undef _Alloc
#undef _Free
#undef _Realloc
   };
   
};

using ArenaAllocator  = allocation::ArenaAllocator<>;
using ArenaCheckpoint = allocation::ArenaCheckpoint<>;

using DetArena  = allocation::StdArenaAllocatorWrapper<Det>;
using IntArena  = allocation::StdArenaAllocatorWrapper<int>;
using RealArena = allocation::StdArenaAllocatorWrapper<real>;
#if(__cplusplus >= 202002L)
using SpanArena = allocation::StdArenaAllocatorWrapper<std::span<Det>>;
#endif


#define ALLOCATOR_H
#endif
