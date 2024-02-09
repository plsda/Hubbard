
namespace allocation
{
   alignas(GlobalSentinel::Checkpoint) u8 GlobalSentinel::_sentinel[sizeof(GlobalSentinel::Checkpoint)];
   
   GlobalSentinel::StaticCtor::StaticCtor()
   { 
      new(&GlobalSentinel::_sentinel) Checkpoint; 
   }
};
