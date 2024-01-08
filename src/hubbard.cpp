#include "hubbard.h"
#include "utils.cpp"
#include "basis.cpp"
#include "solver.cpp"

#include "profiler.h"

int main()
{
   Error_stream errors;
   Hubbard_compute_device compute_device(&errors);
   if(errors.has_errors)
   {
      std::cout << errors;
      return -1;
   }

   real T = 1;
   real U = real(2.7);
   u32 Ns = 5;
   u32 N_up = 2;
   u32 N_down = 2;

   HubbardParams params(T, U, Ns, N_up, N_down);

   KConfigs configs = get_k_orbitals(params);

   real E0 = kfm_basis_compute_E0(compute_device, configs, params);

   std::cout << "N = " << N_up + N_down
             << ", Ns = " << Ns
             << ", T = " << T
             << ", U = " << U
             << ", N_up = " << N_up
             << ", N_down = " << N_down << std::endl;
   std::cout << "E0/Ns = " << E0/Ns << std::endl;

   return 0;
}
