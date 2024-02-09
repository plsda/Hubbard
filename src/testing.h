#ifndef TESTING_H

#define HUBBARD_TEST

#include <span>

#include "common.h"
#include "allocator.h"
#include "utils.h"
#include "basis.h"
#include "compute.h"
#include "solver.h"
#include "profiler.h"

#define TEST_E_TOL real(1e-4)
const size_t TEST_ARENA_SIZE = 150*1024*1024;
const size_t TEST_COMP_ARENA_SIZE = 100*1024*1024;

class HubbardEnvironment : public ::testing::Environment
{
public:
   HubbardEnvironment() :
      errors(),
      cdev(TEST_COMP_ARENA_SIZE, TEST_COMP_ARENA_SIZE, &errors),
      allocator(TEST_ARENA_SIZE)
   {

   }

   void SetUp() override 
   {
      ASSERT_FALSE(errors.has_errors) << errors;
   }

   ErrorStream errors;
   HubbardComputeDevice cdev;
   ArenaAllocator allocator;
};
static HubbardEnvironment* const global_test_env = static_cast<HubbardEnvironment*>(::testing::AddGlobalTestEnvironment(new HubbardEnvironment));
//static HubbardEnvironment* global_test_env;

template <StructuralHubbardParams P>       
class KSBasisTest : public testing::TestWithParam<HubbardParams>
{
public:
   static void SetUpTestSuite();
   static void TearDownTestSuite();

   static inline const HubbardParams params = P;

protected:
   static inline KSBlockIterator itr = KSBlockIterator(P, global_test_env->allocator, P.KS_block_count()*hubbard_memory_requirements(P));
};

class KBasisTest : public testing::TestWithParam<HubbardParams> {};

class HIntTest : public testing::TestWithParam<HubbardParams>
{
public:
   HIntTest() :
      itr(GetParam(), global_test_env->allocator, GetParam().KS_block_count()*hubbard_memory_requirements(GetParam())),
      cdev(hubbard_memory_requirements(GetParam()))
   {

   }

protected:
   void SetUp() override;

   KSBlockIterator itr;
   HubbardComputeDevice cdev;
};

void __set_up_KS_configs(KSBlockIterator& itr, const HubbardParams& params);

void __test_basis_K_and_configs(KSBlockIterator& itr, const std::vector<std::span<Det>, SpanArena>& KS_configs, const std::vector<int, IntArena>& KS_S_path_counts,
                                const std::vector<real, RealArena>& KS_spins, std::vector<real, RealArena>& CSFs, const HubbardParams& params);

void __test_CSF_orthonormality(KSBlockIterator& itr, const std::vector<std::span<Det>, SpanArena>& KS_configs, const std::vector<int, IntArena>& KS_S_path_counts,
                               const std::vector<real, RealArena>& KS_spins, std::vector<real, RealArena>& CSFs, const HubbardParams& params);

void __test_CSF_spins(KSBlockIterator& itr, const std::vector<std::span<Det>, SpanArena>& KS_configs, const std::vector<int, IntArena>& KS_S_path_counts,
                      const std::vector<real, RealArena>& KS_spins, std::vector<real, RealArena>& CSFs, const HubbardParams& params);

#define KS_BASIS_TEST___(suite_name, T, U, Ns, N_up, N_dn)\
   using suite_name = KSBasisTest<StructuralHubbardParams{(real)T, (real)U, Ns, N_up, N_dn}>;\
   TEST_P(suite_name, test_basis_K_and_configs) { ASSERT_NO_FATAL_FAILURE(__test_basis_K_and_configs(itr, itr.KS_configs, itr.KS_S_path_counts, itr.KS_spins, itr.KS_CSF_coeffs, params)); }\
   TEST_P(suite_name, test_CSF_orthonormality)  { ASSERT_NO_FATAL_FAILURE(__test_CSF_orthonormality( itr, itr.KS_configs, itr.KS_S_path_counts, itr.KS_spins, itr.KS_CSF_coeffs, params)); }\
   TEST_P(suite_name, test_CSF_spins)           { ASSERT_NO_FATAL_FAILURE(__test_CSF_spins(          itr, itr.KS_configs, itr.KS_S_path_counts, itr.KS_spins, itr.KS_CSF_coeffs, params)); }\
   INSTANTIATE_TEST_SUITE_P(suite_name##_inst, suite_name, testing::Values(suite_name::params));

#define KS_BASIS_TEST__(count, ...) EXPAND(KS_BASIS_TEST___(KSBasisTest_##count, __VA_ARGS__))
#define KS_BASIS_TEST_(...) EXPAND(KS_BASIS_TEST__(__VA_ARGS__))
#define KS_BASIS_TEST(...) EXPAND(KS_BASIS_TEST_(__COUNTER__, __VA_ARGS__))

#define TESTING_H
#endif
