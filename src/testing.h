#ifndef TESTING_H

#define HUBBARD_TEST

#include <span>
#include "hubbard_compute.h"
#include "allocator.h"
#include "solver.h"

#define TEST_E_TOL real(1e-4)

const size_t TEST_ARENA_SIZE = 100*1024*1024;

class HubbardEnvironment : public ::testing::Environment
{
public:
   HubbardEnvironment() : errors(), cdev(&errors), allocator(TEST_ARENA_SIZE) {};

   void SetUp() override 
   {
      ASSERT_FALSE(errors.has_errors) << errors;
   }

   ErrorStream errors;
   HubbardComputeDevice cdev;
   ArenaAllocator allocator;
};

static HubbardEnvironment* const global_test_env = static_cast<HubbardEnvironment*>(::testing::AddGlobalTestEnvironment(new HubbardEnvironment));

template <StructuralHubbardParams P>       
class KSBasisTest : public testing::TestWithParam<HubbardParams>
{
public:
   static void SetUpTestSuite();
   static void TearDownTestSuite();

   static inline const HubbardParams params = P;

protected:
   static inline KSBlockIterator itr = KSBlockIterator(P, TEST_ARENA_SIZE);
};

class KBasisTest : public testing::TestWithParam<HubbardParams> {};

class HIntTest : public testing::TestWithParam<HubbardParams>
{
public:
   HIntTest() : itr({GetParam()}, TEST_ARENA_SIZE) {}

protected:
   void SetUp() override;

   KSBlockIterator itr;
   HubbardComputeDevice cdev;
};

void __set_up_KS_configs(KSBlockIterator& itr, const HubbardParams& params);

void __test_basis_K_and_configs(KSBlockIterator& itr, const std::vector<std::span<Det>>& KS_configs, const std::vector<int>& KS_S_path_counts,
                                const std::vector<real>& KS_spins, std::vector<real>& SCFs, const HubbardParams& params);

void __test_SCF_orthonormality(KSBlockIterator& itr, const std::vector<std::span<Det>>& KS_configs, const std::vector<int>& KS_S_path_counts,
                               const std::vector<real>& KS_spins, std::vector<real>& SCFs, const HubbardParams& params);

void __test_SCF_spins(KSBlockIterator& itr, const std::vector<std::span<Det>>& KS_configs, const std::vector<int>& KS_S_path_counts,
                      const std::vector<real>& KS_spins, std::vector<real>& SCFs, const HubbardParams& params);

#define KS_BASIS_TEST___(suite_name, T, U, Ns, N_up, N_dn)\
   using suite_name = KSBasisTest<StructuralHubbardParams{(real)T, (real)U, Ns, N_up, N_dn}>;\
   TEST_P(suite_name, test_basis_K_and_configs) { ASSERT_NO_FATAL_FAILURE(__test_basis_K_and_configs(itr, itr.KS_configs, itr.KS_S_path_counts, itr.KS_spins, itr.KS_SCF_coeffs, params)); }\
   TEST_P(suite_name, test_SCF_orthonormality)  { ASSERT_NO_FATAL_FAILURE(__test_SCF_orthonormality( itr, itr.KS_configs, itr.KS_S_path_counts, itr.KS_spins, itr.KS_SCF_coeffs, params)); }\
   TEST_P(suite_name, test_SCF_spins)           { ASSERT_NO_FATAL_FAILURE(__test_SCF_spins(          itr, itr.KS_configs, itr.KS_S_path_counts, itr.KS_spins, itr.KS_SCF_coeffs, params)); }\
   INSTANTIATE_TEST_SUITE_P(suite_name##_inst, suite_name, testing::Values(suite_name::params));

#define KS_BASIS_TEST__(count, ...) EXPAND(KS_BASIS_TEST___(KSBasisTest_##count, __VA_ARGS__))
#define KS_BASIS_TEST_(...) EXPAND(KS_BASIS_TEST__(__VA_ARGS__))
#define KS_BASIS_TEST(...) EXPAND(KS_BASIS_TEST_(__COUNTER__, __VA_ARGS__))

#define TESTING_H
#endif
