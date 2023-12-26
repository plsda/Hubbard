#ifndef TESTING_H

#include "hubbard.h"

#define TEST_E_TOL real(1e-4)

template <StructuralHubbardParams P>       
class KSBasisTest : public testing::TestWithParam<HubbardParams>
{
public:
   static void SetUpTestSuite();
   static void TearDownTestSuite();

   static inline const HubbardParams params = P;

protected:
   static inline std::vector<std::shared_ptr<std::vector<Det>>> Kf_basis; 
   static inline std::vector<int> Kf_counts;
   static inline std::vector<int> Kf_single_counts;
   static inline std::vector<real> Kf_spins;
   static inline std::vector<real> SCFs;
};

class KBasisTest : public testing::TestWithParam<HubbardParams> {};

class HIntTest : public testing::TestWithParam<HubbardParams>
{
protected:
   void SetUp() override;

   int max_config_count;
   int max_path_count;
   std::vector<std::shared_ptr<std::vector<Det>>> Kf_basis; 
   std::vector<int> Kf_counts;
   std::vector<real> SCFs;

};

void set_up_KS_basis(const HubbardParams& params, std::vector<std::shared_ptr<std::vector<Det>>>& Kf_basis,
                     std::vector<int>& Kf_counts, std::vector<int>& Kf_single_counts, std::vector<real>& Kf_spins,
                     std::vector<real>& SCFs, int& max_config_count, int& max_path_count);

void __test_basis_K_and_configs(const std::vector<std::shared_ptr<std::vector<Det>>>& Kf_basis, const std::vector<int>& Kf_counts,
                                const std::vector<real>& Kf_spins, std::vector<real>& SCFs, const HubbardParams& params);

void __test_SCF_orthonormality(const std::vector<std::shared_ptr<std::vector<Det>>>& Kf_basis, const std::vector<int>& Kf_counts,
                               const std::vector<real>& Kf_spins, std::vector<real>& SCFs, const HubbardParams& params);

void __test_SCF_spins(const std::vector<std::shared_ptr<std::vector<Det>>>& Kf_basis, const std::vector<int>& Kf_counts,
                      const std::vector<real>& Kf_spins, std::vector<real>& SCFs, const HubbardParams& params);


#define KS_BASIS_TEST___(suite_name, T, U, Ns, N_up, N_dn)\
   using suite_name = KSBasisTest<StructuralHubbardParams{(real)T, (real)U, Ns, N_up, N_dn}>;\
   TEST_P(suite_name, test_basis_K_and_configs) { ASSERT_NO_FATAL_FAILURE(__test_basis_K_and_configs(Kf_basis, Kf_counts, Kf_spins, SCFs, params)); }\
   TEST_P(suite_name, test_SCF_orthonormality)  { ASSERT_NO_FATAL_FAILURE(__test_SCF_orthonormality(Kf_basis, Kf_counts, Kf_spins, SCFs, params)); }\
   TEST_P(suite_name, test_SCF_spins)           { ASSERT_NO_FATAL_FAILURE(__test_SCF_spins(Kf_basis, Kf_counts, Kf_spins, SCFs, params)); }\
   INSTANTIATE_TEST_SUITE_P(suite_name##_inst, suite_name, testing::Values(suite_name::params));

#define KS_BASIS_TEST__(count, ...) EXPAND(KS_BASIS_TEST___(KSBasisTest_##count, __VA_ARGS__))
#define KS_BASIS_TEST_(...) EXPAND(KS_BASIS_TEST__(__VA_ARGS__))
#define KS_BASIS_TEST(...) EXPAND(KS_BASIS_TEST_(__COUNTER__, __VA_ARGS__))

#define TESTING_H
#endif
