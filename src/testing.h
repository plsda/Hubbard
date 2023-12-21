#ifndef TESTING_H

#include "hubbard.h"

template <StructuralHubbardParams _p>       
class KSBasisTest : public testing::TestWithParam<HubbardParams>
{
public:
   static void SetUpTestSuite();
   static void TearDownTestSuite();

   static inline const HubbardParams params = _p;
   static inline std::vector<std::vector<Det>*> Kf_basis;
   static inline std::vector<int> Kf_counts;
   static inline std::vector<int> Kf_single_counts;
   static inline std::vector<real> Kf_spins;
   static inline std::vector<real> SCFs;
};

void __test_basis_K_and_configs(const std::vector<std::vector<Det>*>& Kf_basis, const std::vector<int>& Kf_counts,
                                const std::vector<real>& Kf_spins, std::vector<real>& SCFs, const HubbardParams& params);

void __test_SCF_orthonormality(const std::vector<std::vector<Det>*>& Kf_basis, const std::vector<int>& Kf_counts,
                               const std::vector<real>& Kf_spins, std::vector<real>& SCFs, const HubbardParams& params);

void __test_SCF_spins(const std::vector<std::vector<Det>*>& Kf_basis, const std::vector<int>& Kf_counts,
                      const std::vector<real>& Kf_spins, std::vector<real>& SCFs, const HubbardParams& params);


#define KS_BASIS_TEST___(suite_name, T, U, Ns, N_up, N_dn)\
   using suite_name = KSBasisTest<StructuralHubbardParams{(real)T, (real)U, Ns, N_up, N_dn}>;\
   TEST_P(suite_name, test_basis_K_and_configs) { __test_basis_K_and_configs(suite_name::Kf_basis, suite_name::Kf_counts, suite_name::Kf_spins, suite_name::SCFs, suite_name::params); }\
   TEST_P(suite_name, test_SCF_orthonormality)  { __test_SCF_orthonormality(suite_name::Kf_basis, suite_name::Kf_counts, suite_name::Kf_spins, suite_name::SCFs, suite_name::params); }\
   TEST_P(suite_name, test_SCF_spins)           { __test_SCF_spins(suite_name::Kf_basis, suite_name::Kf_counts, suite_name::Kf_spins, suite_name::SCFs, suite_name::params); }\
   INSTANTIATE_TEST_SUITE_P(suite_name##_inst, suite_name, testing::Values(StructuralHubbardParams{(real)T, (real)U, Ns, N_up, N_dn}));

#define KS_BASIS_TEST__(count, ...) EXPAND(KS_BASIS_TEST___(KSBasisTest_##count, __VA_ARGS__))
#define KS_BASIS_TEST_(...) EXPAND(KS_BASIS_TEST__(__VA_ARGS__))
#define KS_BASIS_TEST(...) EXPAND(KS_BASIS_TEST_(__COUNTER__, __VA_ARGS__))

#define TESTING_H
#endif
