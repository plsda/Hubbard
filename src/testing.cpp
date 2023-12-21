
#include <gtest/gtest.h>
#include <gmock/gmock.h>

using ::testing::Each;
using ::testing::DoubleNear;
using ::testing::Pointwise;
using ::testing::Eq;
using ::testing::WhenSorted;
using ::testing::Not;

#include "testing.h"
#include "utils.cpp"
#include "basis.cpp"


template <StructuralHubbardParams P>
void KSBasisTest<P>::SetUpTestSuite() 
{
   KConfigs configs = get_k_orbitals(params);

   u32 k_count = params.Ns;
   real m = 0.5*(params.N_up - params.N_down);
   real f_min = std::abs(m); 
   real f_max = 0.5*params.N;
   int f_count = (f_max - f_min) + 1;

   int total_SCF_count = 0;
   size_t configs_max_state_count = 0;
   for(size_t i = 0; i < k_count; i++)
   {
      size_t cur_state_count = configs.configs[i].size();
      if(configs_max_state_count < cur_state_count)
      {
         configs_max_state_count = cur_state_count;
      }
   }

   std::vector<int> single_counts(configs_max_state_count);

   int max_config_count = 0;
   int max_path_count = 0;
   for(size_t K_block_idx = 0; K_block_idx < k_count; K_block_idx++)
   {
      const auto& K_configs = configs.configs[K_block_idx];

      std::for_each(STD_POLICY_UNSEQ RangeItr(size_t(0)), RangeItr(K_configs.size()),
                    [&, K_configs](size_t i){single_counts[i] = count_singles(K_configs[i][0], params);});

      int K_block_SCF_count = 0;

      for(int fidx = 0; fidx < f_count; fidx++)
      {
         real f = f_min + fidx;

         size_t cur_Kf_basis_size = Kf_basis.size();
         form_KS_subbasis(f, m, K_configs, single_counts,
                          Kf_basis, Kf_single_counts, Kf_counts, Kf_spins,
                          max_config_count, max_path_count);
         cur_Kf_basis_size = Kf_basis.size() - cur_Kf_basis_size;

         std::span<std::vector<Det>*> cur_Kf_basis = std::span(Kf_basis.end() - cur_Kf_basis_size, Kf_basis.end());
         std::span<int> cur_Kf_counts              = std::span(Kf_counts.end() - cur_Kf_basis_size, Kf_counts.end());
         std::span<int> cur_Kf_single_counts       = std::span(Kf_single_counts.end() - cur_Kf_basis_size, Kf_single_counts.end());

         std::vector<Det> S_paths;
         S_paths.reserve(max_path_count);
         int Kf_dim = std::reduce(cur_Kf_counts.begin(), cur_Kf_counts.end());
         if(Kf_dim > 0)
         {
            form_SCFs(f, m,
                      cur_Kf_basis, cur_Kf_single_counts, cur_Kf_counts,
                      S_paths, params, SCFs);

            K_block_SCF_count += Kf_dim;
            total_SCF_count += Kf_dim;
         }

      }

      ASSERT_EQ(K_block_SCF_count, configs.block_sizes[K_block_idx]);

   }

   ASSERT_EQ(total_SCF_count, params.basis_size());
}

template <StructuralHubbardParams P>       
void KSBasisTest<P>::TearDownTestSuite() 
{
   Kf_basis.~vector();
   Kf_counts.~vector();
   Kf_single_counts.~vector();
   Kf_spins.~vector();
   SCFs.~vector();
}

void __test_SCF_spins(const std::vector<std::vector<Det>*>& Kf_basis,
                      const std::vector<int>& Kf_counts,
                      const std::vector<real>& Kf_spins,
                      std::vector<real>& SCFs,
                      const HubbardParams& params)
{
   size_t SCF_first_coeff_idx = 0;
   for(int config_idx = 0; config_idx < Kf_basis.size(); config_idx++)
   {
      std::vector<Det>* dets = Kf_basis[config_idx];
      size_t cur_det_count = dets->size();

      for(int path_idx = 0; path_idx < Kf_counts[config_idx]; path_idx++)
      {
         std::span<real> coeffs = std::span(SCFs.begin() + SCF_first_coeff_idx, cur_det_count);
         real f = SCF_spin(*dets, coeffs, params);

         if(!is_close(f, Kf_spins[config_idx], real(1e-6)))
         {
            for(real c : coeffs) { EXPECT_NEAR(c, 0.0, 1e-6); }
         }

         SCF_first_coeff_idx += cur_det_count;
      }
   }
}

void __test_basis_K_and_configs(const std::vector<std::vector<Det>*>& Kf_basis,
                                const std::vector<int>& Kf_counts,
                                const std::vector<real>& Kf_spins,
                                std::vector<real>& SCFs,
                                const HubbardParams& params)
{
   std::vector<int> conf_klist;
   std::vector<int> other_conf_klist;
   conf_klist.reserve(params.N);
   other_conf_klist.reserve(params.N);

   // NOTE: Verify momenta and config compatibilities (states with similar orbital configurations are grouped in the same group)
   for(int conf_idx = 0; conf_idx < Kf_basis.size(); conf_idx++)
   {
      std::vector<Det>* conf = Kf_basis[conf_idx];
      Det conf_ref_det = conf->at(0);

      int conf_K = state_momentum(conf_ref_det, params);
      conf_klist.resize(0);
      det2spinless_statelist(conf_ref_det, params, conf_klist);
      std::sort(conf_klist.begin(), conf_klist.end());

      // All dets in the current (orbital configuration) group should have the same spinless statelist
      for(Det det : *conf)
      {
         ASSERT_EQ(state_momentum(det, params), conf_K);

         other_conf_klist.resize(0);
         det2spinless_statelist(det, params, other_conf_klist);

         ASSERT_THAT(other_conf_klist, WhenSorted(conf_klist));
      }

      for(int other_conf_idx = 0; other_conf_idx < Kf_basis.size(); other_conf_idx++)
      {
         std::vector<Det>* other_conf = Kf_basis[other_conf_idx];
         Det other_conf_ref_det = other_conf->at(0);

         if((conf_idx != other_conf_idx) && (Kf_spins[conf_idx] == Kf_spins[other_conf_idx]))
         {
            other_conf_klist.resize(0);
            det2spinless_statelist(other_conf_ref_det, params, other_conf_klist);

            ASSERT_THAT(other_conf_klist, Not(WhenSorted(conf_klist)));
         }
      }
   }
}

void __test_SCF_orthonormality(const std::vector<std::vector<Det>*>& Kf_basis,
                               const std::vector<int>& Kf_counts,
                               const std::vector<real>& Kf_spins,
                               std::vector<real>& SCFs,
                               const HubbardParams& params)
{
   size_t testbra_first_coeff_idx = 0;
   for(int testbra_idx = 0; testbra_idx < Kf_basis.size(); testbra_idx++)
   {
      std::vector<Det>* testbra_dets = Kf_basis[testbra_idx];
      size_t testbra_det_count = testbra_dets->size();

      for(int testbra_path_idx = 0; testbra_path_idx < Kf_counts[testbra_idx]; testbra_path_idx++)
      {
         std::span<real> testbra_coeffs = std::span(SCFs.begin() + testbra_first_coeff_idx, testbra_det_count);

         size_t testket_first_coeff_idx = 0;
         for(int testket_idx = 0; testket_idx < Kf_basis.size(); testket_idx++)
         {
            std::vector<Det>* testket_dets = Kf_basis[testket_idx];
            size_t testket_det_count = testket_dets->size();

            for(int testket_path_idx = 0; testket_path_idx < Kf_counts[testket_idx]; testket_path_idx++)
            {
               std::span<real> testket_coeffs = std::span(SCFs.begin() + testket_first_coeff_idx, testket_det_count);
               real inner = SCF_inner(*testbra_dets, testbra_coeffs, *testket_dets, testket_coeffs);

               if((testbra_idx != testket_idx) || (testbra_path_idx != testket_path_idx))
               {
                  ASSERT_NEAR(inner, 0, real(1e-6)); 
               } 
               else
               {
                  ASSERT_NEAR(inner, 1, real(1e-6));
               }

               testket_first_coeff_idx += testket_det_count;
            }
         }

         testbra_first_coeff_idx += testbra_det_count;
      }
   }
}

//            T   U   Ns N_up N_dn
KS_BASIS_TEST(1, 2.7, 5,   1,  1)
KS_BASIS_TEST(1, 2.7, 5,   1,  2)
KS_BASIS_TEST(1, 2.7, 5,   2,  1)
KS_BASIS_TEST(1, 2.7, 5,   2,  2)
KS_BASIS_TEST(1, 2.7, 6,   3,  1)
KS_BASIS_TEST(1, 2.7, 7,   4,  2)
KS_BASIS_TEST(1, 2.7, 7,   3,  3)
KS_BASIS_TEST(1, 2.7, 7,   4,  3)


TEST(GTestTest, BasicAssertions)
{
  EXPECT_STRNE("hello", "world");
  EXPECT_EQ(7*6, 42);
}

/*
TEST_P(KBasisTest, test_K_basis)
{
   //  TODO: Parameters
   
   // NOTE: Test that the output of form_K_basis makes sense (takes as input the output of form_K_basis(params))
   
   const HubbardParams& params = GetParam();
   auto [basis, momenta, block_sizes] = form_K_basis(params);

   ASSERT_EQ(basis.size(), params.basis_size());
   auto [K_min, K_max] = std::minmax_element(momenta.begin(), momenta.end());
   ASSERT_GE(*K_min, 0);
   ASSERT_LE(*K_max, params.Ns);

   // Verify momenta and similarity of orbital configurations for states in the given constant-K subbases
   std::set<int> kset;
   size_t block_first_det_idx = 0;
   for(int block_idx = 0; block_idx < block_sizes.size(); block_idx++)
   {
      int block_size = block_sizes[block_idx];
      ASSERT_GT(block_size, 0);
      Det conf_ref_det = basis[block_first_det_idx];

      int conf_K = state_momentum(conf_ref_det, params);
      ASSERT_EQ(conf_K, momenta[block_first_det_idx]);

      ASSERT_TRUE(!kset.contains(conf_K));
      kset.insert(conf_K);

      // Verify that all dets in a config group have the same momentum
      for(size_t didx = block_first_det_idx;
          didx < block_first_det_idx + block_size;
          didx++)
      {
         Det det = basis[didx];

         ASSERT_EQ(state_momentum(det, params), conf_K);
      }

      block_first_det_idx += block_size;
   }
}

TEST_P(HIntTest, test_Hint_symmetry)
{
   // NOTE: Verify that compute_H_int_element gives a symmetric (real) matrix 
   // TODO: Pass/compute Kf_basis

   for(int ket_k_idx = 0; ket_k_idx < Kf_basis.size(); ket_k_idx++)
   {
      const std::vector<Det>& ket_dets = *Kf_basis[ket_k_idx];
      for(int ket_path_idx = 0; ket_path_idx < Kf_counts[ket_k_idx]; ket_path_idx++)
      {
         Eigen::array<Eigen::Index, 3> offsets = {ket_k_idx, ket_path_idx, 0};
         Eigen::array<Eigen::Index, 3> extents = {1, 1, max_config_count};
         Eigen::TensorRef<Arr3R> ket_coeffs = SCFs.slice(offsets, extents);

         for(int bra_k_idx = 0; bra_k_idx < Kf_basis.size(); bra_k_idx++)
         {
            const std::vector<Det>& bra_dets = *Kf_basis[bra_k_idx];
            for(int bra_path_idx = 0; bra_path_idx < Kf_counts[bra_k_idx]; bra_path_idx++)
            {
               offsets = {bra_k_idx, bra_path_idx, 0};
               extents = {1, 1, max_config_count};
               Eigen::TensorRef<Arr3R> bra_coeffs = SCFs.slice(offsets, extents);

               H_Kf_rc = compute_H_int_element(bra_dets, bra_coeffs, ket_dets, ket_coeffs, params);
               H_Kf_cr = compute_H_int_element(ket_dets, ket_coeffs, bra_dets, bra_coeffs, params);

               ASSERT_NEAR(H_Kf_rc, H_Kf_cr);
            }
         }
      }
   }

}

TEST(SolverTest, DISABLED_test_half_filling_energy) 
{
   // TODO:
   //   - Dimer
   //   - Asymptotic (need large enough particle and site count)
   
   //real T = 1;
   //real U = real(2.7);
   //int Ns = 5;
   //int N_up = 2;
   //int N_down = 2;

   //HubbardParams params(T, U, Ns, N_up, N_down);
   //EXPECT_NEAR(kfm_basis_compute_E0(params), )

}

TEST(SolverTest, DISABLED_test_non_half_filling_energy) 
{
   // TODO:
}

TEST(QuadTest, DISABLED_test_quad)
{
   // TODO: Test the integration function
}
*/


