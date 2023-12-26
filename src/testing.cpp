
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
#include "solver.cpp"


//            T   U   Ns N_up N_dn
KS_BASIS_TEST(1, 2.7, 5,   1,  1)
KS_BASIS_TEST(1, 2.7, 5,   1,  2)
KS_BASIS_TEST(1, 2.7, 5,   2,  1)
KS_BASIS_TEST(1, 2.7, 5,   2,  2)
KS_BASIS_TEST(1, 2.7, 6,   3,  1)
KS_BASIS_TEST(1, 2.7, 7,   4,  2)
KS_BASIS_TEST(1, 2.7, 7,   3,  3)
KS_BASIS_TEST(1, 2.7, 7,   4,  3)

INSTANTIATE_TEST_SUITE_P(KBasisTest_small, KBasisTest,
                         testing::Values(
                              //            T   U   Ns N_up N_dn
                              HubbardParams(1, 2.7, 5,   1,  1),
                              HubbardParams(1, 2.7, 5,   1,  2),
                              HubbardParams(1, 2.7, 5,   2,  1),
                              HubbardParams(1, 2.7, 5,   2,  2),
                              HubbardParams(1, 2.7, 6,   3,  1),
                              HubbardParams(1, 2.7, 7,   4,  2),
                              HubbardParams(1, 2.7, 7,   3,  3),
                              HubbardParams(1, 2.7, 7,   4,  3)
                          ));

INSTANTIATE_TEST_SUITE_P(HIntTest_small, HIntTest,
                         testing::Values(
                              //            T   U   Ns N_up N_dn
                              HubbardParams(1, 2.7, 5,   1,  1),
                              HubbardParams(1, 2.7, 5,   1,  2),
                              HubbardParams(1, 2.7, 5,   2,  1),
                              HubbardParams(1, 2.7, 5,   2,  2),
                              HubbardParams(1, 2.7, 6,   3,  1),
                              HubbardParams(1, 2.7, 7,   4,  2),
                              HubbardParams(1, 2.7, 7,   3,  3),
                              HubbardParams(1, 2.7, 7,   4,  3)
                          ));


#define set_up_KS_basis(...) ASSERT_NO_FATAL_FAILURE(EXPAND(__set_up_KS_basis(__VA_ARGS__)));
void __set_up_KS_basis(const HubbardParams& params, 
                       std::vector<std::shared_ptr<std::vector<Det>>>& Kf_basis,
                       std::vector<int>& Kf_counts,
                       std::vector<int>& Kf_single_counts,
                       std::vector<real>& Kf_spins,
                       std::vector<real>& SCFs, 
                       int& max_config_count,
                       int& max_path_count)
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

   max_config_count = 0;
   max_path_count = 0;
   for(size_t K_block_idx = 0; K_block_idx < k_count; K_block_idx++)
   {
      const auto K_configs = configs.configs[K_block_idx];

      std::for_each(STD_POLICY_UNSEQ RangeItr(size_t(0)), RangeItr(K_configs.size()),
                    [K_configs, &single_counts, &params](size_t i){single_counts[i] = count_singles(K_configs[i]->front(), params);});

      int K_block_SCF_count = 0;

      for(int fidx = 0; fidx < f_count; fidx++)
      {
         real f = f_min + fidx;

         size_t cur_Kf_basis_size = Kf_basis.size();
         int cur_max_config_count;
         int cur_max_path_count;
         form_KS_subbasis(f, m, K_configs, single_counts,
                          Kf_basis, Kf_single_counts, Kf_counts, Kf_spins,
                          cur_max_config_count, cur_max_path_count);
         cur_Kf_basis_size = Kf_basis.size() - cur_Kf_basis_size;

         auto cur_Kf_basis = std::span(Kf_basis.end() - cur_Kf_basis_size, Kf_basis.end());
         std::span<int> cur_Kf_counts              = std::span(Kf_counts.end() - cur_Kf_basis_size, Kf_counts.end());
         std::span<int> cur_Kf_single_counts       = std::span(Kf_single_counts.end() - cur_Kf_basis_size, Kf_single_counts.end());

         std::vector<Det> S_paths;
         S_paths.reserve(cur_max_path_count);
         int Kf_dim = std::reduce(cur_Kf_counts.begin(), cur_Kf_counts.end());
         if(Kf_dim > 0)
         {
            form_SCFs(f, m,
                      cur_Kf_basis, cur_Kf_single_counts, cur_Kf_counts,
                      S_paths, params, SCFs);

            K_block_SCF_count += Kf_dim;
            total_SCF_count += Kf_dim;
         }

         if(max_config_count < cur_max_config_count)
         {
            max_config_count = cur_max_config_count;
         }

         if(max_path_count < cur_max_path_count)
         {
            max_path_count = cur_max_path_count;
         }

      }

      ASSERT_EQ(K_block_SCF_count, configs.block_sizes[K_block_idx]);

   }

   ASSERT_EQ(total_SCF_count, params.basis_size());

}

template <StructuralHubbardParams P>
void KSBasisTest<P>::SetUpTestSuite() 
{
   int max_config_count;
   int max_path_count;
   set_up_KS_basis(params, Kf_basis, Kf_counts, Kf_single_counts, Kf_spins, SCFs, max_config_count, max_path_count);
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

void __test_basis_K_and_configs(const std::vector<std::shared_ptr<std::vector<Det>>>& Kf_basis,
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
      //std::vector<Det> conf = *Kf_basis[conf_idx];
      auto conf = Kf_basis[conf_idx];
      Det conf_ref_det = conf->front();

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
         Det other_conf_ref_det = Kf_basis[other_conf_idx]->front();

         if((conf_idx != other_conf_idx) && (Kf_spins[conf_idx] == Kf_spins[other_conf_idx]))
         {
            other_conf_klist.resize(0);
            det2spinless_statelist(other_conf_ref_det, params, other_conf_klist);

            ASSERT_THAT(other_conf_klist, Not(WhenSorted(conf_klist)));
         }
      }
   }
}

void __test_SCF_orthonormality(const std::vector<std::shared_ptr<std::vector<Det>>>& Kf_basis,
                               const std::vector<int>& Kf_counts,
                               const std::vector<real>& Kf_spins,
                               std::vector<real>& SCFs,
                               const HubbardParams& params)
{
   size_t testbra_first_coeff_idx = 0;
   for(int testbra_idx = 0; testbra_idx < Kf_basis.size(); testbra_idx++)
   {
      const auto testbra_dets = Kf_basis[testbra_idx];
      size_t testbra_det_count = Kf_basis[testbra_idx]->size();

      for(int testbra_path_idx = 0; testbra_path_idx < Kf_counts[testbra_idx]; testbra_path_idx++)
      {
         std::span<real> testbra_coeffs = std::span(SCFs.begin() + testbra_first_coeff_idx, testbra_det_count);

         size_t testket_first_coeff_idx = 0;
         for(int testket_idx = 0; testket_idx < Kf_basis.size(); testket_idx++)
         {
            const auto testket_dets = Kf_basis[testket_idx];
            size_t testket_det_count = Kf_basis[testket_idx]->size();

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

void __test_SCF_spins(const std::vector<std::shared_ptr<std::vector<Det>>>& Kf_basis,
                      const std::vector<int>& Kf_counts,
                      const std::vector<real>& Kf_spins,
                      std::vector<real>& SCFs,
                      const HubbardParams& params)
{
   size_t SCF_first_coeff_idx = 0;
   for(int config_idx = 0; config_idx < Kf_basis.size(); config_idx++)
   {
      auto dets = Kf_basis[config_idx];
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


TEST(GTestTest, BasicAssertions)
{
  EXPECT_STRNE("hello", "world");
  EXPECT_EQ(7*6, 42);
}

TEST_P(KBasisTest, test_K_basis)
{
   // NOTE: Test that the output of form_K_basis makes sense 
   
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


void HIntTest::SetUp()
{
   const HubbardParams& params = GetParam();
   std::vector<int> Kf_single_counts;
   std::vector<real> Kf_spins;

   set_up_KS_basis(params, Kf_basis, Kf_counts, Kf_single_counts, Kf_spins, SCFs, max_config_count, max_path_count);

}

TEST_P(HIntTest, test_Hint)
{
   // NOTE: Verify that compute_H_int_element gives a symmetric (real) matrix with reasonable cofficients

   const HubbardParams& params = GetParam();

   size_t testket_first_coeff_idx = 0;
   for(int ket_k_idx = 0; ket_k_idx < Kf_basis.size(); ket_k_idx++)
   {
      const std::vector<Det>& ket_dets = *Kf_basis[ket_k_idx];
      size_t testket_det_count = ket_dets.size();

      for(int ket_path_idx = 0; ket_path_idx < Kf_counts[ket_k_idx]; ket_path_idx++)
      {
         std::span<real> ket_coeffs = std::span(SCFs.begin() + testket_first_coeff_idx, testket_det_count);

         size_t testbra_first_coeff_idx = 0;
         for(int bra_k_idx = 0; bra_k_idx < Kf_basis.size(); bra_k_idx++)
         {
            const std::vector<Det>& bra_dets = *Kf_basis[bra_k_idx];
            size_t testbra_det_count = bra_dets.size();

            for(int bra_path_idx = 0; bra_path_idx < Kf_counts[bra_k_idx]; bra_path_idx++)
            {
               std::span<real> bra_coeffs = std::span(SCFs.begin() + testbra_first_coeff_idx, testbra_det_count);

               real H_Kf_rc = compute_H_int_element(bra_dets, bra_coeffs, ket_dets, ket_coeffs, params);
               real H_Kf_cr = compute_H_int_element(ket_dets, ket_coeffs, bra_dets, bra_coeffs, params);

               ASSERT_NEAR(H_Kf_rc, H_Kf_cr, real(1e-6));
               //ASSERT_NEAR(H_Kf_rc, H_Kf_cr, real(1e-5));
               ASSERT_LE(std::abs(H_Kf_rc), (params.U/params.Ns)*params.Ns*params.Ns*params.Ns*ket_dets.size()*bra_dets.size());

               testbra_first_coeff_idx += testbra_det_count;
            }
         }

         testket_first_coeff_idx += testket_det_count;
      }
   }

}


TEST(SolverTest, test_dimer_E0)
{
   std::vector<HubbardParams> params = 
   {
      //            T  U  Ns N_up N_dn
      HubbardParams(2, 1, 2,  0,   1),
      HubbardParams(2, 1, 2,  0,   2),
      HubbardParams(2, 3, 2,  1,   1),
      HubbardParams(2,-7, 2,  1,   1),
      HubbardParams(2,-1, 2,  1,   2),
      HubbardParams(1, 2, 2,  2,   2),
   };

   for(const HubbardParams& p : params)
   {
      real ground_truth = dimer_E0(p, BCS::PERIODIC);
      real result = kfm_basis_compute_E0(p);

      EXPECT_NEAR(result, ground_truth, TEST_E_TOL) << p;
   }
}

TEST(SolverTest, test_atomic_E0)
{
   std::vector<HubbardParams> params = 
   {
      //            T  U  Ns N_up N_dn
      HubbardParams(0, 1, 2,  1,   1),
      HubbardParams(0, 1, 5,  2,   2),
      HubbardParams(0,-1, 5,  2,   2),
      HubbardParams(0, 8, 7,  5,   2),
      HubbardParams(0, 8, 7,  2,   5),
      HubbardParams(0,-8, 7,  2,   5),
   };

   for(const HubbardParams& p : params)
   {
      real ground_truth = atomic_E0(p);
      real result = kfm_basis_compute_E0(p);

      EXPECT_NEAR(result, ground_truth, TEST_E_TOL) << p;
   }

}

TEST(SolverTest, test_noninteracting_E0)
{
   std::vector<HubbardParams> params = 
   {
      //            T  U  Ns N_up N_dn
      HubbardParams(2, 0, 2,  1,   1),
      HubbardParams(2, 0, 5,  2,   2),
      HubbardParams(7, 0, 7,  5,   2),
      HubbardParams(7, 0, 7,  2,   5),
   };

   for(const HubbardParams& p : params)
   {
      real ground_truth = noninteracting_E0(p, BCS::PERIODIC);
      real result = kfm_basis_compute_E0(p);

      EXPECT_NEAR(result, ground_truth, TEST_E_TOL) << p;
   }

}

TEST(QuadTest, test_quad)
{
   {
      IntArgs args = {.lower = -1, .upper = 9, .abs_tol = 1e-12, .rel_tol = 1e-8, .min_steps = 2, .max_steps = 100};
      auto integrand = [](real x) {return x*x*x + 0.5*x;};
      real ground_truth = 1660;

      real result = quad(integrand, args);

      EXPECT_NEAR(result, ground_truth, real(1e-6));
   }

   {
      IntArgs args = {.lower = -1, .upper = 9, .abs_tol = 1e-12, .rel_tol = 1e-8, .min_steps = 2, .max_steps = 100};
      real a = 3;
      auto integrand = [a](real x) {return std::sin(a*x);};
      real ground_truth = 1.0/a*(std::cos(a*args.lower) - std::cos(a*args.upper));

      real result = quad(integrand, args);

      EXPECT_NEAR(result, ground_truth, real(1e-6));
   }

   {
      IntArgs args = {.lower = -2, .upper = 5, .abs_tol = 1e-12, .rel_tol = 1e-8, .min_steps = 2, .max_steps = 100};
      auto integrand = [](real x) {return std::exp(x);};
      real ground_truth = std::exp(args.upper) - std::exp(args.lower);

      real result = quad(integrand, args);

      EXPECT_NEAR(result, ground_truth, real(1e-6));
   }

   {
      IntArgs args = {.lower = 0, .upper = 5, .abs_tol = 1e-12, .rel_tol = 1e-8, .min_steps = 2, .max_steps = 100};
      auto integrand = [](real x) {return std::exp(-x*x);};
      real ground_truth = 0.8862269255;

      real result = quad(integrand, args);

      EXPECT_NEAR(result, ground_truth, real(1e-6));
   }
}
