
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
#include "allocator.cpp"
#include "solver.cpp"
#include "profiler.cpp"

// NOTE: Do not run these tests in parallel.

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(KBasisTest);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(HIntTest);

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
                              HubbardParams(1, 2.7, 7,   4,  3),
                              HubbardParams(1, 2.7, 9,   4,  4)
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
                              //,HubbardParams(5.2, 3.8, 9, 4,  4)
                          ));

#define set_up_KS_configs(...) ASSERT_NO_FATAL_FAILURE(EXPAND(__set_up_KS_configs(__VA_ARGS__)));
void __set_up_KS_configs(KSBlockIterator& itr, const HubbardParams& params)
{
   KSBlockIterator temp_itr(params, global_test_env->allocator, hubbard_memory_requirements(params));
   itr.clear_block_data();

   for(temp_itr.reset(); temp_itr; ++temp_itr) 
   {
      itr.append_block_data(temp_itr);
   }

   itr.copy_basis(temp_itr);
}

template <StructuralHubbardParams P>
void KSBasisTest<P>::SetUpTestSuite() 
{
   set_up_KS_configs(itr, params);
}

template <StructuralHubbardParams P>       
void KSBasisTest<P>::TearDownTestSuite() 
{
   //itr.~KSBlockIterator(); 
   //allocator.~ArenaAllocator();
}

void __test_basis_K_and_configs(KSBlockIterator& itr,
                                const std::vector<std::span<Det>, SpanArena>& KS_configs,
                                const std::vector<int, IntArena>& KS_S_path_counts,
                                const std::vector<real, RealArena>& KS_spins,
                                std::vector<real, RealArena>& CSFs,
                                const HubbardParams& params)
{
   std::vector<int> conf_klist;
   std::vector<int> other_conf_klist;
   conf_klist.reserve(params.N);
   other_conf_klist.reserve(params.N);

   // NOTE: Verify momenta and config compatibilities (states with similar orbital configurations are grouped in the same group)
   for(int conf_idx = 0; conf_idx < KS_configs.size(); conf_idx++)
   {
      std::span<Det> conf = KS_configs[conf_idx];
      Det conf_ref_det = get_config_ref_det(conf);

      int conf_K = state_momentum(conf_ref_det, params);
      conf_klist.resize(0);
      det2spinless_statelist(conf_ref_det, params, conf_klist);
      std::sort(conf_klist.begin(), conf_klist.end());

      // All dets in the current (orbital configuration) group should have the same spinless statelist
      for(Det det : conf)
      {
         ASSERT_EQ(state_momentum(det, params), conf_K);

         other_conf_klist.resize(0);
         det2spinless_statelist(det, params, other_conf_klist);

         ASSERT_THAT(other_conf_klist, WhenSorted(conf_klist));
      }

      for(int other_conf_idx = 0; other_conf_idx < KS_configs.size(); other_conf_idx++)
      {
         Det other_conf_ref_det = get_config_ref_det(KS_configs[other_conf_idx]);

         if((conf_idx != other_conf_idx) && (KS_spins[conf_idx] == KS_spins[other_conf_idx]))
         {
            other_conf_klist.resize(0);
            det2spinless_statelist(other_conf_ref_det, params, other_conf_klist);

            ASSERT_THAT(other_conf_klist, Not(WhenSorted(conf_klist)));
         }
      }
   }
}

void __test_CSF_orthonormality(KSBlockIterator& itr,
                               const std::vector<std::span<Det>,
                               SpanArena>& KS_configs,
                               const std::vector<int, IntArena>& KS_S_path_counts,
                               const std::vector<real, RealArena>& KS_spins,
                               std::vector<real, RealArena>& CSFs,
                               const HubbardParams& params)
{
   for(auto csf1 : itr.KS_basis())
   {
      for(auto csf2 : itr.KS_basis())
      {
         real inner = CSF_inner(itr.CSF_dets_sp(csf1), itr.CSF_coeffs_sp(csf1),
                                itr.CSF_dets_sp(csf2), itr.CSF_coeffs_sp(csf2));

         if(csf1 != csf2)
         {
            ASSERT_NEAR(inner, 0, real(1e-6)); 
         } 
         else
         {
            ASSERT_NEAR(inner, 1, real(1e-6));
         }

      }
   }
}

void __test_CSF_spins(KSBlockIterator& itr,
                      const std::vector<std::span<Det>,
                      SpanArena>& KS_configs,
                      const std::vector<int, IntArena>&
                      KS_S_path_counts,
                      const std::vector<real, RealArena>& KS_spins,
                      std::vector<real, RealArena>& CSFs,
                      const HubbardParams& params)
{
   for(auto csf : itr.KS_basis())
   {
      real S = CSF_spin(itr.CSF_dets_sp(csf), itr.CSF_coeffs_sp(csf), params);

      if(!is_close(S, KS_spins[csf.config_idx], real(1e-6)))
      {
         for(real c : itr.CSF_coeffs_sp(csf)) { ASSERT_NEAR(c, 0.0, 1e-6); }
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

TEST_P(KBasisTest, test_K_basis_sort)
{
   // NOTE: Test that the output of form_K_basis is correctly sorted by sort_K_basis

   const HubbardParams& params = GetParam();
   KBasis kbasis = form_K_basis(params);
   sort_K_basis(kbasis, params);

   int i0 = 0;
   for(int block_size : kbasis.block_sizes)
   {
      if(block_size == 0) { continue; }
      int i1 = i0 + block_size;

      real m = 0.5*std::abs(params.N_up - params.N_down);
      int same_count = 1;
      Det prev = kbasis.basis[i0];
      for(int i = i0 + 1; i < i1; i++)
      {
         Det cur = kbasis.basis[i];
         if(cmp_det_config(prev, cur, params))
         {
            same_count++;
         }
         else
         {
            int single_count = count_singles(prev, params);
            ASSERT_EQ(same_count, dets_per_orbital_config(single_count, params));
            same_count = 1;
            prev = cur;
         }
      }
      int single_count = count_singles(prev, params);
      ASSERT_EQ(same_count, dets_per_orbital_config(single_count, params));

      i0 += block_size;
   }
}


void HIntTest::SetUp()
{
   const HubbardParams& params = GetParam();
   set_up_KS_configs(itr, params);
}

TEST_P(HIntTest, test_Hint)
{
   // NOTE: Verify that H_int_element gives a symmetric (real) matrix with reasonable cofficients

   const HubbardParams& params = GetParam();

   for(auto csf1 : itr.KS_basis())
   {
      for(auto csf2 : itr.KS_basis())
      {
         real H_KS_rc = cdev.H_int_element(itr.CSF_dets(csf1), itr.CSF_coeffs(csf1), itr.CSF_size(csf1), 
                                           itr.CSF_dets(csf2), itr.CSF_coeffs(csf2), itr.CSF_size(csf2),
                                           itr.params);
         real H_KS_cr = cdev.H_int_element(itr.CSF_dets(csf2), itr.CSF_coeffs(csf2), itr.CSF_size(csf2), 
                                           itr.CSF_dets(csf1), itr.CSF_coeffs(csf1), itr.CSF_size(csf1),
                                           itr.params);

         ASSERT_NEAR(H_KS_rc, H_KS_cr, real(1e-5));
         ASSERT_LE(std::abs(H_KS_rc), (params.U/params.Ns)*params.Ns*params.Ns*params.Ns*itr.CSF_size(csf1)*itr.CSF_size(csf2));
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
      // Create a new HubbardModel instance each time since params change and have to rebuild the basis anyways
      real result = HubbardModel(p, global_test_env->cdev, global_test_env->allocator).E0();

      EXPECT_NEAR(result, ground_truth, TEST_E_TOL) << p;
   }
}

TEST(SolverTest, test_atomic_E0)
{
   std::vector<HubbardParams> params = 
   {
      //            T  U    Ns N_up N_dn
      HubbardParams(0, 1,   2,  1,   1),
      HubbardParams(0, 1.7, 5,  2,   2),
      HubbardParams(0,-1,   5,  2,   2),
      HubbardParams(0, 8,   7,  5,   2),
      HubbardParams(0, 8.3, 7,  2,   5),
      HubbardParams(0,-2.7, 7,  2,   5),
      HubbardParams(0, 5,   9,  4,   4),
   };

   for(const HubbardParams& p : params)
   {
      real ground_truth = atomic_E0(p);
      real result = HubbardModel(p, global_test_env->cdev, global_test_env->allocator).E0();

      EXPECT_NEAR(result, ground_truth, TEST_E_TOL) << p;
   }

}

TEST(SolverTest, test_noninteracting_E0)
{
   std::vector<HubbardParams> params = 
   {
      //            T    U  Ns N_up N_dn
      HubbardParams(2,   0, 2,  1,   1),
      HubbardParams(2.8, 0, 5,  2,   2),
      HubbardParams(7,   0, 7,  5,   2),
      HubbardParams(4.2, 0, 7,  2,   5),
      HubbardParams(5,   0, 9,  4,   4),
   };

   for(const HubbardParams& p : params)
   {
      real ground_truth = noninteracting_E0(p, BCS::PERIODIC);
      real result = HubbardModel(p, global_test_env->cdev, global_test_env->allocator).E0();

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

//int main(int argc, char **argv)
//{
//   testing::InitGoogleTest(&argc, argv);
//   global_test_env = static_cast<HubbardEnvironment*>(::testing::AddGlobalTestEnvironment(new HubbardEnvironment));
//   return RUN_ALL_TESTS();
//}
