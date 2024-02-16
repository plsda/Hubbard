# Hubbard
"Hubbard" is an exact-diagonalization solver for the 1D Hubbard model. It can solve and interactively plot the ground state energy (and can be extended to compute other quantities, such as higher-energy states
and self energies) of the 1D Hubbard model with periodic boundary conditions in a given subspace with fixed particle number and magnetization of the full Fock space.  With periodic boundary conditions and in
terms of momentum-space annihilation, $\hat{c}\_{k\sigma}$, and creation, $\hat{c}^\dagger\_{k\sigma}$, operators, the system is described by the Hamiltonian
```math
\sum_{k = 1}^{N_s} \sum_{\sigma = \uparrow, \downarrow} \epsilon(k) \hat{c}^\dagger_{k\sigma}\hat{c}_{k\sigma} +
\frac{U}{N_s} \sum_{k,k',q = 1}^{N_s} \hat{c}^\dagger_{k+q\uparrow}\hat{c}_{k\uparrow} \hat{c}^\dagger_{k'-q\downarrow}\hat{c}_{k'\downarrow},
```
where $\epsilon, U, N_s$ are the non-interacting per-particle energy, interaction strength and site count, respectively. An analytic Bethe-Ansatz-based solution for the ground state of the 1D Hubbard model has
also been presented in the literature [1].

The solver operates in the momentum-spin basis, and 
utilizes the translation ($\mathbb{Z}_{N_s}$) and spin-rotation ($SU(2)$) symmetries of the periodic Hubbard model to reduce the problem of solving the energy eigenstate(s) into a series of small and dense diagonalization
problems, whereas computation in the site-basis yields a much larger, but sparse, diagonalization problem. The computational basis for each subspace spanned by states of a given total momentum and total spin, consisting of spin
configuration functions (CSFs), is formed using the so-called genealogical coupling method, which affords one to systematically and efficiently from CSFs as linear combinations of determinantal basis states.

The current implementation is naive in the sense that the basis as well as the Hamiltonian matrix are formed and stored in their entirety, and the Hamiltonian matrix is formed
by simply explicitly applying annihilation and creation operators as-is to the determinantal basis. In addition, currently only systems with up to 16 sites and periodic boundary conditions are supported.

![Plot for a 10-site, zero-magnetization system and comparison to the result for half-filled Hubbard systems presented in [1].](example_run.png)


## Building
Building the project requires CMake >= 3.25, Eigen 3.4 and OpenGL >= 3.0, in addition to the dependencies included as submodules. To build the project (with CUDA) for the first time, in the root directory, run:
```
mkdir build && cd build
cmake -DEigen_3_DIR="your/path/to/eigen3/cmake" -DHUBBARD_TESTING=ON -DHUBBARD_UPDATE_SUBMODULES=ON -DHUBBARD_USE_CUDA=ON ..
cmake --build .
```

## Testing
This project uses GoogleTest for unit testing. All tests are included in `testing.cpp`, and they can be executed by building the target `hubbard_testing`
and then running `ctest`.

## References
[1]: E. Lieb, and F. Y. Wu. The one-dimensional Hubbard model: a reminiscence. Physica A, 2003.

[2]: W. Duch. From determinants to spin eigenfunctions—a simple algorithm. Int. J. Quantum Chem., 1986.

[3]: W. Duch, and K. Jacek. Symmetric Group Graphical Approach to the Configuration Interaction Method. Int. J. Quantum Chem., 1982.

[4]: T. Helgaker, P. Jorgensen, and J. Olsen. Molecular Electronic-Structure Theory. Wiley, 2014.

