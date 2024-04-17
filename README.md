This repository contains a library of functions to solve Fredholm's integral equation of the 1st kind and Abel integral equation, two example driver programs, and the user manuals.

Files in this repo:

prgrad_reg.c - the library

prgrad_reg_common.c - equation kernels (see user manuals).

abel_prgrad_test.c - example driver program for Abel's equation.

fredh_prgrad_test.c - same for Fredholm's equation.

Makefile - makefile to compile the driver programs.

test_compact.fredh - the solution of the test data (define within the code of fredh_prgrad_test) on the set of compact functions without Tikhonov's regularization.

test_reg_W21.fredh - the solution of the same test data with Tikhonov's regilariztion in W21 space.

test_reg_W22.fredh - the solution of the same test data with Tikhonov's regilariztion in W22 space.

sim_power32.dat - test input file to use with abel_prgrad_test.c

sim_power32.abel - solution obtained from the input file.

abel_usage.pdf, fredholm_usage.pdf - user manuals to use the library.

Author contacts: Igor Antokhin, igor@sai.msu.ru
