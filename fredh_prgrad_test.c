/*
  Solving Fredholm integral equations of the first kind.

  Various a-priori constraints can be applied to the unknown fucntions. If a
  given set of constraints defines a compact set, Tikhonov's regularization
  is optional. You can switch it on/off. If not (or you don't have any
  constraints apart from smoothness), you must use the regularization.
  Minimization of the residual functional is performed by projections of
  conjugate gradients. In case the Tikhonov's regularization is used, the
  regularization parameter can be choosen according to the principle of
  generalized residual.

  Author: Igor Antokhin (igor@sai.msu.ru)
  Last change: Feb 2017

*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <float.h>
#include <malloc.h>

#define PI M_PI

// Function declarations.

double gasdev( void ); // Random Gaussian number generator.

// External functions (from prgrad_reg.c).

extern void ptizr_proj();  // Solve a Fredholm's equation by minimization of the Tikhonov's functional, chooses regularization parameter by the principle of generalized residual.
extern void pticr0();      // Compute the matrix of the operator "a" in a*z=u.
extern void pticr3();      // Compute residual ||az-u0||^2.
extern void ptilrb();      // Compute the matrix and vector of a priori constraints.

extern double **matrix();   // Allocate memory for a double 2D array.
extern void free_matrix();  // Free memory for a double 2D array.
extern double *vector();    // Allocate memory for a 1D array.
extern void free_vector();  // Free memory for 1D array.

int main( void ) {

  int n, m, l, maxiter_reg, maxiter_minim, adjust_alpha;

  int i, errorcode, ierr, verbose, iter_reg, iter_minim;
  
  int icore;
  
  char metric[20];

  int kernel_type, switch_contype;

  double alpha, h, eps, dx, delta2;

  double ds, del2, c2, sumv, sigma, tmp;
  
  double ax, rstar;
  
  double s1, s2, x1, x2;

// Input data.
  double *x, *u, *u0, *v;

// Model, eq. core.
  double *z, *z0, *s, **a, *az;

// Constraints matrix and vector.
  double **con = NULL, *b = NULL;
  int n_con;

/*
  Set all pointers to NULL. This is to avoid undefined behaviour of free()
  function. If I allocate memory for some arrays and fail to allocate one of
  them, I have to free all memory allocated. But I do not know which arrays
  were allocated and which not. The readinputdata() function only returns
  the errorcode ierr indicating that the memory allocation problem occured.
  of course I could return the list of arrays allocated but it is simplier
  to just initialize all pointers to NULL and then call free() for all of
  them. In case of the NULL argument free(NULL) does nothing.
*/

  x = u = u0 = v = z = z0 = s = az = NULL;

  a = NULL;

// Set dimensions on x and s vars.
  n = 101; // Size of the grid on the argument of the unknown function.
  m = 41;  // Size of the grid on input data.

// Allocate memory for various arrays.
  if( !(x = vector( m )) || !(u = vector( m )) || !(u0 = vector( m )) || !(v = vector( m )) ||
      !(z = vector( n )) || !(z0 = vector( n )) || !(s = vector( n )) || !(az = vector( m )) ||
      !(a = matrix( m, n )) ) {
    free_vector( x );
    free_vector( u );
    free_vector( u0 );
    free_vector( v );

    free_vector( z );
    free_vector( z0 );
    free_vector( s );
    free_vector( az );

    free_matrix( a, m );

    printf( "Error allocating memory for working arrays. Exiting.\n" );

    exit( 0 );
  }

// Set various parameters defining the behaviour of the algorithm.
  verbose = 1;
 
// Set grid on s from s1 to s2.
  s1 = 0.0;
  s2 = 1.0;
  ds = (s2-s1)/(n-1);
  for( i=0; i<n; i++ ) {
    s[i] = s1+ds*i;
  }

// ==== Create a simulated model containing Gaussian noise. ===

// Set the exact solution z0.

// Convex:
//    z0[i] = 1.0-ss[i]*ss[i];

// Concave-convex z with icore=0, inflation point l=40.
  l = 40; // Index of inflection point.
  for( i=0; i<n; i++ ) {
    if( i <= 2*l ) {
      z0[i] = 3.0/2.0*( cos(PI*(double)i/2.0/(double)l)+1.0 ); // z0(0)=3.0
    } else {
      z0[i] = 0.0;
    }
  }

// Compute the exact input data u0 for the test function z0.

// Set a grid on the argument of u.

  x1 = 0.0;
  x2 = 1.0;
// In this test model I set an even grid on the argument of u. For real data
// the grid may be uneven so you will have to use dx=x[i+1]-x[i] in all
// places below which use dx.
  dx = (x2-x1)/(m-1);

// I set weights below to 1. For real data, you may set them to be e.g.
// v[i]=1/sigma[i]^2, where sigma is the error of input data points.

  sumv = 0.0;
  for( i=0; i<m; i++ ) {
    x[i] = x1+dx*i; // Grid knots.
    v[i] = 1.0;                // Weights.
    sumv += v[i]*dx;
  }
  sumv = sumv/(x2-x1);
              
// Compute the A operator matrix.
  pticr0( 4, a, s, x, n, m );

// Compute the exact right-hand part of Fredholm eq. u0.
  pticr3(a, z0, u0, n, m);

// Add Gaussian noise to u0, save to u.
  delta2 = 0.0;  // The uncertainty of input data (see my pdf comments).
  sigma = 0.01; // Sigma of gaussian noise added to simulated input data.
  
  for( i=0; i<m; i++ ) {
    tmp = sigma*gasdev();
    u[i] = u0[i]+tmp;
    delta2 += tmp*tmp*dx;
  }

// === Finished creating the simulated model. ===

// Solve the simulated data.

// Set initial approximation of z. Generally it is better to avoid initial
// approximation to be at the border of a region defined by a priori
// constraints (no constraints should be active for initial approximation).
// Below I provide the "initialapprox" function which sets appropriate
// initial functions located strictly within the region defined by
// constraints of various types. You may use it for real data. However, in
// this test case, my unknown function is a cosine, and the initial
// approximation for the concave-convex case in the above function is also a
// cosine. So using the above function would make it too easy for the
// algorithm to come to the solution. This is why IN THIS CASE ONLY(!) I set
// the initial approximation of z to be a constant, so most a priori
// constraints are initially active.

  for( i=0; i<n; i++ ) {
    z[i] = 1.0;
  }

// Set other parameters. Note that some of the parameters are specific to
// the WR+O case (used when computing the corresponding kernels) and will not
// be used with your own kernels. Just set them to an arbitrary value. These
// parameters are:
// rstar, ax.

  kernel_type = 4; // For the test case. May also be 1,2 for WR+O systems, 3 for Abel equation.

// Type of a priori constraints on z:
// 1 - non-negative, monotonically non-increasing
// 2 - 1 + concave (z"<=0)
// 3 - 1 + convex  (z">=0)
// 4 - 1 + concave-convex (at i<l z"<=0, at i>l z">=0)
// 5 - non-negative

// The contypes 1-4 define compact sets, so you may solve Fredgolm eq. with
// or without Tikhonov's regularization. The contype 5 is not a compact set
// so in this case you must use Tikhonov's regularization.

  switch_contype = 4;

  rstar = 0.0; // Not used in this test case.
  ax = 0.0; // Not used in this test case.

  icore = 0; // If >0, fixes z[i<=icore]=c2 (see my pdf notes). In this test case, set to 0.
  c2 = 0.0;  // Fixes z[0]=c2 if icore>0 and kernel_type=1 (see my pdf notes).
             //  Not used in this test case.

  eps = 0.01; // See my pdf notes.
  h = 0.0;

// Regularization control:

// Reg. on, search for optimal alpha
  alpha = 1.0e-6;
  adjust_alpha = 1;

// Reg. on, solve at fixed alpha.
//  alpha = 1.0e-6;
//  adjust_alpha = 0;

// Reg. off.
//  alpha = 0.0;
//  adjust_alpha = 0;

  strcpy( metric, "W22" ); // Used when Tikhonov's regularizarion switched on. May also be  "L2" and "W22".
  
  maxiter_reg = 1000;   // See my pdf notes.
  maxiter_minim = 1000;

// Compute the constraints matrix and the right-hand part of constraints con and b.
  ptilrb( kernel_type, switch_contype, &n_con, &con, &b, n, c2, ds, icore, l, &ierr );

  if( ierr >=200 ) {

    printf( "Error in ptilrb: ierr= %d. Exiting.\n", ierr );

    free_vector( x );
    free_vector( u );
    free_vector( u0 );
    free_vector( v );

    free_vector( z );
    free_vector( z0 );
    free_vector( s );
    free_vector( az );

    free_matrix( a, m );

    free_vector( b );
    free_matrix( con, n_con );

    exit( 0 );  
  }
  
// Call the function to solve Fredholm equation.

// Note that I use the index of the inflection point l which was defined
// above when computing the test model (direct problem). In other words,
// when solving the equation, I use a correct l. For comments on how to
// search for l if it is unknown, see my pdf notes.

  ptizr_proj( kernel_type, switch_contype, n_con, con, b, rstar, u, v, sumv, s,
              x, x1, x2, n, m, z, c2, delta2, eps, h, adjust_alpha, &alpha, metric,
              l, icore, ax, &del2, maxiter_reg, &iter_reg, maxiter_minim, &iter_minim,
              verbose, &errorcode );

  if( errorcode >= 200 ) {
    printf( "Error in ptizr_proj: errorcode= %d. Exiting.\n", errorcode );

    free_vector( x );
    free_vector( u );
    free_vector( u0 );
    free_vector( v );

    free_vector( z );
    free_vector( z0 );
    free_vector( s );
    free_vector( az );

    free_matrix( a, m );

    free_vector( b );
    free_matrix( con, n_con );

    exit( 0 );  
  }

// Compute Az for the solution z.
  pticr3( a, z, az, n, m );

// Print out the results.

  printf( "Simulated input data (sigma of Gaussian noise sig= %lf):\n", sigma );
  printf( "       x      u0(exact)   u(with noise)     az\n" );
  for( i=0; i<m; i++ ) {
    printf( "%12.5lf %12.5le %12.5le %12.5le\n", x[i], u0[i], u[i], az[i] );
  }

  printf( "Solution of simulated input data:\n" );
  printf( "kernel_type= %d switch_contype= %d delta2= %le adjust_alpha= %d alpha= %le metric= %s\n",
          kernel_type, switch_contype, delta2, adjust_alpha, alpha, metric );
  printf( "iter_reg= %d  iter_minim= %d  del2= %le errorcode= %d\n", iter_reg, iter_minim, del2, errorcode );
  printf( "       s       z0(exact)     z(model)\n" );
  for( i=0; i<n; i++ ) {
    printf( "%12.5lf %12.5le %12.5le\n", s[i], z0[i], z[i] );
  }

// Free memory.
  free_vector( x );
  free_vector( u );
  free_vector( u0 );
  free_vector( v );

  free_vector( z );
  free_vector( z0 );
  free_vector( s );
  free_vector( az );

  free_matrix( a, m );

  free_vector( b );
  free_matrix( con, n_con );
  
  exit(0);

}

double gasdev( void ) {
// Code from Numerical recipies, I replaced their ran1 by system random.
// Returns random number distributed normally, with mean=0 and sigma=1.
// To get gauss(mean,sigma), multiply the output by sigma and add mean.

// I use system-provided function "random" despite Num Rec warn that the
// user must be very careful, as often such functions are badly implemented
// with linear algorithm, and further, RAND_MAX is often too low (e.g.
// 32767). In Fedora/Linux, though, the algorithm is non-linear and
// RAND_MAX=2147483647.

  static int iset=0;
  static double gset;
  double fac,r,v1,v2;
  double ran1();

  if  (iset == 0) {
    do {
      v1=2.0*random()/(double)RAND_MAX-1.0;
      v2=2.0*random()/(double)RAND_MAX-1.0;
      r=v1*v1+v2*v2;
    } while (r >= 1.0);
    fac=sqrt(-2.0*log(r)/r);
    gset=v1*fac;
    iset=1;
    return v2*fac;
  } else {
    iset=0;
    return gset;
  }
}

void initialapprox( int switch_contype, double c2, int l, double *z, int n ) {

int i;

  switch( switch_contype ) {
  
    case 1: // Monotonically non-increasing function. Does not use l.
        
      for( i=0; i<n; i++ )
        z[i] = c2*(1.0-1.0/(n-1)*i);
      break;
            
    case 2: // Concave function. Does not use l.
            // It is a cosine function dropping from c2 at i=0 to zero at n-1.

      for( i=0; i<n; i++ ) {
        z[i] = c2*cos( PI/2.0 * (double)i/(double)(n-1) );
      }
      break;
      
    case 3: // Convex function. Does not use l.
            // It is a cosine function dropping from c2 at i=0 to zero at n-1.
                  
      for( i=0; i<n; i++ ) {
        z[i] = c2 * ( cos( PI/2.0 * (double)(i+n-1)/(double)(n-1) ) + 1.0 );
      }
      break;
      
    case 4: // Concave-convex function.
            // l - inflation point.
            // If I want z[n-1]=0, l must be smaller than (n-1)/2.
            // It is a cosine function dropping from c2 at i=0 to c2/2 at l and to 0 at 2l.

      for( i=0; i<n; i++ ) {
        if( i <= 2*l ) {
          z[i] = c2/2.0 * ( cos( PI * (double)i/2.0/(double)l ) + 1.0 );
        } else {
          z[i] = 0.0;
        }
      }

      break;

    default:
      printf( "Wrong switch_contype!!!\n" ); // Set return error code, free memory in main program!!!
      break;
  }

  return;

}
