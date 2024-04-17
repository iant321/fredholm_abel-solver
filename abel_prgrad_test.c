/*

 Driver program for solving Abel integral equation.
 For instructions to use, see comments in the code and my comments in abel_usage.pdf.

 I.Antokhin (igor@sai.msu.ru), Sept 2016

*/


#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <float.h>
#include <malloc.h>
#include <sys/resource.h>

#define NMAX    200  // max size of the grid on the right part of the eq. 
#define MMAX    200  // max size of the grid on the unknown function

#define C2MAX    1.0  // C2 limit.                                        

#define MAX_LEN  200

#define PI M_PI

// Function declarations.

// External functions.

// Allocation and freeing memory for 2D and 1D arrays.
extern double **matrix();
extern void free_matrix();
extern double *vector();
extern void free_vector();

extern void ptizr_proj();  // Solve a linear integral equation by minimization of the Tikhonov's functional, chooses regularization parameter by the principle of generalized residual.
extern void pticr0();      // Compute the matrix of the operator "a" in a*z=u.
extern void pticr3();      // Compute residual ||az-u0||^2.
extern void ptilrb();      // Compute constraints matrix and vector.
extern void kernel_abel();   // Compute the core of Abel equation.


extern double sqr();        // Square of argument.

void initialapprox( int switch_contype, double c2,  double *s, double *z,  int l, int n ); // Initial approximation.

double gasdev( void ); // Random Gaussian number generator.


int main( void ) {

char metric[5]; // Metric for ||z||^2

int i, l, m, n, imax_reg, imax_minim, iter_reg, iter_minim, verbose;

int icore;

int kernel_type;

int n_con;

int adjust_alpha;

int switch_contype;

int ierr;

double rs, ax;

double c2, h;

double alpha; // Regularization parameter.

double del2;   // Resulting deviation.
double delta2; // Deviation such that when the current one is below delta2, iterations stop.
double sumv, eps;

double sigma, delta0, norm_a;

double *s, *x, *u, *z, *z_exact, *az, *v, *sig_u;
double **a_abel, **con, *b;

char str[MAX_LEN+1];

FILE  *infile;

// The code below is for the simulated model 1 (power law with index 3/2).
  n = 100;
  m=100;

// Allocate memory for various arrays.
  if( !(s = vector(n) ) ) {
    printf("Error allocating memory for s\n" );
    exit(0);
  }
  if( !(z = vector(n) ) ) {
    free_vector(s);
    printf("Error allocating memory for z\n" );
    exit(0);
  }
  if( !(z_exact = vector(n) ) ) {
    free_vector(s);
    free_vector(z);
    printf("Error allocating memory for z_exact\n" );
    exit(0);
  }
  if( !(x = vector(m) ) ) {
    free_vector(s);
    free_vector(z);
    free_vector(z_exact);
    printf("Error allocating memory for u\n" );
    exit(0);
  }
  if( !(u = vector(m) ) ) {
    free_vector(s);
    free_vector(z);
    free_vector(z_exact);
    free_vector(x);
    printf("Error allocating memory for u\n" );
    exit(0);
  }
  if( !(sig_u = vector(m) ) ) {
    free_vector(s);
    free_vector(z);
    free_vector(z_exact);
    free_vector(x);
    free_vector(u);
    printf("Error allocating memory for sig_u\n" );
    exit(0);
  }
  if( !(v = vector(m) ) ) {
    free_vector(s);
    free_vector(z);
    free_vector(z_exact);
    free_vector(x);
    free_vector(u);
    free_vector(sig_u);
    printf("Error allocating memory for v\n" );
    exit(0);
  }
  if( !(az = vector(m) ) ) {
    free_vector(s);
    free_vector(z);
    free_vector(z_exact);
    free_vector(x);
    free_vector(u);
    free_vector(sig_u);
    free_vector(v);
    printf("Error allocating memory for az\n" );
    exit(0);
  }

// Read input data.

  if( !( infile = fopen( "sim_power32.dat", "r" ) ) ) {
    printf( "Input file does not exist! Exiting...\n" );
    exit( 0 );
  }

// Skip first 2 lines.
  for( i=0; i<2; i++ ) {
    fgets( str, MAX_LEN, infile );
  }

  for( i=0; i<n; i++ ) {
    if( fgets( str, MAX_LEN, infile ) != NULL ) {
      if( sscanf( str, "%lf %lf", &s[i], &z_exact[i] ) != 2 ) {
        printf( "Error reading exact solution, i= %d\n", i );
        exit( 0 );
      }
    }
  }

// Skip 1 line.
  fgets( str, MAX_LEN, infile );

// Read exact input data.
  m = 0;
  while(  fgets( str, MAX_LEN, infile ) != NULL ) {
    if( sscanf( str, "%lf %lf", &x[m], &u[m] ) != 2 ) {
      printf( "Error reading exact solution, i= %d\n", i );
      exit( 0 );
    }
    m++;
  }

  fclose( infile );

  printf( "n= %d m= %d\n", n, m );

// End reading data.

/*
 Add gaussian noise scaled as sqrt(u) and normalized so that the relative error of u(0) is equal to delta0.
*/
  delta0 = 0.01; // Relative error of u[0];
  norm_a = 1.0/delta0/u[0]/delta0; // Scaling factor. sigma_gauss = sqrt(norm_2*u)
  srandom( 100 );
  for( i=0; i<m; i++ ) {
    sigma = sqrt(fabs(u[i])/norm_a); // Standard deviation.
    u[i] += sigma*gasdev(); // Gaussian random with mean=0 and sigma (gasdev gives a random number with mean=0 and sigma=1).
  }

/*
 Compute data error delta2 (see my comments in abel_usage.pdf).
 For real data, calculate delta2 similarly, but using real measured sigma for every data point.
*/
  delta2 = 0.0;  // Data noise squared.
  for( i=0; i<m-1; i++ ) {
    sigma = sqrt(fabs(u[i])/norm_a); // Standard deviation.
    delta2 += sigma*sigma*(x[i+1]-x[i]);
  }

// Compute the sum of weights of input data points. In this test example,
// weights of all input data points are equal. For real data, compute them,
// e.g. from sigma of data points.

  sumv = 0.0;
  for( i=0; i<m; i++ ) {
    v[i] = 1.0;            // In this example all weights are equal.
    if( i < m-1) {
      sumv += v[i]*(x[i+1]-x[i]);
    }
  }
  sumv = sumv/(x[m-1]-x[0]);

  printf( "sumv= %le delta2= %le\n", sumv, delta2 );

// Start algorithm.

// Some parameters of ptizr_proj are specific to solving Fredholm's
// equation. I have to provide them here but they are not used inside ptizr_proj.

// Set algorithm parameters.

  kernel_type = 3;         // Use kernel for Abel equation.
  c2 = C2MAX;              // Needed only to set initial approximation of z.
  rs = 0.0;                // Not used in Abel equation (see pticr0).
  ax = 0.0;                // Not used in Abel equation.
  eps = 0.01;              // Threshold for choosing alpha (solving ||Az-u||^2=delta2). Accept alpha such that ||Az-u||^2-delta2 < eps*delta2.
  h = 0.0;                 // Uncertainty of the A operator.
  adjust_alpha = 0;        // Whether to search for optimal alpha in Tikhonov's regularization (1- search, 0 - solve at fixed alpha).
  alpha = 0.0;             // Either fixed value of alpha or its initial value depending on adjust_alpha).
  strcpy( metric, "W21" ); // Metric for z in Tikhonov's regularization ( "L2", "W21", or "W22" ).
  icore = 0;               // Must be set to 0 for Abel equation (see ptilr6). Non-zero values are only used with kernel_type=1 when solving Fredholm equation.
  
  imax_minim = 2000;       // Max number of conjugate gradient iterations within one regularization step.
  imax_reg = 1000;         // Max number of Tikhonov's regularization iterations.
  verbose = 0;             // Switch on/off (1/0) some additional stdout printing in ptizr_proj.

/*
  switch_contype:

  All constraints restrict z to be monotonically non-increasing and non-negative. 
  
 = 1 - monotonic (i.e. just the above constraint)
 = 2 - concave z" <= 0
 = 3 - convex  z" >= 0
 = 4 - concave-convex
 = 5 - non-negative (this is to use Tikhonov's regularization only) - used only with Abel eq., that is kernel_type=3
*/

/*
 Below is an example for a convex function. In the case of a concave-convex
 function with the unknown position of the inflection point this position
 can be found by e.g. the golden section method.
*/

  switch_contype = 3;

// There is no inflection point in this case, set l to an arbitrary value.
  l = 1;

// Get apriori constraints. icore for Abel equation must be set to 0.
// VERY IMPORTANT!!! Before first call to ptilrb, n_col MUST be set to 0!!! See comments in the function.
  n_con = 0; // The number of constraints (computed in ptilrb).
  ptilrb( kernel_type, switch_contype, &n_con, &con, &b, n, c2, icore, l, &ierr );

  if( ierr == 202 ) {
    printf("Memory allocation error in ptilrb. Exiting.\n" );
    exit( 0 );
  }

// Set initial approximation of z.
  initialapprox( switch_contype, c2, s, z, l, n );

// Call the main function.
  ptizr_proj( kernel_type, switch_contype, n_con, con, b, rs, u, v, sumv, s, x, x[0], x[m-1], 
              n, m, z, c2, delta2, eps, h, adjust_alpha, &alpha, metric, l, icore, ax, 
              &del2, imax_reg, &iter_reg, imax_minim, &iter_minim, verbose, &ierr );

  printf( "del2= %le iter_reg= %d iter_minim= %d alpha= %le\n", del2, iter_reg, iter_minim, alpha );
  printf( "        s            z_exact         z_model\n" );
  for( i=0; i<n; i++ ) {
    printf( "%15.7le %15.7le %15.7le\n", s[i], z_exact[i], z[i] );
  }

// Compute Az (model u(x)).

// Allocate memory for the Abel eq. kernel.
  if( !( a_abel = matrix( m, n ) ) ) {
    printf( "Memory allocation error for a_abel. Exiting...\n" );
    exit( 0 );
  }
  kernel_abel( a_abel, s, x, n, m );

// Compute model u (az).
  pticr3( a_abel, z, az, n, m );
  
  printf( "      x          u         az(model u)\n" );
  for( i=0; i<m; i++ ) {
    printf( "%le %le %le\n", x[i], u[i], az[i] );
  }

// Free memory.
  free_matrix( a_abel, m );  // Mandatory.
  free_matrix( con, n_con );
  free_vector( b );

  free_vector(s);            // If these arrays were allocated dynamically above.
  free_vector(z);
  free_vector(z_exact);
  free_vector(x);
  free_vector(u);
  free_vector(sig_u);
  free_vector(v);
  free_vector(az);

  exit( 0 );

}

void initialapprox( int switch_contype, double c2, double *s, double *z, int l, int n ) {
/*
  Initial approximation of z.
*/

int i;
double smax;
double cnst;


  switch( switch_contype ) {

    case 1:
// Linear monotonically decreasing function.
      for( i=0; i<n; i++ ) {
        z[i] = c2*(n-1-i)/(double)(n-1);
      }
      break;

    case 2:
// Concave function f" <= 0.
      smax = s[n-1]*1.001;
      for( i=0; i<n; i++ ) {
        z[i] = 0.9*c2*( 1.0-s[i]*s[i]/smax/smax );
      }
      break;

    case 3:
// Convex function f">=0. z~1/r^2, z(s0)=c2.
// To avoid NaN at s=0, shift the argument.

      cnst = 0.01*c2; // Depends also on scale of s.  Det to a value such that
                      // initial approximation is reasonable - not too big and not too small.

      for( i=0; i<n; i++ ) {
        z[i] = 0.9*c2*cnst*cnst/(s[i]+cnst)/(s[i]+cnst);
      }
      break;

    case 4:
// Concave-convex function. Inflation point is at index l.
      for( i=0; i<n; i++ ) {
        if( i <= 2*l ) {
          z[i] = c2/2.0*(cos( (double)i/(2.0*l)*PI )+1.0);
        } else {
          z[i] = 0.0;
        }
      }
      break;

    case 5: // Non-negative function, so can use any of the above, use the same as for case 3.
// Convex function f">=0. z~1/r^2, z(s0)=c2.
// To avoid NaN at s=0, shift the argument.

      cnst = 0.01*c2; // Depends also on scale of s.  Det to a value such that
                      // initial approximation is reasonable - not too big and not too small.

      for( i=0; i<n; i++ ) {
        z[i] = 0.9*c2*cnst*cnst/(s[i]+cnst)/(s[i]+cnst);
      }
      break;


    default:
      printf( "Wrong switch_contype!!!\n" ); // Set return error code, free memory in main program!!!
      break;
  }

  return;

}

double gasdev( void ) {
// Code from Numerical recipies, I replaced their ran1 by system random.
// Returns random number distributed normally, with mean=0 and sigma=1.
// To get gauss(mean,sigma), multiply the output by sigma and add mean.

// I use system-provided function "random" despice Num Rec warn that the
// user must be very careful, as often such functions are badly implemented
// with a linear algorithm, and further, RAND_MAX is often too low (e.g.
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
