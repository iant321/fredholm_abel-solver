/*
  This code contains functions which solve Fredholm's equations of the 1st
  kind and Abel integral equatiob by the method of projection of conjugate
  gradients and uses the Tikhonov's regularization method.
*/

#include <stdlib.h>
#include <stdio.h>
#include <malloc.h>
#include <math.h>
#include <string.h>
#include <float.h> // This is where DBL_MAX constant is kept.

#define NTEST   1001  // Max size for various test arrays which I do not allocate dynamically. IMPORTANT!!!
                      // This number MUST be equal or larger than NMAX, MMAX constants in the calling program
                      // wr_prgrad or abel_prgrad!!!

// Equations kernels are declared here.
extern void kernel1();
extern void kernel2();
extern void kernel_abel();
extern void kernel_test();

// Spline approximation
extern void spline_approx();

/*
  Functions which allow me to allocate/free one- and two-dimensional arrays
  dynamically. The problem here is that I have to pass 2D arrays as function
  arguments and within such a function I want to use convenient notation
  like a[i][j] and not some form of (*a)[i*ncols+j]. Yet I do not know the
  number of columns in advance so I cannot declare the formal argument "a" as
  e.g. double a[][NCOLS]. The functions below allow me to get around this problem.
*/

double **matrix(int nrows, int ncols);
void free_matrix(double **m, int nrows);
double *vector( int n );
void free_vector( double *v );
int *int_vector( int n );
void free_int_vector( int *v );

/*
  Standard functions common for all methods of solving the Fredholm's eq. from the book of 
  Goncharsky, Cherepashxhuk, Yagola. See comments in the code of the functions.
*/

void pticr0( int kernel_type, double **a, double *x, // The function is changed compared to the original.
             double *y, double r, double ax, int n, int m );

void pticr1( double *b, double *c, double p, double *a, int n );
void pticr2( double *a, double r, int n );
void ptici2( int *a, int r, int n );
void pticr3( double **a, double *z, double *u, int n, int m );
void pticr4( double **a, double *u, double *u0, double *v, double *hy, double sumv, double *g, int n, int m );
void pticr5( double *a, double *b, double *v, double sumv, double *hy, int m, double *s );
void pticr6( double *a, double *b, int n, double *s );
void pticr7( double *a, double *b, int *v, int n, double *s );
void pticr8( double *z, double alpha, double hs, double *gr, int n, char *metric );
void pticr8_new( double *z, double alpha, double hs, double *gr, int n, char *metric, int *iend );
void pticr9( double an2, double *z, int n, double alpha, double hs, double *as2, char *metric );
void deriv4( double *x, int n, double h, double *deriv );

/*
  Functions ptizr* are to solve the Fredholm's equation of the 1st kind by
  the regularization method of Tikhonov. The choice of the regularization
  parameter alpha is done by the principle of generalized discrepancy. The
  functions are DIFFERENT from those presented in the original books, even
  if they have identical names. In my case, I have some additional a-priory
  constraints on the unknown functions:
  
  1.They are moonotonically non-increasing;
  2.They are convexo-concaved.
  3.In the primary minimum of the light curve (WR in front), the value Ia(0) is fixed.
  
  All these constraints do not allow me to use the original form of the
  functions as those are using the only a-priori constraint that the
  unknown fuction is non-negative.

  So instead of using the original function ptizr1 I use ptilr1. ptilr1 and
  the functions it calls are also modified to make use of the regularization
  method (see the code and the comments below).
*/

void ptizr_proj( int kernel_type, int switch_contype, int n_con, double **con, double *b, double rstar, 
                 double *u0, double *v, double sumv, double *x, double *y, double ymin, double ymax, 
                 int n, int m, double *z, double c2, double dl2,  double eps, double h, int adjust_alpha, 
                 double *alpha, char *metric, int l, int icore, double ax, double *eta, 
                 int imax_reg, int *iter_reg, int imax_minim, int *iter_minim, int verbose, int *ierr );

void ptizra( double an2, double z[], int n, double dl, double dh, double hs, double *an4 );

/*
  The functions ptilr* are to minimize a quadratic(!) functional by the
  method of projection of conjugate gradients.

  The functions ptilr, ptilr0, ptilr1, ptilr6, ptilrb, pticr0 are modified
  compared to their original versions to adapt them to my particular problem
  and to make use of the regularization method.

  Note that when using regularization, I do not use ptilr. use ptizr
  instead. ptilr is used when solving the problem on the compact class of
  functions without regularization.
*/

int ptilr0( double **a,  int n );

/*
void ptilr(int kernel_type, double rstar, double u0[], double v[], double
           x[], double y[], int n, int m, double z[], double c2, double dl,
           int imax, int l, double *eta, int *iter, int *ierr );
*/
void ptilr1( int kernel_type, double **a, double *u0, double *v, double sumv, double *hy,
             int mn, double **con, double *b, int icore, int l, double hs, double del2,
             double dgr, int icm, double **c, double *d, double *z, int n, int n_con,
             double alpha, char *metric, double *eta, int *ici, int *ierr );

void ptilr3( double **aj, double **con, int n, int n_con, int *mask );

int ptilr4( double **aj, double **p, double **pi, int m, int n, int n_con );

void ptilr5( double **con, double *b, int *mask, int n, int *m,
             int n_con, double **pi, double **c, double *d,
             double *z, double alpha, double hs, int *k, int *iend, double dgr,
             double **a, double *u0, double *v, double sumv, double *hy, int mn, char *metric );

void ptilr6( int kernel_type, int icore, double **con, double *b, double *z, double **p, int *mask, 
             int n, int *m, int n_con, double **c, double *d, int *iend, double alpha, double hs, char *metric, int l );

//void ptilr6( int kernel_type, int icore, double **con, double *z, double **p, int *mask, 
//             int n, int *m, int n_con, double **c, double *d, int *iend, double alpha, double hs, char *metric );

void ptilr7( double *z, double **c, double *d, double *gr, int n );

//void ptilra( int kernel_type, double **a, double *u, double *v, double sumv, double *hy, double **c, double *d, int n, int m );
void ptilra( double **a, double *u, double *v, double sumv, double *hy, double **c, double *d, int n, int m );

void ptilrb( int kernel_type, int switch_contype, int *n_con, double ***con,
             double **b, int n, double c2, int icore, int l, int *ierr );

// Function declarations begin here.

double **matrix(int nrows, int ncols) {

// Allocate memory for a matrix of doubles.

int i, j;
double **m;
    
  if( !(m = (double **)malloc((unsigned)nrows * sizeof(double *))) )
    return NULL;
  for( i=0; i<nrows; i++ ) {
    if( !(m[i] = (double *)malloc((unsigned)ncols * sizeof(double))) ) {
      for( j=i-1; j>=0; j-- )
        free( (double *)m[j] );
      return NULL;
    }
  }  
  return m;
}

void free_matrix(double **m, int nrows) {
int i;
  for( i=nrows-1; i>=0; i-- )
    free( (double *)m[i] );
  free( (double *)m );
  return;
}

double *vector( int n ) {

// Allocate memory for a vector of doubles.

  double *v;
  if( !( v = (double *)malloc( (unsigned)n*sizeof(double) ) ) )
    return NULL;
  return v;
}

int *int_vector( int n ) {

// Allocate memory for a vector of integers.

  int *v;
  if( !(v = (int *)malloc( (unsigned)n*sizeof(int) ) ) )
    return NULL;
  return v;
}

void free_vector( double *v ) {
  free( (char *)v );
  return;
}

void free_int_vector( int *v ) {
  free( (char *)v );
  return;
}

void pticr0( int kernel_type, double **a, double *x, double *y, double r, double ax, int n, int m ) {
/*
  Compute the kernel of the Fredholm or Abel equation.

  This function is modified compared to the original one. The original one
  assumes that grids on x,y are even and the matrix "a" is calculated using
  a function which give the value of the kernel at the point (y,x). In my
  case, the grid on y is uneven. Also, I have the kernel functions kernel1 and
  kernel 2 which compute the whole matrix "a" at once. They need x and y as
  arrays. Also, the kernels for the primary and secondary minima are
  different, that's why the kernel_type parameter.

  Important: I include step of the grid on x (hx) in the matrix "a" so in
  functions using a to compute e.g. discrepancy functional etc. I do not
  have to multiply the sums by hx.

  Conversion of the intergal eq.

     x2
  integral[ ak(y,x)*z(x)*dx ] = u0(y),  y=y[0],...y[m-1]
     x1

  to a linear system. Rectangle formula for the above integral is used.

  Note that my grid on x is such that its knots atre located in the middle
  of the grid cells. That is, if the size of the grid cell is hx, my knots
  are 0.5hx, 1.5hx, .... For this reason in all formulae approximating
  various integrals I just multiply the integrated function by hx. I do not
  have to multiply f(a) and f(b) by hx/2 as was done in the original version
  of the program.
     
  kernel_type - kernel type number. 1,2 for Fredholm eq, 3 for Abel, 4 for test model.
  a[m,n]      - the matrix of the operator "a"
  x           - array containing the grid (even) on x (unknown function)
  y           - array containing the grid (uneven, phases of the light curve) on y (right-hand side of eq)
  r           - radius of the normal (O) star
  ax          - coefficient of the linear darkening for the O star disk
  n           - dimension of the grid on x
  m           - dimension of the grid on y
  kernel1,kernel2 - functions which compute ak above

  Integration occurs on the second variable ak(y,x)
*/

int i, j;
double hx;

  if( kernel_type == 1 )
    kernel1( a, x, y, r, ax, n, m );
  else if( kernel_type == 2 )
    kernel2( a, x, y, r, n, m );
  else if( kernel_type == 3 )
    kernel_abel( a, x, y, n, m );
  else if( kernel_type == 4 )
    kernel_test(a, x, y, n, m);
  else
    kernel_test(a, x, y, n, m);

  if( kernel_type != 3 ) { // In case 3 (Abel eq) I do not need hx in the kernel a.
    hx = x[1]-x[0]; // Grid step on x
    for( i=0; i<n; i++ ) {
      for( j=0; j<m; j++ ) {
        a[j][i] *= hx;
  // If the grid knots are located in the middle of the grid intervals, comment out the following two lines.
        if( i == 0 || i == n-1 )
          a[j][i] /= 2.0;
      }
    }
  }

  return;
}

void pticr1( double *b, double *c, double p, double *a, int n) {

// Compute a[i] = b[i]+p*c[i], i = 0,n-1.

int i;

  for( i=0; i<n; i++ ) {
    a[i] = b[i]+p*c[i];
  }
  return;
}

void pticr2( double *a, double r, int n ) {

// Fill array with a number.

int i;

  for( i=0; i<n; i++ ) {
    a[i] = r;
  }
  return;
}

void ptici2( int *a, int r, int n ) {

// Fill array with an integer.

int i;

  for( i=0; i<n; i++ ) {
    a[i] = r;
  }
  return;
}

void pticr3( double **a, double *z, double *u, int n, int m ) {
/*
  Multiplication of a matrix a[m,n] by the vector z[n]
  u[m] - resulting vector.
*/
int i, j;

  for( i=0; i<m; i++ ) {
    u[i] = 0.0;
    for( j=0; j<n; j++ ) {
      (u[i]) += a[i][j]*z[j];
    }  
  }
  return;
}

void pticr4( double **a, double *u, double *u0, double *v, double *hy, double sumv, double *g, int n, int m ) {
/*
  Compute the gradient of the discrepancy norm (norm of az-u0).

  This function is different from the original one as my grid on "u0" is
  uneven, and also the data points have varying weights. For mathematical
  details, see my notes on the algorithm in the "latex" directory.

  a[m][n] - operator's matrix
     u[m] - model light curve u=(a,z)
    u0[m] - right-hand side of the Fredholm's equation
     v[m] - weights of the "u0" points 
     y[m] - the grid on u0
     sumv - normalized integral of data point weights
     g[n] - resulting gradient
        n - size of grid on z
        m - size of grid on u0
*/

int i, j;

  for( j=0; j<n; j++ ) {
    g[j] = 0.0;
    for( i=0; i<m; i++ ) {
      (g[j]) += a[i][j]*(u[i]-u0[i])*v[i]*hy[i];
    }
    (g[j]) *= 2.0/sumv;
  }
  return;
}

void pticr5( double *az, double *u, double *v, double sumv, double *hy, int m, double *s ) {
/*
  Compute the discrepancy functional. This is a modified version of the
  original function, taking weights (v) into account. Another difference: in
  the original version, the grid in the right-hand side of the equation is
  even. Thus, in this function, only the sum of (az-u)^2 was computed. Then,
  in the calling routine, it is multiplied by the step size. In my case, the
  grid is uneven so I have to directly compute the norm using the rectangle
  approximation of the integral
  
  int = sum (az_i-u_i)^2*v_i*hy_i/sumv
  
  hy_i is defined as the interval from (y_i+y_(i-1))/2 to (y_i+y_(i+1))/2;
  for y_0 the first point is ymin, for y_(n-1) the last point is ymax. In the
  calling routine, ymin is set to cos(inclination), ymax is set by ThetaCrit.
  
  Input parameters:
  az,u[m] - input vectors (a*z and u0 in the calling routine)
  v    - weights of u
  sumv - integral of weights divided by integration interval
  hy   - array of steps in y
  m    - dimension of a,b,v,x
  s    - result

*/

int i;

  *s = 0.0;
  for( i=0; i<m; i++ ) {
    (*s) += (az[i]-u[i])*(az[i]-u[i])*v[i]*hy[i];
  }
  *s = *s/sumv;
  return;
}

void pticr6( double *a, double *b, int n, double *s ) {
/*
  Compute scalar product of vectors a and b of length n.
  s - result.
*/

int i;

  *s = 0.0;
  for( i=0; i<n; i++ ) {
    (*s) += a[i]*b[i];
  }
  return;
}

void pticr7( double *a, double *b, int *v, int n, double *s ) {
/*
  s is the scalar product of the vectors a and b of length n with weights ( or steps, depends on context) v.
  s - result.
*/

int i;

  *s = 0.0;
  for( i=0; i<n; i++ ) {
    (*s) += a[i]*b[i]*v[i];
  }
  return;
}

void pticr8_simpson( double *z, double alpha, double hs, double *gr, int n, char *metric ) {
/*
  Compute the gradient of the stabilizing term and add it to gr[n]
  grad(||z||^2) = 2z
  grad(||z'||^2 = -2z"
  grad(||z"||^2 = 2z^(4)

  The function is significantly reworked compared to the original version.
  This is an improved version in that the gradients are computed from the
  discrete approximations of the above norms, and the approximations are
  using trapezidal form, not rectangular, for gradients of ||z||^2 and ||z'||^2 and
  simpson's rule for grad||z"||^2 in the W22 metric.

  z[n] - the point
  alpha - regularization parameter.
  hs - step of the grid on "z"
  gr - result
  n - dimension of z
  metric - string denoting the metric used - "L2", "W21", "W22".
*/

int i;
double deriv4;
// double deriv2;

  if( alpha <= 0.0 ) {
    return;
  }

  if( strcmp( metric, "L2" ) == 0 ) {
    for( i=0; i<n; i++ ) {
      (gr[i]) += 2.0*alpha*z[i]*hs;
    }
  } else if( strcmp( metric, "W21" ) == 0 ) {

    for( i=0; i<n; i++ ) {
      (gr[i]) += 2.0*alpha*z[i]*hs;
      if( i>0 && i<n-1 ) {
        (gr[i]) += -2.0*alpha*(z[i-1]-2*z[i]+z[i+1])/hs; // hs because I have /hs^2*hs
      }
      if( i == 0 ) {
        (gr[i]) += -2.0*alpha*(z[1]-z[0])/hs;
      }
      if( i == n-1 ) {
        (gr[i]) += -2.0*alpha*(z[n-2]-z[n-1])/hs;
      }
    }

/*
// A bit more accurate approximation, identical to that in the W22 case below.
    for( i=0; i<n; i++ ) {

      if( i>=1 && i<=n-2 ) { // 2z
        (gr[i]) += 2.0*alpha*z[i]*hs;
      }
      if( i==0 || i==n-1) {
         (gr[i]) += alpha*z[i]*hs;
      }

      if( i>0 && i<=n-3 ) { // -2z"
        (gr[i]) += -2.0*alpha*(z[i+1]-2*z[i]+z[i-1])/hs; // hs because I have /hs^2*hs
        deriv2 = (z[i+1]-2*z[i]+z[i-1])/hs;
      }
      if( i==0 ) {
        (gr[i]) += -2.0*alpha*(z[1]-z[0])/hs;
      }
      if( i==n-2 ) {
        (gr[i]) += -alpha*(z[n-1]-3*z[n-2]+2*z[n-3])/hs;
      }
      if( i==n-1 ) {
        (gr[i]) += alpha*(z[n-1]-z[n-2])/hs;
      }

    }
*/

  } else if( strcmp( metric, "W22" ) == 0 ) {

    for( i=0; i<n; i++ ) {

      if( i>=1 && i<=n-2 ) { // 2z
        (gr[i]) += 2.0*alpha*z[i]*hs;
      }
      if( i==0 || i ==n-1) {
         (gr[i]) += alpha*z[i]*hs;
      }

/*
      if( i>0 && i<=n-3 ) { // -2z"
        (gr[i]) += -2.0*alpha*(z[i+1]-2*z[i]+z[i-1])/hs; // hs because I have /hs^2*hs
        deriv2 = (z[i+1]-2*z[i]+z[i-1])/hs;
      }
*/

      if( i==0 ) {
        (gr[i]) += -2.0*alpha*(z[1]-z[0])/hs;
      }
      if( i==n-2 ) {
        (gr[i]) += -alpha*(z[n-1]-3*z[n-2]+2*z[n-3])/hs;
      }
      if( i==n-1 ) {
        (gr[i]) += alpha*(z[n-1]-z[n-2])/hs;
      }

/*
// Compute the 4th derivation of z. Approximation by 5 points, 4th degree polinomial.
      deriv4 = 0.0;
      if( i>1 && i<n-2 ) {
        deriv4 = ( z[i+2]-4*z[i+1]+6*z[i]-4*z[i-1]+z[i-2] );
      }

// This approximation is from the condition that z is constant beyond the limits of the interval on s.
      if( i == 0 ) {
//        deriv4 =  ( z[2]-4*z[1]+3*z[0] ); // from main formula
        deriv4 =  ( 2*z[2]-5*z[1]+3*z[0] )/2.0; // from derivative of discrete approx, symmetric approx.
//        deriv4 =  ( z[2]-2*z[1]+z[0] ); // from derivative of discrete approx, left-interval approx.
      }
      if( i == 1 ) {
//        deriv4 = ( z[3]-4*z[2]+6*z[1]-3*z[0] ); // from main formula
        deriv4 = ( 2*z[3]-8*z[2]+11*z[1]-5*z[0] )/2.0; // from derivative of discrete approx, symmetric approx.
//        deriv4 = ( z[3]-4*z[2]+5*z[1]-2*z[0] ); // from derivative of discrete approx, left-interval approx.
      }
      if( i == n-2 ) {
//        deriv4 = ( -3*z[n-1]+6*z[n-2]-4*z[n-3]+z[n-4] ); // from main formula
        deriv4 = ( -5*z[n-1]+11*z[n-2]-8*z[n-3]+2*z[n-4] )/2.0; // from derivative of discrete approx, symmetric approx.
//        deriv4 = ( -3*z[n-1]+6*z[n-2]-4*z[n-3]+z[n-4] ); // from derivative of discrete approx, left-interval approx.
      }
      if( i == n-1 ) {
//        deriv4 = ( 3*z[n-1]-4*z[n-2]+z[n-3] ); // from main formula
        deriv4 = ( 3*z[n-1]-5*z[n-2]+2*z[n-3] )/2.0;  // from derivative of discrete approx, symmetric approx.
//        deriv4 = ( 2*z[n-1]-3*z[n-2]+z[n-3] );  // from derivative of discrete approx, left-interval approx.
      }
*/


// Compute the derivation of \int(z")^2ds. Approximation of the integral by simpson's rule.
      deriv4 = 0.0;

      if( i>=2 && i<=n-3 && (i%2)==0 ) { // Even indexes.
//        printf( "i even = %3d\n", i );
        deriv4 = 8.0/3.0*( z[i+2]-3*z[i+1]+4*z[i]-3*z[i-1]+z[i-2] );
      }

      if( i>=3 && i<=n-4 && (i%2) ) { // Odd indexes.
//        printf( "i odd = %3d\n", i );
        deriv4 = 4.0/3.0*( z[i+2]-6*z[i+1]+10*z[i]-6*z[i-1]+z[i-2] );
      }

// This approximation is from the condition that z is constant beyond the limits of the interval on s.
      if( i == 0 ) {
        deriv4 =  2.0/3.0*( 4*z[2]-9*z[1]+5*z[0] );
      }
      if( i == 1 ) {
        deriv4 = 2.0/3.0*( 2*z[3]-12*z[2]+19*z[1]-9*z[0] );
      }
      if( i == n-2 ) {
        deriv4 = 2.0/3.0*( -9*z[n-1]+19*z[n-2]-12*z[n-3]+2*z[n-4] );
      }
      if( i == n-1 ) {
        deriv4 = 2.0/3.0*( 5*z[n-1]-9*z[n-2]+4*z[n-3] );
      }


      deriv4 = deriv4/hs/hs/hs; //  h^-3 because I have /hs^4*hs

//      printf( "i= %3d d4= %17.10e d2= %17.10e\n", i, deriv4, deriv2 );
//      if( deriv4 > 4000.0 ) deriv4 = 4000.0;
      (gr[i]) += alpha*deriv4; // 2z^(4)

    }

  }
  return;
}

void pticr8_old( double *z, double alpha, double hs, double *gr, int n, char *metric ) {
/*
  Compute the gradient of the stabilizing term and add it to gr[n]
  grad(||z||^2) = 2z
  grad(||z'||^2 = -2z"
  grad(||z"||^2 = 2z^(4)

  The function is significantly reworked compared to the original version.

  z[n] - the point
  alpha - regularization parameter.
  hs - step of the grid on "z"
  gr - result
  n - dimension of z
  metric - string denoting the metric used - "L2", "W21", "W22".
*/

int i;
double deriv4;
// double deriv2;
// double xs[NTEST], zz[NTEST], z_sp[NTEST], weight[NTEST];
// int j, mbin, nsh;

  if( alpha <= 0.0 ) {
    return;
  }

  if( strcmp( metric, "L2" ) == 0 ) {
    for( i=0; i<n; i++ ) {
      (gr[i]) += 2.0*alpha*z[i]*hs;
    }
  } else if( strcmp( metric, "W21" ) == 0 ) {
    for( i=0; i<n; i++ ) {
      (gr[i]) += 2.0*alpha*z[i]*hs;
      if( i>0 && i<n-1 ) {
        (gr[i]) += -2.0*alpha*(z[i-1]-2*z[i]+z[i+1])/hs; // hs because I have /hs^2*hs
      }
      if( i == 0 ) {
        (gr[i]) += -2.0*alpha*(z[1]-z[0])/hs;
      }
      if( i == n-1 ) {
        (gr[i]) += -2.0*alpha*(z[n-2]-z[n-1])/hs;
      }
    }
  } else if( strcmp( metric, "W22" ) == 0 ) {

/*
// Compute spline-approximation of z
    for( i=1; i<n+1; i++ ) {
      xs[i] = i*hs;
      weight[i] = 1.0;
      zz[i] = z[i-1];
    }
    spline_approx( xs, zz, weight, n+1, z_sp );
//    printf( "!!!!!!!!!!\n" );
    for(i=0; i<n; i++ ) {
      z_sp[i] = z_sp[i+1];
    }      
*/

/*
// Compute moving average of z for smoothing it.
    mbin = 3;
    nsh = mbin/2;
    for( i=0; i<n; i++ ) {
      z_sp[i] = 0.0;
      if( i < nsh ) {
        for( j=1;j<=nsh-i; j++ ) {
          z_sp[i] += z[0];
        }
        for( j=0; j<=i+nsh; j++ ) {
          z_sp[i] += z[j];
        }
      } else if( i>=nsh && i<=n-1-nsh ) {
        for( j=i-nsh; j<=i+nsh; j++ ) {
          z_sp[i] += z[j];
        }
      } else {
        for( j=1; j<=i+nsh-(n-1); j++ ) {
          z_sp[i] +=z[n-1];
        }
        for( j=i-nsh; j<n; j++ ) {
          z_sp[i] +=z[j];
        }
      }
      z_sp[i] /= mbin;
    }
*/

/*
// Compute moving average of z for smoothing it.
    mbin = 3;
    nsh = mbin/2;
    for( i=0; i<n; i++ ) {
      if( i < nsh ) {
        z_sp[i] += z[i];
      } else if( i>=nsh && i<=n-1-nsh ) {
        z_sp[i] = 0.0;
        for( j=i-nsh; j<=i+nsh; j++ ) {
          z_sp[i] += z[j];
        }
        z_sp[i] /= mbin;
      } else {
        z_sp[i] +=z[i];
      }
    }
*/

/*
    printf( "mbin=%d nsh=%d\n", mbin, nsh );
    for( i=0; i<n; i++ ) {
      printf( "%3d %17.10e %17.10e %17.10e\n", i, z[i], z_sp[i], z[i]-z_sp[i] );
    }
*/

    for( i=0; i<n; i++ ) {
      (gr[i]) += 2.0*alpha*z[i]*hs;   // 2z

/*
      if( i>0 && i<n-1 ) { // -2z"
        (gr[i]) += -2.0*alpha*(z[i-1]-2*z[i]+z[i+1])/hs; // hs because I have /hs^2*hs
        deriv2 = (z[i-1]-2*z[i]+z[i+1])/hs;
      }

      if( i == 0 ) {
        (gr[i]) += -2.0*alpha*(z[1]-z[0])/hs;
        deriv2 = (z[1]-z[0])/hs;
      }
      if( i == n-1 ) {
        (gr[i]) += -2.0*alpha*(z[n-2]-z[n-1])/hs;
        deriv2 = (z[n-2]-z[n-1])/hs;
      }
*/

/*
// Compute the 4th derivation of z. Approximation by 5 points, 4th degree polinomial.
      deriv4 = 0.0;
      if( i>1 && i<n-2 ) {
        deriv4 = ( z[i+2]-4*z[i+1]+6*z[i]-4*z[i-1]+z[i-2] );
      }

// This approximation is from the condition that z is constant beyond the limits of the interval on s.
      if( i == 0 ) {
//        deriv4 =  ( z[2]-4*z[1]+3*z[0] ); // from main formula
        deriv4 =  ( 2*z[2]-5*z[1]+3*z[0] )/2.0; // from derivative of discrete approx, symmetric approx.
//        deriv4 =  ( z[2]-2*z[1]+z[0] ); // from derivative of discrete approx, left-interval approx.
      }
      if( i == 1 ) {
//        deriv4 = ( z[3]-4*z[2]+6*z[1]-3*z[0] ); // from main formula
        deriv4 = ( 2*z[3]-8*z[2]+11*z[1]-5*z[0] )/2.0; // from derivative of discrete approx, symmetric approx.
//        deriv4 = ( z[3]-4*z[2]+5*z[1]-2*z[0] ); // from derivative of discrete approx, left-interval approx.
      }
      if( i == n-2 ) {
//        deriv4 = ( -3*z[n-1]+6*z[n-2]-4*z[n-3]+z[n-4] ); // from main formula
        deriv4 = ( -5*z[n-1]+11*z[n-2]-8*z[n-3]+2*z[n-4] )/2.0; // from derivative of discrete approx, symmetric approx.
//        deriv4 = ( -3*z[n-1]+6*z[n-2]-4*z[n-3]+z[n-4] ); // from derivative of discrete approx, left-interval approx.
      }
      if( i == n-1 ) {
//        deriv4 = ( 3*z[n-1]-4*z[n-2]+z[n-3] ); // from main formula
        deriv4 = ( 3*z[n-1]-5*z[n-2]+2*z[n-3] )/2.0;  // from derivative of discrete approx, symmetric approx.
//        deriv4 = ( 2*z[n-1]-3*z[n-2]+z[n-3] );  // from derivative of discrete approx, left-interval approx.
      }
*/

// Compute the derivation of \int(z")^2ds. Approximation of the integral by simpson's rule.
      deriv4 = 0.0;

      if( i>=2 && i<=n-3 && (i%2)==0 ) { // Even indexes.
        deriv4 = 8.0/3.0*( z[i+2]-3*z[i+1]+4*z[i]-3*z[i-1]+z[i-2] );
      }

      if( i>=3 && i<=n-4 && (i%2) ) { // Odd indexes.
        deriv4 = 4.0/3.0*( z[i+2]-6*z[i+1]+10*z[i]-6*z[i-1]+z[i-2] );
      }

// This approximation is from the condition that z is constant beyond the limits of the interval on s.
      if( i == 0 ) {
        deriv4 =  2.0/3.0*( 4*z[2]-9*z[1]+5*z[0] );
      }
      if( i == 1 ) {
        deriv4 = 2.0/3.0*( 2*z[3]-12*z[2]+19*z[1]-9*z[0] );
      }
      if( i == n-2 ) {
        deriv4 = 2.0/3.0*( -9*z[n-1]+19*z[n-2]-12*z[n-3]+2*z[n-4] );
      }
      if( i == n-1 ) {
        deriv4 = 2.0/3.0*( 5*z[n-1]-9*z[n-2]+4*z[n-3] );
      }

      deriv4 = deriv4/hs/hs/hs; //  h^-3 because I have /hs^4*hs

// Compute the 4th derivation of z. Approximation by 7 points, 6th degree polinomial.
/*
      if( i>2 && i<n-3 ) {
        deriv4 = ( -z[i+3]+12*z[i+2]-39*z[i+1]+56*z[i]-39*z[i-1]+12*z[i-2]-z[i-3] );
      }
      if( i == 2 ) {
        deriv4 = ( -z[5]+12*z[4]-39*z[3]+56*z[2]-39*z[1]+11*z[0] );
      }
      if( i == 1 ) {
        deriv4 = ( -z[4]+12*z[3]-39*z[2]+56*z[1]-28*z[0] );
      }
      if( i == 0 ) {
        deriv4 =  ( -z[3]+12*z[2]-39*z[1]+28*z[0] );
      }
      if( i == n-3 ) {
        deriv4 = ( 11*z[n-1]-39*z[n-2]+56*z[n-3]-39*z[n-4]+12*z[n-5]-z[n-6] );
      }
      if( i == n-2 ) {
        deriv4 = ( -28*z[n-1]+56*z[n-2]-39*z[n-3]+12*z[n-4]-z[n-5] );
      }
      if( i == n-1 ) {
        deriv4 = ( 28*z[n-1]-39*z[n-2]+12*z[n-3]-z[n-4] );
      }
      deriv4 = deriv4/6.0/hs/hs/hs; //  h^-3 because I have /hs^4*hs

//      deriv4 = (deriv41+deriv42)/2.0;
*/    

//      printf( "i= %3d d4= %17.10e d2= %17.10e\n", i, deriv4, deriv2 );
//      if( deriv4 > 4000.0 ) deriv4 = 4000.0;
      (gr[i]) += alpha*deriv4; // 2z^(4)

    }

  }
  return;
}

void pticr8( double *z, double alpha, double hs, double *gr, int n, char *metric ) {
/*
  Compute the gradient of the stabilizing term and add it to gr[n]
  grad(||z||^2) = 2z*hs
  grad(||z'||^2 = -2z"*hs
  grad(||z"||^2 = 2z^(4)*hs

  The function is significantly reworked compared to the original version.

  In this version I compute the 4th derivative of z using int doubles.
  floats are 4 bytes, doubles are 8, int doubles are 16 bytes.

  z[n] - the point
  alpha - regularization parameter.
  hs - step of the grid on "z"
  gr - result
  n - dimension of z
  metric - string denoting the metric used - "L2", "W21", "W22".
*/

int i;
double deriv4; //, deriv2;

  if( alpha <= 0.0 ) {
    return;
  }

  if( strcmp( metric, "L2" ) == 0 ) {
    for( i=0; i<n; i++ ) {
      (gr[i]) += 2.0*alpha*z[i]*hs;
    }

  } else if( strcmp( metric, "W21" ) == 0 ) {

    for( i=0; i<n; i++ ) {
      (gr[i]) += 2.0*alpha*z[i]*hs;
      if( i>0 && i<n-1 ) {
        (gr[i]) += -2.0*alpha*(z[i-1]-2*z[i]+z[i+1])/hs; // hs because I have /hs^2*hs
      }

      if( i == 0 ) {
        (gr[i]) += -2.0*alpha*(z[1]-z[0])/hs;
      }
      if( i == n-1 ) {
        (gr[i]) += -2.0*alpha*(z[n-2]-z[n-1])/hs;
      }

    }

  } else if( strcmp( metric, "W22" ) == 0 ) {

//    printf( " i        z           deriv2         deriv4\n" );
    for( i=0; i<n; i++ ) {
      (gr[i]) += 2.0*alpha*z[i]*hs;   // 2z

      if( i>0 && i<n-1 ) { // -2z"
//        deriv2 = -2.0*alpha*(z[i-1]-2*z[i]+z[i+1])/hs;
        (gr[i]) += -2.0*alpha*(z[i-1]-2*z[i]+z[i+1])/hs; // hs because I have /hs^2*hs
      }

      if( i == 0 ) {
//        deriv2 = -2.0*alpha*(z[1]-z[0])/hs;
        (gr[i]) += -2.0*alpha*(z[1]-z[0])/hs;
      }

      if( i == n-1 ) {  // Comment out if discrete approx. = rectangular left.
//        deriv2 = -2.0*alpha*(z[n-2]-z[n-1])/hs;
        (gr[i]) += -2.0*alpha*(z[n-2]-z[n-1])/hs;
      }


// Compute the 4th derivation of z. Approximation by 5 points, 4th degree
// polynomial. Actually, the formulae below are direct derivatives of the
// formula approximating ||z"||^2 by the rectangular formula.
// n assumption is used that beyound the limits of integration, z is constant.
      if( i>1 && i<n-2 ) {
        deriv4 = 2*( z[i+2]-4*z[i+1]+6*z[i]-4*z[i-1]+z[i-2] );
      }
      if( i == 0 ) {
        deriv4 = 2*( z[2]-3*z[1]+2*z[0] ); // From direct differentiation of the discrete approximation of \int z"ds, constant beyond the edges.
//        deriv4 = 2*( z[2]-2*z[1]+z[0] ); // From direct differentiation of the discrete approximation of \int z"ds, free edges.
//        deriv4 = 2*( z[2]-4*z[1]+3*z[0] ); // From main formula, z constant beyond the edges.
//        deriv4 = 2*( z[2]-2*z[1]+z[0] ); // From main formula, free ends in z.
//        deriv4 = 0.0;
      }
      if( i == 1 ) {
        deriv4 = 2*( z[3]-4*z[2]+6*z[1]-3*z[0] ); // From direct differentiation of the discrete approximation of \int z"ds, constant beyond the edges.
//        deriv4 = 2*( z[3]-4*z[2]+5*z[1]-2*z[0] ); // From direct differentiation of the discrete approximation of \int z"ds, free ends.
//        deriv4 = 2*( z[3]-4*z[2]+6*z[1]-3*z[0] ); // From main formula, z constant beyond the edges.
//        deriv4 = 2*( z[3]-4*z[2]+5*z[1]-2*z[0] ); // From main formula, free ends in z.
//        deriv4 = 0.0;
      }
      if( i == n-2 ) {
        deriv4 = 2*( -2*z[n-1]+5*z[n-2]-4*z[n-3]+z[n-4] ); // From direct differentiation of the discrete approximation of \int z"ds, constant beyond the edges.
//        deriv4 = 2*( -2*z[n-1]+5*z[n-2]-4*z[n-3]+z[n-4] ); // From direct differentiation of the discrete approximation of \int z"ds, free ends.
//        deriv4 = 2*( -3*z[n-1]+6*z[n-2]-4*z[n-3]+z[n-4] ); // From main formula, z constant beyond the edges.
//        deriv4 = 2*( -2*z[n-1]+5*z[n-2]-4*z[n-3]+z[n-4] ); // From main formula, free ends in z.
//        deriv4 = 0.0;
      }
      if( i == n-1 ) {
        deriv4 = 2*( z[n-1]-2*z[n-2]+z[n-3] ); // From direct differentiation of the discrete approximation of \int z"ds, constant beyond the edges.
//        deriv4 = 2*( z[n-1]-2*z[n-2]+z[n-3] ); // From direct differentiation of the discrete approximation of \int z"ds, free ends.
//        deriv4 = 2*( 3*z[n-1]-4*z[n-2]+z[n-3] );  // From main formula, z constant beyond the edges.
//        deriv4 = 2*( z[n-1]-2*z[n-2]+z[n-3] );  // From main formula, free ends in z.
//        deriv4 = 0.0;
      }

      (gr[i]) += alpha*deriv4/hs/hs/hs; // 2z^(4)*hs

//      printf( "%3d %22.15e %22.15e %22.15e\n", i, z[i], deriv2, alpha*deriv4/hs/hs/hs );

    }

  }

  return;

}

void pticr9( double an2, double *z, int n, double alpha,
             double hs, double *as2, char *metric ) {
/*
  Compute stabilizing term and add it to an2.
  an2 - discrepancy ( ||az-u0||^2 )
  z[n] - the point
  alpha - regularization parameter
  hs - step of the grid on z  
  as2 - result.
*/

int i;
double s;

//  printf( "alpha=%e an2=%e\n", alpha, an2 );
  if( alpha <= 0.0 ) {
    *as2 = an2;
    return;
  }
  s = 0.0;

  if( strcmp( metric, "L2" ) == 0 ) {
    for( i=0; i<n; i++ ) {
      s += z[i]*z[i];
    }
  }
  
  if( strcmp( metric, "W21" ) == 0 ) {
    for( i=1; i<n; i++ ) {
      s += z[i]*z[i]+(z[i]-z[i-1])*(z[i]-z[i-1])/hs/hs;
    }
    s += z[0]*z[0];
  }

  if( strcmp( metric, "W22" ) == 0 ) {
    for( i=1; i<n-1; i++ ) {
      s += z[i]*z[i] + // \int z
           (z[i]-z[i-1])*(z[i]-z[i-1])/hs/hs + // \int z'
           (z[i+1]-2*z[i]+z[i-1])*(z[i+1]-2*z[i]+z[i-1])/hs/hs/hs/hs; // \int z"
    }

// Add terms at i=0,1,n-2,n-1. Must be consistent with approximations of
// gradients in pticr8, optimal step in ptilr5!!!
    s += z[0]*z[0] + z[n-1]*z[n-1] + // \int z
         (z[n-1]-z[n-2])*(z[n-1]-z[n-2])/hs/hs + // \int z', at z'(0)=0 so no term at zero index.
         (z[1]-z[0])*(z[1]-z[0])/hs/hs/hs/hs ;   // \int z" // If z is constant beyond the edges. For free ends, comment this line.
  }

  *as2 = an2+alpha*s*hs;
  return;
}

void deriv4( double *y, int n, double h, double *deriv ) {
/*
 Compute the 4th derivative of a tabulated function.
 y - input array of dimension n containing the function,
 h - step, deriv - its 4th derivative.
 I use the formulae from my notes to the method.
*/

int i;

  for( i=2; i<n-2; i++ ) {
    deriv[i] = ( y[i+2]-4*y[i+1]+6*y[i]-4*y[i-1]+y[i-2] ) /h/h/h/h;
  }
  deriv[1] = ( y[3]-4*y[2]+6*y[1]-3*y[0] ) /h/h/h/h;
  deriv[0] = ( y[2]-4*y[1]+2*y[0] ) /h/h/h/h;
  deriv[n-2] = ( -3*y[n-1]+6*y[n-2]-4*y[n-3]+y[n-4] ) /h/h/h/h;
  deriv[n-1] = ( 3*y[n-1]-4*y[n-2]+y[n-3] ) /h/h/h/h;
  return;
}

/*
  ptizr functions.
*/

void ptizr_proj( int kernel_type, int switch_contype, int n_con, double **con, double *b, double rstar, 
                 double *u0, double *v, double sumv, double *x, double *y, double ymin, double ymax, 
                 int n, int m, double *z, double c2, double dl2,  double eps, double h, int adjust_alpha, 
                 double *alpha, char *metric, int l, int icore, double ax, double *eta, 
                 int imax_reg, int *iter_reg, int imax_minim, int *iter_minim, int verbose, int *ierr ) {

/*
  Solving the Fredholm's equation of the 1st kind by the regularization
  method of Tikhonov. Choose the regularization parameter according to the
  principle of generalized residual. To minimize the Tikhonov's functional,
  use the projection method of conjugate gradients.

  Variables:
  kernel_type - kernel type, eq. kernels depend on this.
switch_contype - a switch defining the type of constraints, used in ptilrb.
        n_con - number of a-priori constraints
con[n_con][n] - apriori constraints matrix
     b[n_con] - right-hand side of apriori constraints
        rstar - a geometrical parameter - the raduis of the O star
        u0[m] - right-hand side of the Fredholm's equation
         v[m] - weights of u0
         sumv - integral of weights of the points on the observed light curve
         x[n] - grid on the unknown function z
         y[m] - grid on u0 (reduced orbital phases delta)
         ymin - minimal value of variable y (may be lower than y[0]
         ymax - maximal value of variable y (may be larger than y[m-1]
            n - size of grid on z
            m - size of grid on u0
         z[n] - unknown function
           c2 - the value of the unknown function at 0, in the primary (first) min

          dl2 - square of the error of the right-hand side of the equation.
                Meaning of dl2: the residual
                ||u_delta-u|| = sqrt( \int_c^d [u_delta(y)-u(y)]^2 dy ),
                where u - "true" right-hand side, u_delta - measured
                realization. In other words, if I have a light curve with
                error of one point delta (assume all errors identical), then dl2 = delta^2*(d-c).
          eps - accuracy of searching for the solution. Return if fabs(an4)<eps*dl2, where an4 - generalized residual.
            h - the error of the "a" operator in a*z=u0
 adjust_alpha - if =0, solve at the fixed value of alpha. If = 1 find alpha by the principle of 
                generalized residual - look for alpha such that |rho(alpha)|<= delta^2
        alpha - regularization parameter. If adjust_alpha=0, contains the value of the par.
                If adjust_alpha=1, this value is the initial alpha value.
                If < 0 - no regilarization.
       metric - metric of space used - L2, W21, or W22
            l - index of the inflation point. It the point is absent, l must be equal to 0
          icore - index of the last constraint which has the form of equality, plus 1. Must be >=0 and < l.
          eta - resulting residual
     imax_reg - maximal number of "regularization" iterations - when searching for initial bracketing point in alpha or in secant method.
     iter_reg - number of steps (iterations) in the secant method for searching alpha done
   imax_minim - maximal number of "big loop" iterations in ptilr1.
   iter_minim - number of big loop iterations done
         ierr - exit code; 1** - normal end, 2** - various errors
                = 100 - normal end, exact minimum is found
                = 101 - iterations finished by residual value
                = 102 - iterations finished by the norm of the gradient
                = 103 - alpha became equal to zero while doing iterations
                = 104 - when doing initial try at alpha=0, an4 >= eps*dl2, regularization impossible
                = 200 - initial approximation outside allowable range.
                = 201 - inconsistent adjust_alpha and alpha
                = 202 - errors allocating memory for working arrays
                = 203 - singular matrix of active constraints (when computing the projector)
                = 204 - initial alpha < 0
                = 205 - while searching for the interval on alpha containing the solution, 
                        made imax_reg multiplications by 2.0, residual still negative. 
                        Initial regularization parameter too small.
                = 206 - max number of "big loop" iterations reached, no solution
                = 207 - in secant method, imax_reg secant iterations done, still not reached the exit criteria.
                = 208 - kernel_type <1 or >4.

  Various variables used in the function:
  a[m][n]        - matrix of operator
  c[n][n] - c in the functional form (c*x,x)+(d,x)+e
  d[n]    - d in the functional form (c*x,x)+(d,x)+e
*/

  double **a, **c, *d, *hy, *z0;
  double dl, dh, hx, an4, x1, x2, x_new, f1, f2, as2;

  int i; //, j;
  
  if( *alpha < 0.0 ) {
    *ierr = 204;
    return;
  }

  if( adjust_alpha == 1 && *alpha <= 0.0 ) {
    *ierr = 201;
    return;
  }

// Number of constraints in the two minima of the light curve.
  if( kernel_type > 4 ) {
    *ierr = 208;
    return;
  }

//  printf( "ptizr_proj: n= %d n_con= %d m= %d\n", n, n_con, m );
// Allocate memory for various working arrays.
  if( !( a = matrix( m, n ) ) ) {  // Operator matrix.
    *ierr = 202;
    return;
  } else if( !( c = matrix( n, n ) ) ) { // c in (c*x,x)+(d,x)+e
    *ierr = 202;
    free_matrix( a, m );
    return;
  } else if( !( d = vector( n ) ) ) { //  // d in (c*x,x)+(d,x)+e
    *ierr = 202;
    free_matrix( a, m );
    free_matrix( c, n );
    return;
  } else if( !( hy = vector( m ) ) ) { // Array of steps in the grid on y.
    *ierr = 202;
    free_matrix( a, m );
    free_matrix( c, n );
    free_vector( d );
    return;
  } else if( !( z0 = vector( n ) ) ) { // Array of steps in the grid on y.
    *ierr = 202;
    free_matrix( a, m );
    free_matrix( c, n );
    free_vector( d );
    free_vector( hy );
    return;
  }

// Save z (when entering this function, z contains initial approximation).
// Use this approximation as a starting point in every call to ptilr1.
// Otherwize (if I use the previous solution as initial approximation), I
// get a mess since the solution may slightly violate apriori constraints.

  for( i=0; i<n; i++ ) {
    z0[i] = z[i];
  }

  dl = sqrt( dl2 ); // sqrt of the (observed) discrepancy
  dh = sqrt( h );   // sqrt of the uncertainty of the operator A in Az=u

  hx = x[1]-x[0]; // Step of the grid on stellar disk.

// Steps of the grid in y.

//  printf( "hy=\n" );
//  if( kernel_type == 1 || kernel_type == 2 ) {  // For Fredholm equation.
    for( i=0; i<m; i++ ) {
      if( i == 0 ) {
        hy[i] = (y[0]+y[1])/2.0-ymin;
      } else if( i == m-1 ) {
        hy[i] = (ymax-(y[m-2]+y[m-1])/2.0);
      } else {
        hy[i] = (y[i+1]-y[i-1])/2.0;
      }
//    printf( "%22.15e\n", hy[i] );

    }

// Get the operator matrix.
  pticr0( kernel_type, a, x, y, rstar, ax, n, m );

/*
  printf( "n= %d m=%d a=\n", n, m );
  for( i=0; i<m; i++ ) {
    for( j=0; j<n; j++ ) {
      printf( " %22.15e", a[i][j] );
    }
    printf( "\n" );
  }
*/

// Transform the discrepancy functional to the form (c*x,x)+(d,x)+e (I do not compute e, see comments in the ptilra code).
//  ptilra( kernel_type, a, u0, v, sumv, hy, c, d, n, m );
  ptilra( a, u0, v, sumv, hy, c, d, n, m );

/*
  printf( "c=\n" );
  for( i=0; i<n; i++ ) {
    for( j=0; j<n; j++ ) {
      printf( " %22.15e", c[i][j] );
    }
    printf( "\n" );
  }
  printf( "d=\n" );
  for( i=0; i<n; i++ ) {
    printf( " %22.15e", d[i] );
    printf( "\n" );
  }
*/

// Get the constraints matrix. IMPORTANT!!! - I now call it in the main program. Done in abel_prgrad, but not in wr_prgrad!!!
//  ptilrb( kernel_type, switch_contype, con, b, n, n_con, c2, hx, icore, l );

/*

  I have to either solve equation at a fixed alpha or find alpha such that
  generalized residual is equal to zero (more precisely, fabs(an4)<eps*dl2).

  an4 is monotonically increasing function of alpha which allows one to
  easily find the solution by the secant method. However, the equation may
  have no solution if an4(alpha=0)>=eps*dl2.
  
  So if adjust_alpha=1 (i.e. I have to find optimal alpha), I proceed as follows:
  
  1.Solve integral equation at alpha=0. If an4(0) >= eps*dl2, no solution at
    alpha > 0, return solution at alpha=0.
  
  2.If an4(0) < eps*dl2, solution exists. Starting from initial alpha,
    multiply alpha by 2, find an4, until an4>=eps*dl2. Set initial points of
    the secant method and use it to find alpha such that fabs(an4)<eps*dl2).

  Note: As alpha is usually very small, solution is searched for for variable x=1/alpha.
*/


// Initialize iteration counters.
  *iter_reg = 0;
  *iter_minim = 0;

  *ierr = 0; // This should be always replaced by value returned from various functions below.
  
  *eta = 0.0; // Important, otherwise if ierr in ptilr1 = 47, eta is unset, and ptizra will fail.

  if( adjust_alpha == 0 ) {

// Test if my gradient methos is correct without regularization.
//    *alpha = 0.0;

// Get the minimum of the functional by conjugate gradient method and compute generalized residual.
//    printf( "ptizr_proj: hx= %le\n", hx );
// Copy initial approximation to z.
    for( i=0; i<n; i++ ) {
      z[i] = z0[i];
    }
    ptilr1( kernel_type, a, u0, v, sumv, hy, m, con, b, icore, l, hx, dl2, 0.0, imax_minim, 
            c, d, z, n, n_con, *alpha, metric, eta, iter_minim, ierr );

    if( *ierr == 200 ) {
      printf( "Eclipse %d: initial approximation outside allowable area. Exiting.\n", kernel_type );
      free_matrix( a, m );
      free_matrix( c, n );
      free_vector( d );
      free_vector( hy );
      free_vector( z0 );
      return;
    }

//    printf( "eta= %e n= %d dl= %f dh= %f hx= %f\n", *eta, n, dl, dh, hx );
    ptizra( *eta, z, n, dl, dh, hx, &an4 ); // Get generalized residual.

    pticr9( *eta, z, n, *alpha, hx, &as2, metric ); // Compute the stabilizing term and add it to eta -> as2

    if( verbose ) {
      printf( "alpha= %17.10le eta= %17.10le an4= %17.10le as2=%17.10e ierr= %d\n", *alpha, *eta, an4, as2, *ierr );
    }
    free_matrix( a, m );
    free_matrix( c, n );
    free_vector( d );
    free_vector( hy );
    free_vector( z0 );
    return;
  }

// Here I come if adjust_alpha=1. Search for alpha by the secant method.

// Solve equation at alpha = 0.0.

// Sometimes the secant method does not work. Depends on the curvature of
// the deviation curve. So below I commented it out and replaced by a simple
// method of division by 2. I.Antokhin, June 2016. Solve at alpha=0 to see
// if an4(0) < eps*dl2 or not.

// Copy initial approximation to z.
  for( i=0; i<n; i++ ) {
    z[i] = z0[i];
  }
  ptilr1( kernel_type, a, u0, v, sumv, hy, m, con, b, icore, l, hx, dl2, 0.0, 
          imax_minim, c, d, z, n, n_con, 0.0, metric, eta, iter_minim, ierr );
  if( *ierr == 200 ) {
//      printf( "Eclipse %d: initial approximation outside allowable area. Exiting.\n", kernel_type );
    free_matrix( a, m );
    free_matrix( c, n );
    free_vector( d );
    free_vector( hy );
    free_vector( z0 );
    return;
  }
//  printf( "eta= %e n= %d dl= %f dh= %f hx= %f\n", *eta, n, dl, dh, hx );
  ptizra( *eta, z, n, dl, dh, hx, &an4 ); // Get generalized residual.

  if( isnan(an4) ) {
    printf("after ptizra an4=NaN, z=\n" );
    for( i=0; i<n; i++ ) {
      printf( "%17.9e\n", z[i] );
    }
    exit(0);
  }

  if( verbose ) {
    printf( "adjust_alpha=1, solved at min alpha=0.0\n" );
    printf( "alpha= 0.0 an4= %17.10le\n",  an4 );
  }

/*
  if( an4 >= eps*dl2 ) { // Even at alpha=0 an4 too large. No solution of the equation fabs(an4)=eps*dl2.
    *ierr = 104;
    *alpha = 0.0;
    free_matrix( a, m );
    free_matrix( c, n );
    free_vector( d );
    free_vector( hy );
    free_vector( z0 );
    return;
  }
*/
// I come here if solution of fabs(an4)<eps*dl2 exists.

  *iter_reg = 0; // Iterations counter
    
// Find alpha such that an4(alpha) >= eps*dl2.
  if( verbose ) {
    printf( "Search for alpha such that an4(alpha)>=eps*dl2.\n" );
  }

  x1 = 0.0;
  f1 = an4;

  do {

    (*iter_reg)++;
// Copy initial approximation to z.
    for( i=0; i<n; i++ ) {
      z[i] = z0[i];
    }
    ptilr1( kernel_type, a, u0, v, sumv, hy, m, con, b, icore, l, hx, dl2, 0.0, 
            imax_minim, c, d, z, n, n_con, *alpha, metric, eta, iter_minim, ierr );

    if( *ierr == 200 ) {
//        printf( "Eclipse %d: initial approximation outside allowable area. Exiting.\n", kernel_type );
      free_matrix( a, m );
      free_matrix( c, n );
      free_vector( d );
      free_vector( hy );
      free_vector( z0 );
      return;
    }

    ptizra( *eta, z, n, dl, dh, hx, &an4 ); // Get generalized residual.

    x2 = *alpha;
    f2 = an4;

    if( verbose ) {
      printf( "alpha= %le an4= %le\n", *alpha, an4 );
    }
/*
    if( verbose ) {
      printf( "ierr= %d alpha= %17.10le an4= %17.10le\nz=\n", *ierr, *alpha, an4 );
      for( i=0; i<n; i++ ) {
        printf( "%15.10e ", z[i] );
      }
      printf( "\n" );
    }
*/

// Just in case I happen to get into the -eps*dl2,+eps*dl2 interval while doing multiplications.
/*
    if( fabs( an4 ) < eps*dl2 ) { // Generalized residual is < eps*dl2 and > -eps*dl2. Finish the job.
      *ierr = 100;
      free_matrix( a, m );
      free_matrix( c, n );
      free_vector( d );
      free_vector( hy );
      return;
    }
*/

    if( *iter_reg > imax_reg ) { // Made imax_reg multiplications by 2.0, residual still negative. Initial regularization parameter too small.
      *ierr = 205;
      free_matrix( a, m );
      free_matrix( c, n );
      free_vector( d );
      free_vector( hy );
      free_vector( z0 );
      return;
    }

    *alpha *= 2.0;

  } while ( an4 < eps*dl2 );

// Search for the optimal alpha.

/*
// Set two initial points of the secant method.
  *alpha /= 2.0;
  f2 = an4;
  x2 = 1.0/(*alpha);
  (*alpha) *= 2.0;
  x1 = 1.0/(*alpha);
*/

// Simple method of division by 2.

  if( verbose ) {
    printf( "Initial points: x1= %le x2= %le f1= %le f2= %le\n", x1, x2, f1, f2 );
  }

/*
// Copy initial approximation to z.
  for( i=0; i<n; i++ ) {
    z[i] = z0[i];
  }
  ptilr1( kernel_type, a, u0, v, sumv, hy, m, con, b, icore, l, hx, dl2, 0.0, 
          imax_minim, c, d, z, n, n_con, *alpha, metric, eta, iter_minim, ierr );

  if( *ierr == 200 ) {
//      printf( "Eclipse %d: initial approximation outside allowable area. Exiting.\n", kernel_type );
    free_matrix( a, m );
    free_matrix( c, n );
    free_vector( d );
    free_vector( hy );
    return;
  }

  ptizra( *eta, z, n, dl, dh, hx, &an4 ); // Get generalized residual.
  if( verbose ) {
    printf( "alpha= %17.10le an4= %17.10le\n", *alpha, an4 );
  }

  f1 = an4;

*/
// Two initial points of the method (x1,f1, x2,f2) set and we know that solution exists. Proceed.

//  *iter_reg = 0; // Iterations counter.

  if( verbose ) {
    printf( "Search alpha by division by 2: eps*dl2= %e\n",  eps*dl2 );
    printf( "Search for solution:\n" );
  }

  while( 1 ) { // Exit is always using the "if" conditions within the loop.

//    printf("eta=%e dl2=%e an4=%e eps*dl2=%e\n ", *eta, dl2, an4, eps*dl2 );

    (*iter_reg)++;

    if( *alpha == 0.0 ) { // alpha became equal to 0.
      *ierr = 103;
      free_matrix( a, m );
      free_matrix( c, n );
      free_vector( d );
      free_vector( hy );
      free_vector( z0 );
      return;
    }


    if( *iter_reg > imax_reg ) { // imax_reg iterations done.
      *ierr = 207;
      free_matrix( a, m );
      free_matrix( c, n );
      free_vector( d );
      free_vector( hy );
      free_vector( z0 );
      return;
    }

/*
// If an4 became negative, go to the modified secant method, see page 103 of the "brown book".
    if( f2 < -eps*dl2 ) {
      break;
    }
*/

/*
// The secant method itself.
    x_new = x1-f1/(f1-f2)*(x1-x2);

    printf( "x1=%e x2=%e f1=%e f2=%e x_new=%e\n", x1, x2, f1, f2, x_new );
    
    x1 = x2;
    f1 = f2;
    x2 = x_new;
    *alpha = 1.0/x_new;
*/

/*
    for( i=0; i<n; i++ ) {
      z[i] = c2;
    }
*/

    x_new = (x1+x2)/2.0; // Middle point of interval.
    *alpha = x_new;

// Copy initial approximation to z.
    for( i=0; i<n; i++ ) {
      z[i] = z0[i];
    }
    ptilr1( kernel_type, a, u0, v, sumv, hy, m, con, b, icore, l, hx, dl2, 0.0, 
            imax_minim, c, d, z, n, n_con, *alpha, metric, eta, iter_minim, ierr );

    if( *ierr == 200 ) {
//      printf( "Eclipse %d: initial approximation outside allowable area. Exiting.\n", kernel_type );
      free_matrix( a, m );
      free_matrix( c, n );
      free_vector( d );
      free_vector( hy );
      free_vector( z0 );
      return;
    }

    ptizra( *eta, z, n, dl, dh, hx, &an4 ); // Get generalized residual.
    if( verbose ) {
      printf( "alpha= %17.10le an4= %17.10le ", *alpha, an4 );
      printf( "x1= %le x2= %le f1= %le f2= %le\n", x1, x2, f1, f2 );
    }

/*
//    printf( "an4= %15.8e eta= %15.8e alpha= %12.5le\n", an4, *eta, *alpha );

    f2 = an4;
//  printf( "alpha= %17.10le an4= %17.10le\n", *alpha, an4 );
//  printf( "x1= %le x2= %le f1= %le f2= %le alpha= %le\n", x1, x2, f1, f2, *alpha );

    if( x2 > 0.999*DBL_MAX ) {
      *alpha = 0.0;
    }
*/

    if( fabs( an4 ) < eps*dl2 ) { // Generalized residual is < eps*dl2 and > -eps*dl2. Finish the job.
      *ierr = 100;
      free_matrix( a, m );
      free_matrix( c, n );
      free_vector( d );
      free_vector( hy );
      free_vector( z0 );
      return;
    }

    if( an4 > 0.0 ) {
      x2 = x_new;
      f2 = an4;
    } else {
      x1 = x_new;
      f1 = an4;
    }


  }  // End of the main loop of the secant method.

/*
// I come here if the residual became negative (<-eps*dl2), perform modified secant method.

  while( 1 ) {

    if( fabs( an4 ) < eps*dl2 ) { // Generalized residual is < eps*dl2 and > -eps*dl2. Finish the job.
      *ierr = 100;
      free_matrix( a, m );
      free_matrix( c, n );
      free_vector( d );
      free_vector( hy );
      return;
    }

    if( *alpha == 0.0 ) { // alpha became equal to 0.
      *ierr = 103;
      free_matrix( a, m );
      free_matrix( c, n );
      free_vector( d );
      free_vector( hy );
      return;
    }

    if( *iter_reg > imax_reg ) { // imax secant iterations done.
      *ierr = 207;
      free_matrix( a, m );
      free_matrix( c, n );
      free_vector( d );
      free_vector( hy );
      return;
    }

// The secant method itself.
    x_new = x1-f1/(f1-f2)*(x1-x2);
    *alpha = 1.0/x_new;

    ptilr1( kernel_type, a, u0, v, sumv, hy, m, con, b, icore, l, hx, dl2, 0.0, 
            imax_minim, c, d, z, n, n_con, *alpha, metric, eta, iter_minim, ierr );

    if( *ierr == 200 ) {
//      printf( "Eclipse %d: initial approximation outside allowable area. Exiting.\n", kernel_type );
      free_matrix( a, m );
      free_matrix( c, n );
      free_vector( d );
      free_vector( hy );
      return;
    }

    ptizra( *eta, z, n, dl, dh, hx, &an4 ); // Get generalized residual.

    printf( "an4= %15.8e eta= %15.8e alpha= %12.5le\n", an4, *eta, *alpha );

    if( f1*an4 ) < 0.0 ) {
      x2 = x_new;
      f2 = an4;
    }
    if( f2*an4 < 0 ) {
      x1 = x_new;
      f1 = an4;
    }
//  printf( "alpha= %17.10le an4= %17.10le\n", *alpha, an4 );
//  printf( "x1= %le x2= %le f1= %le f2= %le alpha= %le\n", x1, x2, f1, f2, *alpha );

   if( x2 > 0.9*DBL_MAX ) *alpha = 0.0;

  }  // End of the main loop of the modified secant method.
*/

}

void ptizra( double an2, double *z, int n, double dl, double dh, double hs, double *an4 ) {

/*
  Compute the generalized residual.

  rho(alpha) = beta(alpha) - (delta+h*||z||)^2-mu

  This is a general formula (page 22 of the "blue" book). In this program,
  it is assumed that mu=0.
  
  This version differs from the original one in that the first operator an2
  = an2*hy is missing as my grid on y (the right-hand side of the Fredholm's
  eq. is uneven. So an2 is computed elsewere fully taking into account the
  steps in y.
  
  Note that I still use an even grid on z (unknown function) so computing
  various integrals over this grid is done just like in the original program
  - i.e., the grid step hx is included in the operator matrix "a".

  Variables:
  an2 - beta in the above formula (the residual ||Ax-u||^2)
  z   - current "solution" (unknown function)
  n   - dimension of z
  dl  - sqrt(del2), for meaning of del2 see above
  dh  - h in the above formula - the error of the operator A
  hs  - the grid step on z s in the Fredholm's eq. Int K(y,s)*z(s)*ds = u(y)
  an4 - the generalized residual rho. Returned value.

  
*/

  int i;
  double s, s1;

  if( dh == 0.0 ) {      // No need to compute the norm.
    (*an4) = an2-dl*dl;
    return;
  }
  
 // Compute the norm of the solution in W21.

  pticr6( z, z, n, &s );
  s *= hs;               // s - square of the solution norm in L2.

  s1 = 0.0;             // Compute the norm of the solution derivative.
  for( i = 1; i<n; i++ ) {
    s1 += (z[i]-z[i-1])*(z[i]-z[i-1]);
  }
  s1 /= hs; // s1 - square of the norm of the solution derivative in L2.

  s = sqrt(s+s1);  // s - norm of the solution in W21.

// Generalized residual.
  (*an4) = an2-(dl+dh*s)*(dl+dh*s);

  return;
}


/*
  Here the ptilr* functions begin which provide the method of the projection
  of conjugate gradients.
  
  The ptilr1 function is different from its original version in that my
  version allows for minimization of the regularization fuanctional,
  containg the regularizating term. Thus, I need to compute the gradients
  and norms accounting for this term.
  
  Also, many of these functions differ from the original ones as I have
  other a-priori constraints and as some details of the equations (like
  their kernels etc. are different for the primary and secondary minima of the
  light curve.
*/

int ptilr0( double **a,  int n ) {

/*
  Inversion of a symmetric positive definite matrix a[n][n] of the order n.
  The resulting matrix is stored in the input matrix a. The matrix must have
  n+1 columns, the last one is used as a working array. It is sufficient to
  provide only the elements of "a" located below its diagonal.

  Parameters:
  
  a[n][n+1] - the matrix;
  n         - its order

*/

double x, y, z;
int i, i1, j, j1, k;

 FILE *alpha_log;

  for( i=0; i<n; i++ ) {
    i1 = i+1;
    for( j=i; j<n; j++ ) {
      j1 = j+1;
      x = a[j][i];
      for( k=i-1; k>=0; --k ) {
        x -= a[k][j1]*a[k][i1];
      }
      if( j == i ) {
        if( x > 0.0 )
          y = 1.0/sqrt(x);
        else {

          alpha_log = fopen("alpha_log", "a");
          fprintf(alpha_log, "Singular projection matrix! Exiting...\n");
          fprintf(alpha_log, "x= %f n= %d i= %d k= %d\n", x, n, i, k);
          fclose(alpha_log);

          return 0;  // Singular projection matrix.
        }
        a[i][i1] = y;
      } else {
        a[i][j1] = x*y;
      }
    }
  }
  for( i=0; i<n; i++ ) {
    for( j=i+1; j<n; j++ ) {
      z = 0.0;
      j1 = j+1;
      for( k=j-1; k>=i; --k ) {
        z -= a[k][j1]*a[i][k+1];
      }
      a[i][j1] = z*a[j][j1];
    }
  }
  for( i=0; i<n; i++ ) {
    for( j=i; j<n; j++ ) {
      z = 0.0;
      for( k=j+1; k<n+1; k++ ) {
        z += a[j][k]*a[i][k];
      }
      a[i][j+1] = z;
    }
  }
  for( i=0; i<n; i++ ) {
    for( j=i; j<n; j++ ) {
      a[j][i] = a[i][j+1];
      a[i][j] = a[i][j+1];
    }
  }

  return 1;
}      

void ptilr1( int kernel_type, double **a, double *u0, double *v, double sumv, double *hy,
             int mn, double **con, double *b, int icore, int l, double hs, double del2,
             double dgr, int icm, double **c, double *d, double *z, int n, int n_con,
             double alpha, char *metric, double *eta, int *ici, int *ierr ) {

/*
  Method of projection of conjugate gradients.

  Note the parameter alpha - this is the regularization parameter, which is absent in
  the original version of the function.

  Variables:

  kernel_type - kernel type, masks depend on it.
  a[mn][n] - matrix of the A operator
  u0[mn] - right-hand side of the Fredholm's eq.
  v[mn]  - weights of the points in u0
  sumv - sum of weights
  hy[mn] - step of the grid on the function in the right-hand side of the equation
  mn - dimension of the right-hand eq. side (denoted "m" in other functions)
  con[NNMAX][NMAX] - matrix of constraints
    b[n_con] - constraints vector
    l - array index of the point where the concaved part of z becomes convex.
   hs - step of the grid on z
 del2 - the limit of minimization by the residual value - when ||az-u||^2 < del2, the function returns
        del2 is 
         d
        int delta(y)^2 dy = sum( delta_i^2* dy_i ), or, if all delta_i=delta, = delta^2*(d-c)
         c
        delta is the error of one data point.
  dgr - the criterion of exit by gradient - the function returns when the
        square of the gradient norm ||grad phi(z)||^2 becomes less than this value. dgr = \int_a^b grad*grad*ds
  icm - max number of big loops
  c[n][n] - matrix of the residual functional
  d[n] - vector of the functional phi(z)=(z,Qz)+(d,z)+e
         c and d are the matrix and vector in the representation of the functional as (c*z,z)+(d,z)+e; computed in ptilra
 z[mn] - initial approximation, after returning from the function, contains the solution
    n - dimension of the unknown vector z
   n_con - number of rows in the matrix of constraints con (the number of constraints)
alpha - regularization parameter
metric - solution metric - L2, W21 or W22
  eta - residual = ||Az-u||^2, returned value
  ici - number of "big loop" iterations done
 ierr - exit code:
   =100 - normal end, exact minimum found
   =102 - by the norm of the gradient
   =200 - Initial approximation is not in the allowable area.
   =202 - could not allocate memory for working arrays.
   =203 - singular matrix aj, inversion impossible.
   =206 - by max number of iterations
   =207 - while computing optimal step, division by zero (zero grad).
Old codes, not used anymore:
   =1 - by the value of residual
   =4 - by the relative change of residual (when deltaX/X < EPS)
*/

#define EPS 1.0e-12

int i, k, iend;//, j;
// double eta_old;

// Variables which used to be formal parameters but now are internal ones.

/*
  Working arrays (I allocate them dynamically):

  mask[n_con] - constraints mask, contains the info on whether restictions are active or not
  aj[n_con][n] - matrix of active constraints
  p[n_con][n] - working array
  pi[n][n+1] - working array, projector
  gr[n] - working array, gradient
  w[max[n_con, mn, n] ] - working array
  p1[n] - working array

*/

int *mask;
double **aj, **p, **pi, *gr, *ww, *u;

// Integers: imask - the number of active constraints; ici - the counter of big loops.
int imask=0;

//int imask_act, mask_act[NTEST], mask_changed;

// int imask1, mask1[150];

// Allocate memory for working arrays.

  if( !( mask = int_vector( n_con ) ) ) {
    *ierr = 202;
    return;
  } else if( !( aj = matrix( n_con, n ) ) ) {
    *ierr = 202;
    free_int_vector( mask );
    return;
  } else if( !( p = matrix( n_con, n ) ) ) {
    *ierr = 202;
    free_int_vector( mask );
    free_matrix( aj, n_con );
    return;
  } else if( !( pi = matrix( n, n+1 ) ) ) {
    *ierr = 202;
    free_int_vector( mask );
    free_matrix( aj, n_con );
    free_matrix( p, n_con );
    return;
  } else if( !( gr = vector( n ) ) ) {
    *ierr = 202;
    free_int_vector( mask );
    free_matrix( aj, n_con );
    free_matrix( p, n_con );
    free_matrix( pi, n );
    return;
  } else if( !( ww = vector( n_con ) ) ) {
    *ierr = 202;
    free_int_vector( mask );
    free_matrix( aj, n_con );
    free_matrix( p, n_con );
    free_matrix( pi, n );
    free_vector( gr );
    return;
  } else if( !( u = vector( mn ) ) ) {
    *ierr = 202;
    free_int_vector( mask );
    free_matrix( aj, n_con );
    free_matrix( p, n_con );
    free_matrix( pi, n );
    free_vector( gr );
    free_vector( ww );
    return;
  }


// Important: INITIAL APPROXIMATION MUST LIE WITHIN OR AT THE BOUNDARY OF
// THE ALLOWABLE AREA!!! Some constraints may be active.


// Get the mask of active constraints for the initial approximation.
  imask = 0;

  pticr3( con, z, ww, n, n_con );
  for( i=0; i<n_con; i++ ) {

/*
    if( ww[i] > b[i] ) { // Initial approximation outside allowable area.
      printf( "i=%d ww=%le b=%le\n", i, ww[i], b[i] );
      for( j=1; j<n-1; j++ ) {
        printf( "%3d   %10f %22.15e %22.15e\n", j, j*hs, z[j], z[j-1]-2*z[j]+z[j+1] );
      }

      *ierr = 200;
      free_int_vector( mask );
      free_matrix( aj, n_con );
      free_matrix( p, n_con );
      free_matrix( pi, n );
      free_vector( gr );
      free_vector( ww );
      free_vector( u );
      return;
    } else if( ww[i] == b[i] ) {
      mask[i] = 1;
      imask++;
    } else {
      mask[i] = 0;
    }
*/

    if( ww[i] >= b[i] ) { // Initial approximation outside allowable area.
      mask[i] = 1;
      imask++;
    } else {
      mask[i] = 0;
    }


  }

//  printf( "ptilr1: imask= %d\n", imask );
//  ptici2( mask, 1, n_con ); // Set mask to zero; no active constraints
//  imask = n_con;

  pticr3( a, z, u, n, mn ); // a*z -> u
  pticr5( u, u0, v, sumv, hy, mn, eta ); // Compute residual --> eta
//  eta_old = *eta+1.0; // Make sure the initial eta_old is larger than eta.

  *ici = 0; // Start iterations of the big loop.
  do {
    (*ici)++;

/*
    printf( "ici= %d z=\n", *ici );
    for( i=0; i<n; i++ ) {
      printf( "%3d %22.15e\n", i, z[i] );
    }
*/

    pticr3( a, z, u, n, mn ); // a*z -> u
/*
    printf( "a[m][n]=\n" );
    for( i=0; i<mn; i++ ) {
      for( j=0; j<n; j++ ) {
        printf( " %le", a[i][j] );
      }
      printf( "\n" );
    }
*/

    pticr5( u, u0, v, sumv, hy, mn, eta ); // Compute residual --> eta

/*
    printf( "sumv= %le n= %d mn= %d\n    u         u0      v\n", sumv, n, mn );
    for( i=0; i<mn; i++ ) {
      printf( "%le %le %le\n", u[i], u0[i], v[i] );
    }
    printf( "eta= %22.15e\n", *eta );
*/
//    eta_old = *eta;

    if( *ici > icm ) { // Return by the max number of iterations.
      *ierr = 206;
      break;
    }  

/*
// Normal end. Return by the value of residual. 

// IMPORTANT!!! Note I have to make at least one iteration. Otherwise, let's
// suppose that the initial approximation of z is such that eta<del2. I want
// to run this fuction at e.g. larger value of alpha, so z would definitely
// change a lot. However, the fuction would return without trying to find a
// new approximation.

// Hmm, ici > 1 seems not enough, the solution just don't change much in one
// iteration. So better comment it out completely. Alternatively, one might
// reset initial approx of z every time before running ptilr1.

// OK, another solution is to set del2=0 in the calling routine. This is how
// it is done in the "Blue book" in ptizr.

    if( *ici > 1 && *eta < del2 ) {
      *ierr = 101;
      break;
    }
*/

//    printf( "ptilr1: imask=%d n=%d n_con=%d\n", imask, n, n_con );
    ptilr3( aj, con, n, n_con, mask ); // Prepare mask constraints.
    if( !ptilr4( aj, p, pi, imask, n, n_con ) ) {  // Prepare projector.
      *ierr = 203;
//      *ierr = 0;
      break;
    }

/*
    printf( "Projector: m=%d n= %d n_con= %d\n", imask, n, n_con );
    for( j=0; j<n; j++ ) {
      printf( "%3d", j );
      for( k=0; k<imask; k++ ) {
        printf( " %22.15e", p[k][j] );
      }
      printf( "\n" );
    }
*/

/*
    printf( "mask=\n" );
    for( i=0; i<n_con; i++ ) {
      printf( "%1d", mask[i] );
    }
    printf( "\n" );
*/
                                  
// Minimization on a facet (in a subspace).
// Note the parameter alpha - this is the regularization parameter absent in the original version.
/*
      printf( "   i     z1=\n" );
      for( i=0; i<mn; i++ ) {
        printf( "%d %le\n", i, z[i] );
      }
*/
    ptilr5( con, b, mask, n, &imask, n_con, pi, c, d, z, alpha, hs, &k, &iend, dgr,
            a, u0, v, sumv, hy, mn, metric );

//    printf( "iend=%d\n", iend );


/*
    printf( "iend=%d mask=\n", iend );
    for( i=0; i<n_con; i++ ) {
      printf( "%1d", mask[i] );
    }
    printf( "\n" );
*/
                                        
    if( iend == 3 ) { // Problems with memory allocation.
      *ierr = 202;
      break;
    }
//    printf( "iend= %d\n", iend );
/*
    printf( "ici= %d\n", *ici );
    for( i=0; i<n; i++ ) {
      printf( "%e %e\n", hs/2+hs*i, z[i] );
    }
*/

    if( iend == 0 ) { // New constraints encountered. Do minimization on the newly defined facet.
      continue;
    } else if( iend == 2 ) {     // Norm of the gradient  < dgr, return.
      *ierr = 102;
      pticr3( a, z, u, n, mn );   // a*z -> u
      pticr5( u, u0, v, sumv, hy, mn, eta ); // Compute residual --> eta
      break;       
    }
/*    
     else if( iend == 3 ) {  // Division by zero would occur in ptilr5 when computing optimal step.
      *ierr = 207;
      free_int_vector( mask );
      free_matrix( aj, n_con );
      free_matrix( p, n_con );
      free_matrix( pi, n );
      free_vector( gr );
      free_vector( ww );
      free_vector( u );
      return;
    }
*/

//  Come here if iend == 1. Check if I can remove some constraints.

/*
    printf( "before ptilr6: imask= %3d mask=\n", imask );
    for( i=0; i<n_con; i++ ) {
      printf( "%1d", mask[i] );
    }
    printf( "\n" );
*/
    ptilr6( kernel_type, icore, con, b, z, p, mask, n, &imask, n_con, c, d, &iend, alpha, hs, metric, l );
//   ptilr6( kernel_type, icore, con, z, p, mask, n, &imask, n_con, c, d, &iend, alpha, hs, metric );
  
/*  
    printf( "after  ptilr6: imask= %3d mask=\n", imask );
    for( i=0; i<n_con; i++ ) {
      printf( "%1d", mask[i] );
    }
    printf( "\n" );
*/

    if( iend == 1 ) {         // Some constraints were removed, continue big loop.
      continue;
    }

/*
// In principle, this is the end - no constraints can be removed, stop
// iterations and exit. In practice, however, my mask of active constraints
// may be not quite accurate due to roundoff errors. Recall that I update
// the mask based on the length of steps along descent directions and do not
// check the actual constraints after each step. So do this now. Compute the
// actual mask and if it is different from current mask, continue
// iterations. If not, stop.

    pticr3( con, z, ww, n, n_con );
    imask_act = 0;
    for( j=0; j<n_con; j++ ) {
      if( ww[j] >= b[j] ) {
        mask_act[j] = 1;
        imask_act++;
      } else {
        mask_act[j] = 0;
      }
    }  
    mask_changed = 0;
    for( j=0; j<n_con; j++ ) {
      if( mask[j] != mask_act[j] ) {
        mask_changed = 1;
        break;
      }
    }

    if( mask_changed ) {

      for( j=0; j<n_con; j++ )
        printf( "%1d", mask[j] );
      printf( "\n" );
      for( j=0; j<n_con; j++ )
        printf( "%1d", mask_act[j] );
      printf( "\n" );
      printf( "mask changed\n" );

      for( j=0; j<n_con; j++ ) {
        mask[j] = mask_act[j];
      }
      imask = imask_act;

      continue;

    } else {
*/
    *ierr = 100;               // Normal end. Exact minimum.

    pticr3( a, z, u, n, mn );  // a*z -> u
    pticr5( u, u0, v, sumv, hy, mn, eta ); // Compute residual --> eta

    if( isnan(*eta) ) {
      printf( "eta is NaN: sumv= %e mn=%d\n", sumv, mn );

      printf( " u      u0     v      hy\n" );
      for( i=0; i<mn; i++ ) {
        printf( "%15.7e %15.7e %15.7e %15.7e\n", u[i], u0[i], v[i], hy[i] );
      }
      printf( "   i     z=\n" );
      for( i=0; i<mn; i++ ) {
        printf( "%d %le\n", i, z[i] );
      }
      free_int_vector( mask );
      free_matrix( aj, n_con );
      free_matrix( p, n_con );
      free_matrix( pi, n );
      free_vector( gr );
      free_vector( ww );
      free_vector( u );
      exit(0);
    }    

    break;       
//    }

  } while( 1 ); // End of big loop. Exit from the loop is always inside of the loop.

//  printf( "ptilr1: imask= %d\n", imask );

/*
  printf( "ptilr1: solution:\n" );
  printf( "ici= %d iend= %d ierr= %d\nmask=", *ici, iend, *ierr );
  for( i=0; i<n_con; i++ ) {
    printf( " %1d", mask[i] );
  }
  printf( "\n" );
*/

/*
  imask1 = 0;
  pticr3( con, z, ww, n, n_con );
  for( i=0; i<n_con; i++ ) {
    if( ww[i] >= b[i] ) {
      mask1[i] = 1;
      imask1++;
    } else {
      mask1[i] = 0;
    }
  }
  printf( "mask вычисленная непосредственно\n" );
  k = 0;
  for( i=0; i<n_con; i++ ) {
    if( (k++)%20 == 0 )
      printf( "\n" );
    printf( "%1d ", mask1[i] );
  }
*/
  
/*
  printf( "z=\n" );
  for( i=0; i<n; i++ ) {
    printf( "%13.6e\n", z[i] );
  }
*/

// Free memory for the working arrays.
  free_int_vector( mask );
  free_matrix( aj, n_con );
  free_matrix( p, n_con );
  free_matrix( pi, n );
  free_vector( gr );
  free_vector( ww );
  free_vector( u );

  return;

}         

void ptilr3( double **aj, double **con, int n, int n_con, int *mask ) {
/*
  Set active constraints by mask (the function with the name ptilr2 is
  absent as in the original fortran code the name was used and an additional
  entry to ptilr1).
  
  Parameters:
  
  aj[n_con][n] - matrix of active constraints
  con[n_con][n] - constraints matrix
  n - size of grid on the unknown fuctions
  n_con - number of rows in the constraints matrix
  mask[n_con] - mask array.   
*/

int i, j, k;

  k = -1;
  for( i=0; i<n_con; i++ ) {
    if( mask[i] != 0 ) {
      k++;
      for( j=0; j<n; j++ ) {
        aj[k][j] = con[i][j];
      }
    }
  }
  return;
}

int ptilr4( double **aj, double **p, double **pi, int m, int n, int n_con ) {
/*
  Compute the projector according to formulae from page 91 of the "Brown book":
  
  PI = E - AJ'(AJ*AJ')^{-1}*AJ
  
  Important: PI is the projector to the space of the dimension n-m, where n
  - the size of the grid on the unknown function, m - number of active
  constraints. The total number of constraints may be larger than n, but
  there may be no more than n active constraints. This is why the dimension
  of PI is nxn+1 (n+1 because I need to invert pi and the function which is
  doing this, ptilr0, uses the last column (n+1) as a working array).
  
  Variables:
  aj[n_con][n] - matrix of active constraints
  p[n_con][n] - working array, -(aj*aj')^{-1}*aj 
  pi[n][n+1] - working array - the projector
  m - number of active constraints. Must not exceed n !!!
  n - size of grid on the unknown function
  n_con - number of rows in the constraints matrix.

  Returns 1 on success, 0 on failure (singular matrix).
*/

int i, j, k;
double r;     

//  if( m > n ) m = n;

//  printf( "ptilr4: m=%d n=%d n_con=%d\n", m, n, n_con );

  if( m == 0 ) {                // No active constraints.
    for( i=0; i<n; i++ ) {      // Set pi=0, p=0
      for( j=0; j<n; j++ ) {
        pi[i][j] = 0.0;
      }
      for( j=0; j<n_con; j++ ) {
        p[j][i] = 0.0;
      }
    }
  } else if( m > n ) { // Critical error, should never happen!!!
    printf( "The number of active constraints m (%d) is larger than n (%d)!!!\nSomething is wrong with your constraints matrix. This is a fatal error. Exiting...\n", m, n );
    exit( 0 );
  } else {
    for( i=0; i<m; i++ ) {      // Compute pi=AJ*AJ', AJ' - conjugate matrix
      for( j=0; j<=i; j++ ) {
        r = 0.0;
        for( k=0; k<n; k++ ) {
          r += aj[i][k]*aj[j][k];
        }
        pi[i][j] = r;
        pi[j][i] = r;
      }
      pi[i][m] = 0.0;          // Just in case, set the working column (the one used when inverting the matrix in ptilr0) to 0.
    }        

/*
    printf( "n= %d m=%d n_con= %d pi=\n", n, m, n_con );
    for( i=0; i<n; i++ ) {
      for( j=0; j<n+1; j++ )
        printf( " %1.0f", pi[i][j] );
      printf( "\n" );
    }
*/
    if( !ptilr0( pi, m ) ) {     // Compute inverse matrix pi=inv(pi)=(AJ*AJ')^-1
//      printf( "!!!!!!!!!!!!!!!!!!!!!\n" );
      return 0;                  // Singular matrix pi.
    }
    for( i=0; i<m; i++ ) {       // Compute p=-pi*AJ
      for( j=0; j<n; j++ ) {
        r = 0.0;
        for( k=0; k<m; k++ ) {
          r += pi[i][k]*aj[k][j];
        }
        p[i][j] = -r;
      }
    }
    for( i=0; i<n; i++ ) {       // Compute pi=AJ'*p
      for( j=0; j<=i; j++ ) {
        r = 0.0;
        for( k=0; k<m; k++ ) {
          r += aj[k][i]*p[k][j];
        }
        pi[i][j] = r;
        pi[j][i] = r;
      }
    }
  }                              // end if m != 0
  for( i=0; i<n; i++ ) {         // Compute pi=E+pi, E - unity matrix
    (pi[i][i]) += 1.0;
  }

  return 1;
}            

#define FTOL 1.0e-10
#define TINY 1.0e-25

void ptilr5( double **con, double *b, int *mask, int n, int *m,
             int n_con, double **pi, double **c, double *d,
             double *z, double alpha, double hs, int *k, int *iend,
             double dgr, double **a, double *u0, double *v, double sumv,
             double *hy, int mn, char *metric ) {
/*
  Minimize the functional in the space defined by the active constraints.
  
  Variables:
  
  con[n_con][n] - constraints matrix ( Constraints look like con*z<=b )
  b[n_con] - constraints vector
  mask[n_con] - mask of active constraints
  n - size of grid on unknown fuction
  m - number of active constraints
  n_con - number of rows in the constraints matrix
  gr[n] - working array
  pi[n][n+1] - working array, projector
  c[n][n] - working, c and d are matrix and vector in the representation of the functional in the form
            (c*z,z)+(d,z)+e 
  d[n] - working       
  z[n] - unknown function
 alpha - regularization parameter
    hs - step of the grid on z
     k - iteration counter in this subspace
  iend - return code:
    =0 - new constraint (that is, new facet)
    =1 - minimum found
    =2 - exit by the norm of the gradient - it is too small
    =3 - could not allocate memory for working arrays.
    =4 - number of active constraints == n
    
   dgr - the upper limit of the gradient norm. If the computed norm is less
         than this value, return. This is input parameter.
a[mn][n] - the equation operator, used for testing te optimal step only
  u0[mn] - "light curve" (right-hand side of the equation), used for testing only
   v[mn] - weights of the "light curve" points, used for testing only
    sumv - sum of weights, used for testing only
      hy - steps on "orbital phases", used for testing only
      mn - number of points in the "light curve"
  metric - string defining the metric used - L2, W21 or W22.
*/

int i, j, np;
double an, an1, an2, alm, alp, al, r, r1, r2, p2, p3;

double *p;  // Working array, new conjugate direction. p[n]
double *p1; // Working array, projection of the anti-gradient, p1[n]
double *gr; // Working array, gradient of the Tikhonov's functional, gr[n]
double *gr0, *p10; // Working arrays, gradient of the residual functional, gr0[n] and its projection.

//double gr0[NTEST], *p10; // Working arrays, gradient of the residual functional, gr0[n] and its projection.

// double fold, fcur; // function values in the previous and current
                      // iterations. Can use them for exit criteria, in practice, though, I do not
                      // use them and just make the number of steps equal to current space dimention, see below.

//double pn; // Norm of the new descend direction.

//int imask_act, mask_act[NTEST], mask_changed;

/*
// These vars are for testing the optimal step below
double uu[NTEST], fff, ttt, ttt0, ttt1, ttt2, ai, z0[NTEST], gr1[NTEST], func1, func2, rr, aall, ah2;
//double ww[NTEST];
double p2tmp1, p2tmp2, p3tmp1, p3tmp2;
//double grad1[NTEST], grad2[NTEST], gr1l, gr2l, ppp[NTEST], step;
// =================================================

// These vars are used to test conjugacy of descent directions.
double grad[NTEST][NTEST], pp[NTEST][NTEST], gg, gp, py, y[NTEST];
double gnorm[NTEST], pnorm[NTEST], ynorm;
int kk, flag;
// ==================================================
*/

//double zz[NTEST];
//int mask_t[150], imask_t;

  if( *m >= n ) {
    printf( "iend= 1 Number of active constraints = n\n" );
    *iend=4;
    return;
  }


  if( !( p = vector( n ) ) ) {
    *iend = 3;
    return;
  } else if( !( p1 = vector( n ) ) ) {
    *iend = 3;
    free_vector( p );
    return;
  } else if( !( gr = vector( n ) ) ) {
    *iend = 3;
    free_vector( p );
    free_vector( p1 );
    return;
  } else if( !( gr0 = vector( n ) ) ) {
    *iend = 3;
    free_vector( p );
    free_vector( p1 );
    free_vector( gr );
    return;
  } else if( !( p10 = vector( n ) ) ) {
    *iend = 3;
    free_vector( p );
    free_vector( p1 );
    free_vector( gr );
    free_vector( gr0 );
    return;
  }

  *k = 0;
  an1 = 1.0;
  pticr2( p, 0.0, n ); // When k=0, the descent direction at iteration "k-1" is 0.
//  pn = 0.0;

// Compute the value of the functional at the initial point. I only need this iif use exit criteria based on function change.
/*
// =====================
  pticr3( a, z, uu, n, mn );  // a*z -> uu
  pticr5( uu, u0, v, sumv, hy, mn, &fff );  // Compute the discrepancy functional fff.
  pticr9( fff, z, n, alpha, hs, &fold, metric ); // Compute the stabilizing term and add it to fff -> fold.

  fcur = fold*10;
// =====================
*/

//  Loop on conjugate gradients, check for the minimum by the number of the steps done.
//  printf( "=======\n" );
  do {

//    printf( "k= %d m= %d n= %d alpha= %22.15e hs=%22.15e\n", *k, *m, n, alpha, hs );

//      printf( "fcur= %22.15e fold= %22.15e left= %22.15e right= %22.15e\n", 
//        fcur, fold, 2.0*fabs(fcur-fold), FTOL*(fabs(fold)+fabs(fcur)+TINY) );

//    if( *k > n-*m && *k % (n-*m) == 0 ) {
/*
    if( *k > 0 && *k % (n-*m) == 0 ) { // Reset direction after n-m iterations.
      an1 = 1.0;
      pticr2( p, 0.0, n ); // When k=0, the descent direction at iteration "k-1" is 0.
    }      
*/
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    if( *k == n-*m ) {  // As the exact minimum is reached in n-m steps, we found it. Exit.
//    if( *k == n-*m || 2.0*fabs(fcur-fold) <= FTOL*(fabs(fold)+fabs(fcur)+TINY) ) {
//    if( *k >= n-*m && 2.0*fabs(fcur-fold) <= FTOL*(fabs(fold)+fabs(fcur)+TINY) ) {
//      printf( "iend=1 k=n-m\n" );

      *iend = 1;
      free_vector( p );
      free_vector( p1 );
      free_vector( gr );
      free_vector( gr0 );
      free_vector( p10 );
      return;
    }

/*
// Print z
    printf( "  i             z\n" );
    for( i=0; i<n; i++ ) {
      printf( "%3d %22.15e\n", i, z[i] );
    }
*/

    ptilr7( z, c, d, gr, n );      // Compute gradient of the discrepancy functional ||Az-u||^2. Step of the grid on z is included in c,d.

/*
// Test of the different method to compute grad
    pticr3( a, z, uu, n, mn );  // a*z -> uu
    pticr4( a, uu, u0, v, hy, sumv, gr, n, mn );
*/

/*
// Print grad
    for( i=0; i<n; i++ ) {
      printf( "%22.15e ", gr[i] );
    }
    printf( "\n" );
*/

    for( i=0; i<n; i++ ) { // Save the gradient of the discrepancy functional to gr0.
      gr0[i] = gr[i];
    }

    if( alpha > 0.0 ) {
      pticr8( z, alpha, hs, gr, n, metric ); // Add the gradient of the stabilizing term.
    }


    pticr6( gr, gr, n, &r );       // Compute the norm of the gradient.

    r *= hs;                       // It is a matter of normalization.
    if(  r < dgr ) {               // The norm (in L2) < dgr - exit.
      printf( "iend=2 norm gradient\n" );
      *iend = 2;
      free_vector( p );
      free_vector( p1 );
      free_vector( gr );
      free_vector( gr0 );
      free_vector( p10 );
      return;
    }
    pticr3(pi, gr, p1, n, n);   // p1 = pi*grad - projection of the gradient.
    for( i=0; i<n; i++ ) {      // p1 = -p1, since I need p(k) = -p1+an2*p(k-1) below.
      p1[i] = -p1[i];
    }
    pticr6( p1, p1, n, &an );   // an - norm squared of the gradient projection.

/*
// This gradient must be perpendicular to p. Test.
    flag = 0;
    pticr6( p1, p, n, &gp );
    pticr6( p, p, n, &pn );
    pn = sqrt( pn );
    ynorm = sqrt( an );
    if( pn != 0.0 && ynorm != 0.0 )
      gp = gp/pn/ynorm; // Cosine of the angle between p1 and p.
    else
      gp = 2.0; // Just some arbitrary value > 1.
    if( abs(gp) <= 1.0 )
      gg = acos(gp)*180/PI;
    else
      gg = 0.0;

    if( fabs(gg-90) > 0.1 && gp <=1.0 ) { // Lost accuracy, further minimization in this subspace meaningless.
//      printf( "Lost accuracy 1: iend=1 angle= %8.4f\n", gg );
      flag = 1;

      *iend = 1;
      free_vector( p );
      free_vector( p1 );
      free_vector( gr );
      free_vector( gr0 );
      free_vector( p10 );
      return;

    }

    if( gp > 1.0 )
      flag = 2;
//    printf( "grnorm=%22.15e p_norm = %22.15e cos_gp = %22.15e angle= %7.3f flag= %1d\n", ynorm, pn, gp, gg, flag );
*/
/*
    printf( "    i         z                    p               gr_proj               p*gr_proj\n" );
    for( i=0; i<n; i++ ) {
      printf( "%3d %22.15e %22.15e %22.15e %22.15e\n", i, z[i], p[i], p1[i], p[i]*p1[i] );
    }
    gg = 0.0;
    for( i=0; i<n; i++ ) {
      gg += p[i]*p1[i];
      printf( "%3d %22.15e\n", i, gg );
    }
*/
// End test. 

// Note that in principle I have to multiply an by hs, but hs will cancel
// due to the division in the next line, so I don't multiply.

    an2 = an/an1;               // an2 - coefficient for the next conjugate direction.
    pticr1( p1, p, an2, p, n ); // p - new conjugate direction (of descent). p^(k) = p1+an2*p^(k-1)

/*
// Print z
    printf( "grnorm=%e grprojnorm= %e an2= %e\n", r, an, an2 );
    pticr1( z, p, al, zz, n );   // Do step zz=z+al*p    
    printf( "    i         z                    p               gr_proj                gr\n" );
    for( i=0; i<n; i++ ) {
      printf( "%3d %22.15e %22.15e %22.15e %22.15e\n", i, z[i], p[i], p1[i], gr[i] );
    }
*/

/*
// Test conjugacy of descent directions. If they are, the following conditions must be fulfilled:
// 1.(grad_j,grad_i)=0, i!=j From Num. rec. c book, p.422 ??? Strange, grads are not necessarily perpendicular ???
// 2.(grad_j,p_i)=0, i!=j    From Num. rec. - same
// 3.(y_i,p_j)=0, i!=j, where y_i=grad_i-grad_(i-1) - Main condition, from Gill et al., "Practical Optimization", p.202

// Save current grad and direction.
    kk = *k;
    gnorm[kk] = pnorm[kk] = 0.0;
    for( i=0; i<n; i++ ) {
      grad[kk][i] = p1[i];
      pp[kk][i] = p[i];
      gnorm[kk] += p1[i]* p1[i];
      pnorm[kk] += p[i]*p[i];
    }
    gnorm[kk] = sqrt( gnorm[kk] );
    pnorm[kk] = sqrt( pnorm[kk] );
    
    if( kk > 0 ) {
//      printf( "  j           gg                 gp                    py\n" );
//      printf( "  j      py\n" );
      ynorm = 0.0;
      for( i=0; i<n; i++ ) {
        y[i] = grad[kk][i]-grad[kk-1][i];
        ynorm += y[i]*y[i];
      }
      ynorm = sqrt( ynorm );
      
      for( j=0; j<kk; j++ ) {
        gg = gp = py = 0.0;
        for( i=0; i<n; i++ ) {
          gg += grad[j][i]*grad[kk][i];
          gp += grad[j][i]*pp[kk][i];
          py += y[i]*pp[kk][i];
        }
        gg = gg/gnorm[j]/gnorm[kk];
        gp = gp/gnorm[j]/pnorm[kk];
        py = py/ynorm/pnorm[kk];
//        printf( "%3d %22.15e %22.15e %22.15e   %7.3f  %7.3f  %7.3f\n", j, gg, gp, py, acos(gg)*180/PI, acos(gp)*180/PI, acos(py)*180/PI );
//        printf( "%3d %7.3f  py\n", j, acos(py)*180/PI );
        if( fabs(acos(py)*180/PI-90.0 ) > 0.1 ) {
//           printf( "Lost accuracy 2: con_angle= %8.4f\n", acos(py)*180/PI );
          *iend = 1;
          free_vector( p );
          free_vector( p1 );
          free_vector( gr );
          free_vector( gr0 );
          free_vector( p10 );
          return;
        }

      }
    }
// End test of directions conjugancy.
*/


// Compute various quantities needed for calculating the optimal step along
// the descent direction. See my notes.

    pticr6( p, gr0, n, &r );     // r=(p,gr0)

    r1 = 0.0;                   // r1=(c*p,p)
    p2 = 0.0;                   // Stabilizing term for denominator
    p3 = 0.0;                   // Stabilizing term for nominator.
    for( i=0; i<n; i++ ) {
      r2 = 0.0;
      for( j=0; j<n; j++ ) {
        r2 += c[i][j]*p[j];
      }
      r1 += p[i]*r2;
// Compute the parts related to the stabilizing term, for various metrics.
      if( alpha > 0.0 ) {

        if( strcmp( metric, "L2" ) == 0 ) {
          p2 += p[i]*p[i];
          p3 += z[i]*p[i];
        } // End L2 metric
        
        if( strcmp( metric, "W21" ) == 0 ) {
          p2 += p[i]*p[i];
          p3 += z[i]*p[i];
          if( i > 0 ) {
            p2 += (p[i]-p[i-1])*(p[i]-p[i-1])/hs/hs;
            p3 += (z[i]-z[i-1])*(p[i]-p[i-1])/hs/hs;
          }
        }  // End W21 metric

        if( strcmp( metric, "W22" ) == 0 ) {

          p2 += p[i]*p[i];  // ||z||^2
          p3 += z[i]*p[i];

          if( i > 0 ) { // ||z'||^2
            p2 += (p[i]-p[i-1])*(p[i]-p[i-1])/hs/hs;
            p3 += (z[i]-z[i-1])*(p[i]-p[i-1])/hs/hs;
          }
          
          if( i>=1 && i<=n-2 ) { // ||z"||^2
            p2 += (p[i+1]-2*p[i]+p[i-1])*(p[i+1]-2*p[i]+p[i-1])/hs/hs/hs/hs;
            p3 += (z[i+1]-2*z[i]+z[i-1])*(p[i+1]-2*p[i]+p[i-1])/hs/hs/hs/hs;
          }
          if( i==0 ) { // This term is needed if z is constant beyound the limits of the interval [a,b].
            p2 += (p[1]-p[0])*(p[1]-p[0])/hs/hs/hs/hs; // For free ends, comment these lines.
            p3 += (z[1]-z[0])*(p[1]-p[0])/hs/hs/hs/hs;
          }

        }  // End W22 metric

      }
    }

//    printf( "1: p2=%15.10e p3=%15.10e hs=%15.10e\n", p2*hs, p3*hs, hs );

    p2 *= hs;
    p3 *= hs;

//    printf( "k=%d r=%15.10e r1=%15.10e p2=%15.10e p3=%15.10e alpha=%15.10e\n", *k, r, r1, p2, p3, alpha );

    if( alpha > 0.0 && r1+alpha*p2 == 0.0 ) { // Divizion by zero will occur.
      printf( "iend=1 Division by zero\n" );
      *iend = 1; // 3?
      free_vector( p );
      free_vector( p1 );
      free_vector( gr );
      free_vector( gr0 );
      free_vector( p10 );
      return;
    }

    if( alpha <= 0.0 && r1 == 0.0 ) {  // Divizion by zero will occur.
      printf( "iend=1 Division by zero\n" );
      *iend = 1; // 3?
      free_vector( p );
      free_vector( p1 );
      free_vector( gr );
      free_vector( gr0 );
      free_vector( p10 );
      return;
    }
    
    if( alpha > 0.0 ) {
      al = -(r+2.0*alpha*p3)/(r1+alpha*p2)/2.0;  // Optimal step - see my formula in the notes. (1)
    } else {
      al = -r/r1/2.0;
    }

//    printf( "a_opt= %22.15e\n", al );

/*
// Check optimal step.
// ================================
    for( i=0; i<n; i++ ) {
      z0[i] = z[i];
    }
    ai = al-0.001*fabs(al);
    pticr1( z0, p, ai, zz, n );   // Do step z=z0+ai*p    
    pticr3( a, zz, uu, n, mn );  // a*z -> uu
    pticr5( uu, u0, v, sumv, hy, mn, &fff );  // Compute the discrepancy functional fff.
    pticr9( fff, zz, n, alpha, hs, &ttt1, metric ); // Compute the stabilizing term and add it to fff -> ttt.

    ai = al+0.001*fabs(al);
    pticr1( z0, p, ai, zz, n );   // Do step z=z0+ai*p    
    pticr3( a, zz, uu, n, mn );  // a*z -> uu
    pticr5( uu, u0, v, sumv, hy, mn, &fff );  // Compute the discrepancy functional fff.
    pticr9( fff, zz, n, alpha, hs, &ttt2, metric ); // Compute the stabilizing term and add it to fff -> ttt.

    ai = al;
    pticr1( z0, p, ai, zz, n );   // Do step z=z0+ai*p    
    pticr3( a, zz, uu, n, mn );  // a*z -> uu
    pticr5( uu, u0, v, sumv, hy, mn, &fff );  // Compute the discrepancy functional fff.
    pticr9( fff, zz, n, alpha, hs, &ttt0, metric ); // Compute the stabilizing term and add it to fff -> ttt.

    if( ttt0 > ttt1 || ttt0 > ttt2 ) {
//      printf( "Wrong step %22.15e | %22.15e  %22.15e %22.15e\n", al, ttt1, ttt0, ttt2 );
    }


// Test of the optimal step
// ==================================================
      printf( "a_opt= %15.10e\n", al );
      printf( "TEST OPT_STEP\n" );
      for( i=0; i<n; i++ ) {
        z0[i] = z[i];
      }
      ai = al-0.01*fabs(al);
      j = 0;
      do {
        pticr1( z0, p, ai, zz, n );   // Do step z=z0+ai*p    
        pticr3( a, zz, uu, n, mn );  // a*z -> uu
        pticr5( uu, u0, v, sumv, hy, mn, &fff );  // Compute the discrepancy functional fff.
        i++;
        if( i == 4 ) {
          *iend = 203;
          free_vector( p );
          free_vector( p1 );
          free_vector( gr );
          free_vector( gr0 );
          free_vector( p10 );
          return;
        }

        pticr9( fff, zz, n, alpha, hs, &ttt, metric ); // Compute the stabilizing term and add it to fff -> ttt.
        printf( "a= %22.15e Fcore= %22.15e Ftotal= %22.15e\n", ai, fff, ttt );
        ai += 0.0001*fabs(al);
      } while( ai < al+0.01*fabs(al) );
// End Test optimal step.

      *iend = 1; // 3?
      free_vector( p );
      free_vector( p1 );
      free_vector( gr );
      free_vector( gr0 );
      free_vector( p10 );
      return;
//    }

    if( flag == 1 ) {
      printf( "flag=1 %22.15e  %22.15e %22.15e\n", ttt1, ttt0, ttt2 );
    }
// ====================================
*/

    if( al <= 0.0 ) {           // If al<=0 - Rounding errors - al should always
                                // be positive if the direction of decsent if calculated
                                // correctly. Return with the current z in the hope
                                // that next iterations will improve the situation.
//      printf( "al: <0 al= %e r= %e p3= %e r1= %e p2= %e alpha= %e\n", al, r, p3, r1, p2, alpha );

/*
// Theoretically, after this step we should still stay within the current
// subspace. Due to round off errors, however, this may be not true. So
// compute the actual mask and if it is different from "mask", get out of
// this function and start over with the new projection.

      pticr3( con, z, ww, n, n_con );
      imask_act = 0;
      mask_changed = 0;
      for( j=0; j<n_con; j++ ) {
        if( ww[j] >= b[j] ) {
          mask_act[j] = 1;
          imask_act++;
        } else {
          mask_act[j] = 0;
        }
      }  
      for( j=0; j<n_con; j++ ) {
        if( mask[j] != mask_act[j] ) {
          mask_changed = 1;
          break;
        }
      }

      if( mask_changed ) {
        for( j=0; j<n_con; j++ )
          printf( "%1d", mask[j] );
        printf( "\n" );
        for( j=0; j<n_con; j++ )
          printf( "%1d", mask_act[j] );
        printf( "\n" );
        printf( "mask changed\n" );

        for( j=0; j<n_con; j++ ) {
          mask[j] = mask_act[j];
        }
        (*m) = imask_act;

        *iend = 0;
        free_vector( p );
        free_vector( p1 );
        free_vector( gr );
        free_vector( gr0 );
        free_vector( p10 );
        return;
      }
*/
//      printf( "iend=1 Step < 0\n" );

//      mask[i] = 1;                 // Include new resctriction to mask.
//      (*m)++;
//      *iend = 0;

      *iend = 1;
      free_vector( p );
      free_vector( p1 );
      free_vector( gr );
      free_vector( gr0 );
      free_vector( p10 );
      return;
    }

    alm = al+1.0;               // Search for the nearest plane.
    np = 0;

/*
// Actual Mask
    pticr3( con, z, ww, n, n_con );
    for( j=0; j<n_con; j++ ) {
      if( ww[j] >= b[j] ) {
        mask_act[j] = 1;
      } else {
        mask_act[j] = 0;
      }
    }  
// End
*/

    for( i=0; i<n_con; i++ ) {
      if( mask[i] == 1 ) {      // No need to check active constraints.
//      if( mask_act[i] == 1 ) {      // No need to check active constraints.
        continue;
      }
      r = 0.0;
      r1 = 0.0;
      for( j=0; j<n; j++ ) {
        r += con[i][j]*z[j];
        r1 += con[i][j]*p[j];
      }
      if( r1 <= 0.0 ) {         // r1~cos(theta) - the angle between vectors p and normale to the plane. If cos<0= we look away from the plane.
        continue;               // Look towards the plane when angle<90deg, i.e. cos>0.
      }
      alp = (b[i]-r)/r1;        // alp - distance to the plane.

//      printf( "alp: i=%d b=%e r=%e r1=%e alp=%e\n", i, b[i], r, r1, alp );

      if( alp <= 0.0 ) { // This happens if the point is at a border, but mask=0 due to rounding errors.

/*
        printf( "alp: i=%d b=%e r=%e r1=%e alp=%e\n", i, b[i], r, r1, alp );
        if( i>0 && i<n-1 ) {
          printf( "z= %22.15e %22.15e %22.15e\n", z[i-1], z[i], z[i+1] );
        }
*/

/*
        printf( "iend=0 alp= %22.15e i= %3d\n", alp, i );
        mask[i] = 1;                 // Include new resctriction to mask.
        (*m)++;
        *iend = 0;
*/
//        printf( "alp<0: %22.15e\n", alp );
        *iend = 1;
        free_vector( p );
        free_vector( p1 );
        free_vector( gr );
        free_vector( gr0 );
        free_vector( p10 );
        return;

      }

      if( alm <= alp ) {        // Search for the minimal distance.
        continue;
      }
      alm = alp;
      np = i;                   // np = number of constraint.
    }

//    printf( "almin= %22.15e\n", alm );

    if( al >= alm ) {           // If al >= alm - have new constraint.
      break;
    }

/*
// Test mask.
    pticr3( con, z, ww, n, n_con );
    for( j=0; j<n_con; j++ ) {
      if( ww[j] >= b[j] ) {
        mask_act[j] = 1;
      } else {
        mask_act[j] = 0;
      }
    }  
    printf( "m = ");
    for( j=0; j<n_con; j++ )
      printf( "%1d", mask[j] );
    printf( "\n" );
    printf( "m1= ");
    for( j=0; j<n_con; j++ )
      printf( "%1d", mask_act[j] );
    printf( "\n" );
// End test mask
*/
    pticr1( z, p, al, z, n );   // Do step z=z+al*p    

/*
// Test mask.
    pticr3( con, z, ww, n, n_con );
    for( j=0; j<n_con; j++ ) {
      if( ww[j] >= b[j] ) {
        mask_act[j] = 1;
      } else {
        mask_act[j] = 0;
      }
    }  
    printf( "m2= ");
    for( j=0; j<n_con; j++ )
      printf( "%1d", mask_act[j] );
    printf( "\n" );
// End test mask
*/

// !!!!!!!!! This is a hack - if some components of z are negative, set all subsequent components to zero.
/*
   for( i=0; i<n; i++ ) {
     if( z[i] < 0.0 ) {
       for( j=i; j<n; j++ ) {
         z[j] = 0.0;
       }
       break;
     }
   }
*/

// I need these lines if I use fcur and fold for exit criteria based on function change. Which I don't.
/*
// ========================================
    fold = fcur;
    pticr3( a, z, uu, n, mn );  // a*z -> uu
    pticr5( uu, u0, v, sumv, hy, mn, &fff );  // Compute the discrepancy functional fff.
    pticr9( fff, z, n, alpha, hs, &fcur, metric ); // Compute the stabilizing term and add it to fff -> fcur.
//    printf( "k= %d m= %d Fcore= %22.15e Ftotal= %22.15e\n", *k, *m, fff, fcur );
// ========================================
*/

    (*k)++;
    an1 = an;

  } while( 1 );                 // Exit from the loop is always inside.

//  printf( "al>alm: al=%20e alm=%20e\n", al, alm );

  al = alm;                     // Reached a new point on the border.


//  printf( "Actual restricted step al= %e\n", al );
  pticr1( z, p, al, z, n );     // Do step z=z+al*p

/*
  for( i=0; i<n; i++ ) {
    printf( "%3d %22.15e %22.15e\n", i, z[i], p[i] );
    z[i] = zz[i];   // !!!!!!!!!!!!!!!!!!!!!!!!!!!!
  }
*/

/*
  pticr3( a, z, uu, n, mn );  // a*z -> uu
  pticr5( uu, u0, v, sumv, hy, mn, &fff );  // Compute the discrepancy functional fff.
  pticr9( fff, z, n, alpha, hs, &ttt, metric ); // Compute the stabilizing term and add it to fff -> ttt.
  printf( "k= %d m= %d Fcore= %22.15e Ftotal= %22.15e max\n", *k, *m, fff, ttt );
*/

  if( !mask[np] ) {
    mask[np] = 1;                 // Include new resctriction to mask.
    (*m)++;
  }

//  printf( "iend=0 New constraint\n" );
  *iend = 0;
  free_vector( p );
  free_vector( p1 );
  free_vector( gr );
  free_vector( gr0 );
  free_vector( p10 );
  return;
}

void ptilr6( int kernel_type, int icore, double **con, double *b, double *z, double **p, int *mask, 
             int n, int *m, int n_con, double **c, double *d, int *iend, double alpha, double hs, char *metric, int l ) {
/*
  A function which removes constraints. Differs from the original one in
  that I never remove the last and the first constraints for the 1st
  minimum.

  Parameters:
  kernel_type - kernel type (1, 2 for Fredholm, 3 for Abel)
  icore - index of the last constraint of the equality type
  con[n_con][n] - constraints matrix
  b[n_con] - right-hand side of the constraints inequalities
  z[n] - unknown function
  p[n_con][n] - working array,  -(aj*aj')^{-1}*aj, where aj - matrix of active constraints, see ptilr4.
  mask[n_con] - mask of active constraints
  n - size of grid on unknown function
  m - number of active constraints
  n_con - number of rows in the constraints matrix
  c[n][n] - working array - Matrix in the representation of ||Az-u||^2=(Cz,z)+(D,z)+e
  d[n] - working array, see previous line
  iend - return code
  =0 - no constraints can be removed
  =1 - some constraints were removed
  =3 - could not allocate memory for gr
  alpha - regularization parameter
  hs - step of the grid on s
  metric - metric used (L2, W21, W22)
*/

int i, j, k, nnn;
double al, r;
double *gr; // Working array, gradient of the discrepancy functional, gr[n]
//double ww[NTEST];

//double gr0[NTEST];

//double al1, al2;
//int nnn1, nnn2;

//  printf( "ptilr6 entered\n" );
  if( !( gr = vector( n ) ) ) {
    *iend = 3;
    return;
  }

  ptilr7( z, c, d, gr, n ); // Compute the gradient of the discrepancy functional
//  for( i=0; i<n; i++ ) {
//    gr0[i] = gr[i];
//  }
  if( alpha > 0.0 ) {
    pticr8( z, alpha, hs, gr, n, metric ); // Add the gradient of the stabilizing term.
  }

/*
  printf( "ptilr6:\n     gr0                gr\n" );
  for( i=0; i<n; i++ ) {
    printf( "%3d %22.15e %22.15e\n", i, gr0[i], gr[i] );
  }
*/

/*
  printf( "Projector: m=%d n= %d n_con= %d\n", *m, n, n_con );
  for( j=0; j<n; j++ ) {
    printf( "%3d", j );
    for( k=0; k<*m; k++ ) {
      printf( " %22.15e", p[k][j] );
    }
    printf( "\n" );
  }
*/

//  printf( "Shadow parameters:\n" );
  *iend = 0;
  al = 1.0;

//  al1 = al2 = 1.0;

  k = -1;
  for( i=0; i<n_con; i++ ) {
    if( mask[i] == 0 ) {
      continue;
    }
    k++;
    r = 0.0;  // For active constraints, compute the shadow parameter.
    for( j=0; j<n; j++ ) {
      r += p[k][j]*gr[j]; // -(aj*aj')^{-1}*aj, minus sign is correct, see my latex notes.
    }
//    printf( " %3d %22.15e %d\n", i, r, k );


    if( r >= al || i <= icore ) {  // Search for the minimal shadow parameter, skip those wich correspond to i<=icore.
      continue;
    }
    al = r;
    nnn = i;

/*
    if( i < l && r < al1 && i >= icore ) {
      al1 = r;
      nnn1 = i;
    }
    if( i >= l && r < al2 ) {
      al2 = r;
      nnn2 = i;
    }
*/
  }
/*
  if( al1 >= 0.0 && al2 >= 0.0 )
    al = 1.0;
  else if( al2 < 0.0 ) {
    al = al2;
    nnn = nnn2;
  } else {
    al = al1;
    nnn = nnn1;
  }
*/
  if( al >= 0.0 ) {  // if al >= 0, return (iend=0), otherwise, remove constraint number nnn.
//    printf( "ptilr6: cannot remove mask\n" );
    free_vector( gr );
    return;
  }
  mask[nnn] = 0;
  *iend = 1;
  (*m)--;

//  printf( "ptilr6: removed mask at np=%3d al11= %e al2= %e nnn1= %d nnn2= %d\n", nnn, al1, al2, nnn1, nnn2 );
//  printf( "ptilr6: removed mask at np=%3d\n", nnn );
  
  free_vector( gr );
  return;
}


void ptilr7( double *z, double **c, double *d, double *gr, int n ) {
/*
  Compute the gradient of a quadratic functional which has the form
  (c*z,z)+(d,z)+e. (c,d,e, are computed in ptilra). Note that this function
  does NOT compute the gradient of the regularization term. For that, see
  pticr8.

  For mathematical details on how I get the formulae for this gradient, see
  my notes on the algorithm in the "latex" directory.
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
  Parameters:
  z[n] - unknown function
  c[n][n], d[n] - coefficients in the representation of the functional (c*z,z)+(d,z)+e
  gr[n] - gradient
  n - size of grid on z

*/

int i, j;
double r;

// Gradient = 2cz+d
  for( i=0; i<n; i++ ) {
    r = 0.0;
    for( j=0; j<n; j++ ) {
      r += (c[i][j]+c[j][i])*z[j];
    }
    gr[i] = r+d[i];
  }
  return;
}

//void ptilra( int kernel_type, double **a, double *u, double *v, double sumv, double *hy, double **c, double *d, int n, int m ) {
void ptilra( double **a, double *u, double *v, double sumv, double *hy, double **c, double *d, int n, int m ) {
/*

  Reduce the residual functional to the form (c*z,z)+(d,z)+e. In my case I
  do not compute e=sum_j u[j]^2*v[j]*hy[j]/sumv as this is a constant term
  which has no effect on the gradient. I need this form to compute the
  gradient of the functional.

  In tex: $C=A^*A$, $D=-2A^*\vec{u_\delta}$, $e=||\vec{u_\delta}||^2_{L_2}$
  This is however, not exact expressions as they do not account for the
  facts that (1)the grid on phases in uneven; (2)A includes step on s.

  Note that this function assumes that the grid step on the integration
  variable (x) is included in the a matrix so I do not have to multiply the
  sums below by the value of this step.

  This version is different from the original one in that I take weights of
  points in the right-hand side "u" into account.

  For mathematical details of how I compute "c" and "d" see my notes on the
  algorithm in the "latex" directory.

  Parameters:
  kernel_type - kernel type (1, 2 for Fredholm, 3 for Abel, 4 for test model). For 1,2,4 hy is included. For 3, it is skipped as it is included in matrix "a".
  a[m][n] - the matrix of the operator a in a*z=u. "a" includes step on z hx (hx/2 for 1st and last points)!
  u[m] - the right-hand side
  v[m] - weights of u
  sumv - integral of v divided by the integration interval. This is normalization coefficient
  hy[m] - steps of the grid on y.
  c[n][n] - the matrix used in the new form of the functional (output)
  d[n] - the vector in the new form (output)
  n - size of grid on z
  m - size of grid in u.
  
*/

int i, j, k;
double r;

  for( i=0; i<n; i++ ) {          // d=-2*a'*u            
    r = 0.0;
    for( j=0; j<m; j++ ) {
      r += a[j][i]*u[j]*hy[j]*v[j]; // First index in a - on the right-hand side of the equation. 2nd - grid on unknown function.
    }
    d[i] = -2.0*r/sumv;
  }
  for( i=0; i<n; i++ ) {          // l=a'*a  
    for( j=0; j<=i; j++ ) {
      r = 0.0;
      for( k=0; k<m; k++ ) {
        r += a[k][i]*a[k][j]*hy[k]*v[k];
      }
      c[i][j] = r/sumv;
      c[j][i] = c[i][j];
    }
  }

  return;
}

#define BTOL 1.0e-10

void ptilrb( int kernel_type, int switch_contype, int *n_con, double ***con,
             double **b, int n, double c2, int icore, int l,
             int *ierr ) {

/*
  Allocate memory and compute the constraints matrix. This function is VERY
  different from the original one as I use different a-priori constraints.

  Since I call this function many times and every time I have to allocate
  memory for con, the first thing to do is to check if con is already
  allocated and if yes, free the memory and allocate it again with required
  size. IMPORTANT!!! Before first call to ptilrb, n_con MUST be set to 0!!!
  Its value is used to determine if the memory for con and b was previously
  allocated or not. If n_con == 0, memory was not allocated. If n_con != 0,
  memory was allocated and I have to free it first and then allocated again
  with necessary size.

  Another important point. I create con and b dynamically within this
  function. To return these arrays to the calling function, I have to pass
  to ptilrb not the pointers to the matrix and vector but pointers to these
  pointers. This is why ***con and **b. For details, see
  test_dynamic_arrays.c For this reason, note *con and * b in functions
  free_vector and free_matrix.

  In MY original version of the function I had only constraints for the
  Fredholm equation, for the 1st and 2nd minima of the light curve. And, I
  had only one set of constrains in each case. So if I wanted to use a
  different set of constraints, I had to use another version of the
  function. This is why I had many ptilrb functions for different types of
  constraints.
  
  When solving Abel equation, I have to use three types of constraints in
  the same code sequentially, to choose the best one (see my paper notes).
  So I have to implement all these constraints in one function. This is what
  I do in this version of the function. This also simplifies the use of
  ptizr_proj as I do not have to edit prgrad.c every time I want to use a
  different set of constraints.

  This is achived through the swith "switch_contype" passed as a parameter
  to ptilrb. To be able to use it in e.g. wr_prgrad.c or abel_prgrad.c, I
  have to also pass the switch to ptizr_proj. Thus, the formal parameters of
  these function are changed compared to my original versions of these
  functions.


  Parameters:
  kernel_type - number of minimum: 1 or 2 in Fredholm eq. 3 for abel equation, 4 for test model
  switch_contype - switch defining the type of constraints
  con[n_con][n]  - constraints matrix
  b[n_con]       - right-hand side of constraints
  n             - dimension of z
  n_con          - number of constraints
  c2          - value of z[0] for the 1st mim

  icore         - The index of the last equality constraint. The equality
                constraint must be first in the matrix of constraints. icore must be >=0 and <= l.
                In my case icore = icore+1 as I require z[n-1]=0 in addition to z[i]=c2, i=1,...,icore
  l           - array index of the inflation point
*/

int i, j, k;

//double tau0, tau, tau1p, tau2p, xi, rabs; // For testing non-zero values of b in the 1st min.
double **tmp_con, *tmp_b;

  *ierr = 0;
//  printf( "ptilrb: kernel_type= %d l= %d\n", kernel_type, l );

  if( kernel_type < 1 && kernel_type > 4 ) {
//    printf( "Wrong kernel_type in ptilrb!!! Exiting.\n" );
    *ierr = 208;
    return;
  }

  if( kernel_type == 1 ) {  // Fredholm equation, 1st eclipse. All constraint types below include
                            // non-transparent core: at i<=icore z[i]<=c2. The initial approximation
                            // MUST set z[i]=c2 at i<=icore, all these constraints are always
                            // set to be active in the calling function.

    switch( switch_contype ) {

      case 1: // Monotonically non-increasing function with non-transparent core.

// The number of constraints n_con = (icore+1)+(n-2-icore+1)+1 = n+1
// In the non-transparent core z[i] = c2. Note that in ptilr6 I never remove these constraints!!!

        if( *n_con != 0 ) { // Memory was allocated, free it.
          free_matrix( *con, *n_con );
          free_vector( *b );
        }

// Allocate memory.
        *n_con = n+1; // See comments below.
        if( !( tmp_con = matrix( *n_con, n ) ) ) { // Constraints matrix.
          *ierr = 202;
          return;
        }
        if( !( tmp_b = vector( *n_con ) ) ) { // Constraints vector (right-hand side of constraints).
          *ierr = 202;
          free_matrix( tmp_con, *n_con );
          return;
        }

// Just in case - reset the constraints matrix and the right-hand side vector.
        for( i=0; i<*n_con; i++ ) {
          for( j=0; j<n; j++ ) {
            tmp_con[i][j] = 0.0;
          }
          tmp_b[i] = 0.0;
        }

        for(i=0; i<=icore; i++ ) {
          tmp_con[i][i] = 1.0;           // z[i]<=c2
          tmp_b[i] = c2;
        }

        k = icore+1;
        for( i=icore; i<n-1; i++ ) { // z[i]>=z[i+1] i.e. -z[i]+z[i+1] <=0
          tmp_con[k][i] = -1.0;
          tmp_con[k][i+1] = 1.0;
          tmp_b[k] = 0.0;
          k++;
        }

        tmp_con[k][n-1] = -1.0;          // -z[n-1] <= 0, i.e. z[n-1] >= 0
        tmp_b[k] = 0.0;

        break;
      
      case 2: // Concave function with non-transparent core.

// The number of constraints n_con = (icore+1)+1+(n-2-icore)+1+1 = n+2
// In the non-transparent core z[i] = c2. Note that in ptilr6 I never remove these constraints!!!

        if( *n_con != 0 ) { // Memory was allocated, free it.
          free_matrix( *con, *n_con );
          free_vector( *b );
        }

// Allocate memory.
        *n_con = n+2; // See comments below.
        if( !( tmp_con = matrix( *n_con, n ) ) ) { // Constraints matrix.
          *ierr = 202;
          return;
        }
        if( !( tmp_b = vector( *n_con ) ) ) { // Constraints vector (right-hand side of constraints).
          *ierr = 202;
          free_matrix( tmp_con, *n_con );
          return;
        }

// Just in case - reset the constraints matrix and the right-hand side vector.
        for( i=0; i<*n_con; i++ ) {
          for( j=0; j<n; j++ ) {
            tmp_con[i][j] = 0.0;
          }
          tmp_b[i] = 0.0;
        }

        for(i=0; i<=icore; i++ ) {
          tmp_con[i][i] = 1.0;              // z[i]<=c2
          tmp_b[i] = c2;
        }

        k = icore+1;                    // z[icore]>=z[icore+1] i.e. -z[icore]+z[icore+1] <=0
        tmp_con[k][icore] = -1.0;
        tmp_con[k][k] = 1.0;
        tmp_b[k] = 0.0;
        k++;

        for( i=icore+1; i<n-1; i++ ) {  // Concave - second derivs are negative.
          tmp_con[k][i] = -2.0;
          tmp_con[k][i-1] = 1.0;
          tmp_con[k][i+1] = 1.0;
          tmp_b[k] = 0.0;
          k++;
        }

        tmp_con[k][n-1] = 1.0;              // z[n-1]-z[n-2] <= 0, i.e. z[n-2]>= z[n-1]
        tmp_con[k][n-2] = -1.0;
        tmp_b[k] = 0.0;
        k++;

        tmp_con[k][n-1] = -1.0;             // -z[n-1] <= 0, i.e. z[n-1] >= 0
        tmp_b[k] = 0.0;

        break;

      case 3: // Convex function with non-transparent core.

// The number of constraints n_con = (icore+1)+1+(n-2-icore)+1+1 = n+2.
// In the non-transparent core z[i] = c2. Note that in ptilr6 I never remove these constraints!!!

        if( *n_con != 0 ) { // Memory was allocated, free it.
          free_matrix( *con, *n_con );
          free_vector( *b );
        }

// Allocate memory.
        *n_con = n+2; // See comments below.
        if( !( tmp_con = matrix( *n_con, n ) ) ) { // Constraints matrix.
          *ierr = 202;
          return;
        }
        if( !( tmp_b = vector( *n_con ) ) ) { // Constraints vector (right-hand side of constraints).
          *ierr = 202;
          free_matrix( tmp_con, *n_con );
          return;
        }

// Just in case - reset the constraints matrix and the right-hand side vector.
        for( i=0; i<*n_con; i++ ) {
          for( j=0; j<n; j++ ) {
            tmp_con[i][j] = 0.0;
          }
          tmp_b[i] = 0.0;
        }

        for(i=0; i<=icore; i++ ) {      // z[i]<=c2
          tmp_con[i][i] = 1.0;
          tmp_b[i] = c2;
        }

        k = icore+1;                    // z[icore]>=z[icore+1] i.e. -z[icore]+z[icore+1] <=0
        tmp_con[k][icore] = -1.0;
        tmp_con[k][k] = 1.0;
        tmp_b[k] = 0.0;
        k++;

        for( i=icore+1; i<n-1; i++ ) {  // Convex - second derivs are positive.
          tmp_con[k][i] = 2.0;
          tmp_con[k][i-1] = -1.0;
          tmp_con[k][i+1] = -1.0;
          tmp_b[k] = 0.0;
          k++;
        }

        tmp_con[k][n-1] = 1.0;              // z[n-1]-z[n-2] <= 0, i.e. z[n-2]>= z[n-1]
        tmp_con[k][n-2] = -1.0;
        tmp_b[k] = 0.0;
        k++;

        tmp_con[k][n-1] = -1.0;             // -z[n-1] <= 0, i.e. z[n-1] >= 0
        tmp_b[k] = 0.0;

        break;

      case 4: // Concave-convex function with non-transparent core.

// The number of constraints n_con = (icore+1)+1+(l-1-icore)+(n-2-l)+1+1 = n+1
// In the non-transparent core z[i] = c2. Note that in ptilr6 I never remove these constraints!!!

        if( *n_con != 0 ) { // Memory was allocated, free it.
          free_matrix( *con, *n_con );
          free_vector( *b );
        }

// Allocate memory.
        *n_con = n+1; // See comments below.
        if( !( tmp_con = matrix( *n_con, n ) ) ) { // Constraints matrix.
          *ierr = 202;
          return;
        }
        if( !( tmp_b = vector( *n_con ) ) ) { // Constraints vector (right-hand side of constraints).
          *ierr = 202;
          free_matrix( tmp_con, *n_con );
          return;
        }

// Just in case - reset the constraints matrix and the right-hand side vector.
        for( i=0; i<*n_con; i++ ) {
          for( j=0; j<n; j++ ) {
            tmp_con[i][j] = 0.0;
          }
          tmp_b[i] = 0.0;
        }

        for(i=0; i<=icore; i++ ) {
          tmp_con[i][i] = 1.0;             // z[i]<=c2
          tmp_b[i] = c2;
        }

        k = icore+1;
        tmp_con[k][icore] = -1.0;          // -z[icore]+z[icore+1] <=0, i.e. z[icore]>=z[icore+1]
        tmp_con[k][icore+1] = 1.0;
        tmp_b[k] = 0.0;
        k++;
        
        for( i=icore+1; i<l; i++ ) {   // Concave part - second derivs are negative.
          tmp_con[k][i] = -2.0;
          tmp_con[k][i-1] = 1.0;
          tmp_con[k][i+1] = 1.0;
          tmp_b[k] = 0.0;
          k++;
        }

        for( i=l+1; i<n-1; i++ ) {     // Convex part - second derivs are positive.
          tmp_con[k][i] = 2.0;
          tmp_con[k][i-1] = -1.0;
          tmp_con[k][i+1] = -1.0;
          tmp_b[k] = 0.0;
          k++;
        }

        tmp_con[k][n-1] = 1.0;             // z[n-1]-z[n-2] <= 0, i.e. z[n-2]>= z[n-1]
        tmp_con[k][n-2] = -1.0;
        tmp_b[k] = 0.0;
        k++;

        tmp_con[k][n-1] = -1.0;            // -z[n-1] <= 0, i.e.  z[n-1] >= 0
        tmp_b[k] = 0.0;
        k++;

        break;
      
      default:
        break;
    } // End switch.

  } // End kernel_type=1.


  if( kernel_type == 2 ) { // Fredholm equation, second eclipse.

    switch( switch_contype ) {

      case 1: // Monotonically non-increasing function.

// The number of constraints = n-1+1 = n

        if( *n_con != 0 ) { // Memory was allocated, free it.
          free_matrix( *con, *n_con );
          free_vector( *b );
        }

// Allocate memory.
        *n_con = n; // See comments below.
        if( !( tmp_con = matrix( *n_con, n ) ) ) { // Constraints matrix.
          *ierr = 202;
          return;
        }
        if( !( tmp_b = vector( *n_con ) ) ) { // Constraints vector (right-hand side of constraints).
          *ierr = 202;
          free_matrix( tmp_con, *n_con );
          return;
        }

// Just in case - reset the constraints matrix and the right-hand side vector.
        for( i=0; i<*n_con; i++ ) {
          for( j=0; j<n; j++ ) {
            tmp_con[i][j] = 0.0;
          }
          tmp_b[i] = 0.0;
        }

        for(i=0; i<n-1; i++) {
          tmp_con[i][i] = -1.0;         // -z[i]+z[i+1] <= 0, i.e. z[i] >= z[i+1]
          tmp_con[i][i+1] = 1.0;
          tmp_b[i] = 0.0;
        }

        tmp_con[n-1][n-1] = -1.0;       // z[n-1] >= 0 i.e. -z[n-1] <= 0
        tmp_b[n-1] = 0.0;

        break;

      case 2:  // Concave function.

// The number of constraints 1+n-2+1 = n.

        if( *n_con != 0 ) { // Memory was allocated, free it.
          free_matrix( *con, *n_con );
          free_vector( *b );
        }

// Allocate memory.
        *n_con = n; // See comments below.
        if( !( tmp_con = matrix( *n_con, n ) ) ) { // Constraints matrix.
          *ierr = 202;
          return;
        }
        if( !( tmp_b = vector( *n_con ) ) ) { // Constraints vector (right-hand side of constraints).
          *ierr = 202;
          free_matrix( tmp_con, *n_con );
          return;
        }

// Just in case - reset the constraints matrix and the right-hand side vector.
        for( i=0; i<*n_con; i++ ) {
          for( j=0; j<n; j++ ) {
            tmp_con[i][j] = 0.0;
          }
          tmp_b[i] = 0.0;
        }

        tmp_con[0][0] = -1.0;            // -z[0]+z[1] <= 0, i.e. z[0] >= z[1]
        tmp_con[0][1] = 1.0;
        tmp_b[0] = 0.0;

        for(i=1; i<n-1; i++) {       // Concave part - second derivs are negative.
          tmp_con[i][i] = -2.0;          // z[i-1]-2z[i]+z[i+1] <= 0
          tmp_con[i][i-1] = 1.0;
          tmp_con[i][i+1] = 1.0;
          tmp_b[i] = 0.0;
        }

    // z[n-1] >= 0
        tmp_con[n-1][n-1] = -1.0;       // -z[n-1] <= 0
        tmp_b[n-1] = 0.0;

        break;

      case 3:  // Convex function.

// The number of constraints 1+n-2+1+1 = n+1.

        if( *n_con != 0 ) { // Memory was allocated, free it.
          free_matrix( *con, *n_con );
          free_vector( *b );
        }

// Allocate memory.
        *n_con = n+1; // See comments below.
        if( !( tmp_con = matrix( *n_con, n ) ) ) { // Constraints matrix.
          *ierr = 202;
          return;
        }
        if( !( tmp_b = vector( *n_con ) ) ) { // Constraints vector (right-hand side of constraints).
          *ierr = 202;
          free_matrix( tmp_con, *n_con );
          return;
        }

// Just in case - reset the constraints matrix and the right-hand side vector.
        for( i=0; i<*n_con; i++ ) {
          for( j=0; j<n; j++ ) {
            tmp_con[i][j] = 0.0;
          }
          tmp_b[i] = 0.0;
        }

        tmp_con[0][0] = -1.0;            // -z[0]+z[1] <= 0, i.e. z[0] >= z[1]
        tmp_con[0][1] = 1.0;
        tmp_b[0] = 0.0;

        for(i=1; i<n-1; i++) {           // Convex part - second derivs are positive.
          tmp_con[i][i] = 2.0;           // -z[i-1]+2z[i]-z[i+1] <= 0
          tmp_con[i][i-1] = -1.0;
          tmp_con[i][i+1] = -1.0;
          tmp_b[i] = 0.0;
        }

        tmp_con[n-1][n-1] = 1.0;         // z[n-1]-z[n-2] <= 0, i.e. z[n-2]>= z[n-1]
        tmp_con[n-1][n-2] = -1.0;
        tmp_b[n-1] = 0.0;

// z[n-1] >= 0
        tmp_con[n][n-1] = -1.0;       // -z[n-1] <= 0
        tmp_b[n] = 0.0;

        break;

      case 4:  // Concave-convex function.
      
// The number of a priori constraintss is n_con = 1+l-1+(n-2-l)+1+1 = n.

        if( *n_con != 0 ) { // Memory was allocated, free it.
          free_matrix( *con, *n_con );
          free_vector( *b );
        }

// Allocate memory.
        *n_con = n; // See comments below.
        if( !( tmp_con = matrix( *n_con, n ) ) ) { // Constraints matrix.
          *ierr = 202;
          return;
        }
        if( !( tmp_b = vector( *n_con ) ) ) { // Constraints vector (right-hand side of constraints).
          *ierr = 202;
          free_matrix( tmp_con, *n_con );
          return;
        }

// Just in case - reset the constraints matrix and the right-hand side vector.
        for( i=0; i<*n_con; i++ ) {
          for( j=0; j<n; j++ ) {
            tmp_con[i][j] = 0.0;
          }
          tmp_b[i] = 0.0;
        }

        tmp_con[0][0] = -1.0;            // -z[0]+z[1] <= 0, i.e. z[0] >= z[1]
        tmp_con[0][1] = 1.0;
        tmp_b[0] = 0.0;

        for(i=1; i<l; i++) {         // Concave part - second derivs are negative.
          tmp_con[i][i] = -2.0;           // z[i-1]-2z[i]+z[i+1] <= 0
          tmp_con[i][i-1] = 1.0;
          tmp_con[i][i+1] = 1.0;
          tmp_b[i] = 0.0;
        }

        for( i=l+1; i<n-1; i++ ) {   // Convex part - second derivs are positive.
          tmp_con[i-1][i] = 2.0;        // -z[i-1]+2z[i]-z[i+1] <= 0
          tmp_con[i-1][i-1] = -1.0;
          tmp_con[i-1][i+1] = -1.0;
          tmp_b[i-1] = 0.0;
        }

        tmp_con[n-2][n-1] = 1.0;         // z[n-1]-z[n-2] <= 0, i.e. z[n-2]>= z[n-1]
        tmp_con[n-2][n-2] = -1.0;
        tmp_b[n-2] = 0.0;

        tmp_con[n-1][n-1] = -1.0;        // -z[n-1] <= 0
        tmp_b[n-1] = 0.0;

        break;

      default:
        break;

    } // End switch.

  } // End kernel_type=2.

  if( kernel_type == 3 ) {  // Abel equation where the unknown function is 1/v(s).

    switch( switch_contype ) {

      case 1: // Monotonically non-increasing function.

// The number of constraints = n-1+1 = n

        if( *n_con != 0 ) { // Memory was allocated, free it.
          free_matrix( *con, *n_con );
          free_vector( *b );
        }

// Allocate memory.
        *n_con = n;
        if( !( tmp_con = matrix( *n_con, n ) ) ) { // Constraints matrix.
          *ierr = 202;
          return;
        }
        if( !( tmp_b = vector( *n_con ) ) ) { // Constraints vector (right-hand side of constraints).
          *ierr = 202;
          free_matrix( tmp_con, *n_con );
          return;
        }

// Just in case - reset the constraints matrix and the right-hand side vector.
        for( i=0; i<*n_con; i++ ) {
          for( j=0; j<n; j++ ) {
            tmp_con[i][j] = 0.0;
          }
          tmp_b[i] = 0.0;
        }

        for(i=0; i<n-1; i++) {
          tmp_con[i][i] = -1.0;         // -z[i]+z[i+1] <= 0, i.e. z[i] >= z[i+1]
          tmp_con[i][i+1] = 1.0;
          tmp_b[i] = 0.0;
        }
        tmp_con[n-1][n-1] = -1.0;       // z[n-1] >= 0 i.e. -z[n-1] <= 0
        tmp_b[n-1] = 0.0;

        break;

      case 2:  // Concave function. выпуклая

// The number of constraints 1+n-2+1 = n.

        if( *n_con != 0 ) { // Memory was allocated, free it.
          free_matrix( *con, *n_con );
          free_vector( *b );
        }

// Allocate memory.
        *n_con = n; // See comments below.
        if( !( tmp_con = matrix( *n_con, n ) ) ) { // Constraints matrix.
          *ierr = 202;
          return;
        }
        if( !( tmp_b = vector( *n_con ) ) ) { // Constraints vector (right-hand side of constraints).
          *ierr = 202;
          free_matrix( tmp_con, *n_con );
          return;
        }

// Just in case - reset the constraints matrix and the right-hand side vector.
        for( i=0; i<*n_con; i++ ) {
          for( j=0; j<n; j++ ) {
            tmp_con[i][j] = 0.0;
          }
          tmp_b[i] = 0.0;
        }

        tmp_con[0][0] = -1.0;            // -z[0]+z[1] <= 0, i.e. z[0] >= z[1]
        tmp_con[0][1] = 1.0;
        tmp_b[0] = 0.0;

        for(i=1; i<n-1; i++) {           // Concave part - second derivs are negative.
          tmp_con[i][i] = -2.0;          // z[i-1]-2z[i]+z[i+1] <= 0
          tmp_con[i][i-1] = 1.0;
          tmp_con[i][i+1] = 1.0;
          tmp_b[i] = 0.0;
        }

    // z[n-1] >= 0
        tmp_con[n-1][n-1] = -1.0;        // -z[n-1] <= 0
        tmp_b[n-1] = 0.0;

        break;

      case 3:  // Convex function. вогнутая

// The number of constraints 1+n-2+1+1 = n+1.

        if( *n_con != 0 ) { // Memory was allocated, free it.
          free_matrix( *con, *n_con );
          free_vector( *b );
        }

// Allocate memory.
        *n_con = n+1;
        if( !( tmp_con = matrix( *n_con, n ) ) ) { // Constraints matrix.
          *ierr = 202;
          return;
        }
        if( !( tmp_b = vector( *n_con ) ) ) { // Constraints vector (right-hand side of constraints).
          *ierr = 202;
          free_matrix( tmp_con, *n_con );
          return;
        }

// Just in case - reset the constraints matrix and the right-hand side vector.
        for( i=0; i<*n_con; i++ ) {
          for( j=0; j<n; j++ ) {
            tmp_con[i][j] = 0.0;
          }
          tmp_b[i] = 0.0;
        }

        tmp_con[0][0] = -1.0;            // -z[0]+z[1] <= 0, i.e. z[0] >= z[1]
        tmp_con[0][1] = 1.0;
        tmp_b[0] = 0.0;

        for(i=1; i<n-1; i++) {       // Convex - second derivs are positive.
          tmp_con[i][i] = 2.0;           // -z[i-1]+2z[i]-z[i+1] <= 0
          tmp_con[i][i-1] = -1.0;
          tmp_con[i][i+1] = -1.0;
          tmp_b[i] = 0.0;
        }

        tmp_con[n-1][n-1] = 1.0;         // z[n-1]-z[n-2] <= 0, i.e. z[n-2]>= z[n-1]
        tmp_con[n-1][n-2] = -1.0;
        tmp_b[n-1] = 0.0;

        tmp_con[n][n-1] = -1.0;          // -z[n-1] <= 0, i.e. z[n-1] >= 0
        tmp_b[n] = 0.0;

        break;

      case 4:  // Concave-convex function.
      
// The number of a priori constraints is n_con = 1+l-1+(n-2)-l+1+1 = n.

        if( *n_con != 0 ) { // Memory was allocated, free it.
          free_matrix( *con, *n_con );
          free_vector( *b );
        }

// Allocate memory.
        *n_con = n;
        if( !( tmp_con = matrix( *n_con, n ) ) ) { // Constraints matrix.
          *ierr = 202;
          return;
        }
        if( !( tmp_b = vector( *n_con ) ) ) { // Constraints vector (right-hand side of constraints).
          *ierr = 202;
          free_matrix( tmp_con, *n_con );
          return;
        }

// Just in case - reset the constraints matrix and the right-hand side vector.
        for( i=0; i<*n_con; i++ ) {
          for( j=0; j<n; j++ ) {
            tmp_con[i][j] = 0.0;
          }
          tmp_b[i] = 0.0;
        }

        tmp_con[0][0] = -1.0;              // -z[0]+z[1] <= 0, i.e. z[0] >= z[1]
        tmp_con[0][1] = 1.0;
        tmp_b[0] = 0.0;

        for(i=1; i<l; i++) {           // Concave part - second derivs are negative.
          tmp_con[i][i] = -2.0;            // z[i-1]-2z[i]+z[i+1] <= 0
          tmp_con[i][i-1] = 1.0;
          tmp_con[i][i+1] = 1.0;
          tmp_b[i] = 0.0;
        }

        for( i=l+1; i<n-1; i++ ) {     // Convex part - second derivs are positive.
          tmp_con[i-1][i] = 2.0;           // -z[i-1]+2z[i]-z[i+1] <= 0, i.e. z[i-1]-2z[i]+z[i+1] >= 0
          tmp_con[i-1][i-1] = -1.0;
          tmp_con[i-1][i+1] = -1.0;
          tmp_b[i-1] = 0.0;
        }

        tmp_con[n-2][n-1] = 1.0;           // z[n-1]-z[n-2] <= 0, i.e. z[n-2]>= z[n-1]
        tmp_con[n-2][n-2] = -1.0;
        tmp_b[n-2] = 0.0;

        tmp_con[n-1][n-1] = -1.0;          // -z[n-1] <= 0
        tmp_b[n-1] = 0.0;

        break;

      case 5: // Non-negative functions. For use with Tikhonov's regularization only.

// The number of constraints = n

        if( *n_con != 0 ) { // Memory was allocated, free it.
          free_matrix( *con, *n_con );
          free_vector( *b );
        }

// Allocate memory.
        *n_con = n;
        if( !( tmp_con = matrix( *n_con, n ) ) ) { // Constraints matrix.
          *ierr = 202;
          return;
        }
        if( !( tmp_b = vector( *n_con ) ) ) { // Constraints vector (right-hand side of constraints).
          *ierr = 202;
          free_matrix( tmp_con, *n_con );
          return;
        }

// Just in case - reset the constraints matrix and the right-hand side vector.
        for( i=0; i<*n_con; i++ ) {
          for( j=0; j<n; j++ ) {
            tmp_con[i][j] = 0.0;
          }
          tmp_b[i] = 0.0;
        }

        for(i=0; i<n; i++) {
          tmp_con[i][i] = -1.0;         // -z[i] <= 0, i.e. z[i] >= 0
          tmp_b[i] = 0.0;
        }

        break;

      default:
        break;

    } // End switch.

  } // End kernel_type=3.

  if( kernel_type == 4 ) { // Fredholm equation, test model. This section is mostly identical to the one with kernel_type=2.

    switch( switch_contype ) {

      case 1: // Monotonically non-increasing function.

// The number of constraints = n-1+1 = n

        if( *n_con != 0 ) { // Memory was allocated, free it.
          free_matrix( *con, *n_con );
          free_vector( *b );
        }

// Allocate memory.
        *n_con = n; // See comments below.
        if( !( tmp_con = matrix( *n_con, n ) ) ) { // Constraints matrix.
          *ierr = 202;
          return;
        }
        if( !( tmp_b = vector( *n_con ) ) ) { // Constraints vector (right-hand side of constraints).
          *ierr = 202;
          free_matrix( tmp_con, *n_con );
          return;
        }

// Just in case - reset the constraints matrix and the right-hand side vector.
        for( i=0; i<*n_con; i++ ) {
          for( j=0; j<n; j++ ) {
            tmp_con[i][j] = 0.0;
          }
          tmp_b[i] = 0.0;
        }

        for(i=0; i<n-1; i++) {
          tmp_con[i][i] = -1.0;         // -z[i]+z[i+1] <= 0, i.e. z[i] >= z[i+1]
          tmp_con[i][i+1] = 1.0;
          tmp_b[i] = 0.0;
        }

        tmp_con[n-1][n-1] = -1.0;       // z[n-1] >= 0 i.e. -z[n-1] <= 0
        tmp_b[n-1] = 0.0;

        break;

      case 2:  // Concave function.

// The number of constraints 1+n-2+1 = n.

        if( *n_con != 0 ) { // Memory was allocated, free it.
          free_matrix( *con, *n_con );
          free_vector( *b );
        }

// Allocate memory.
        *n_con = n; // See comments below.
        if( !( tmp_con = matrix( *n_con, n ) ) ) { // Constraints matrix.
          *ierr = 202;
          return;
        }
        if( !( tmp_b = vector( *n_con ) ) ) { // Constraints vector (right-hand side of constraints).
          *ierr = 202;
          free_matrix( tmp_con, *n_con );
          return;
        }

// Just in case - reset the constraints matrix and the right-hand side vector.
        for( i=0; i<*n_con; i++ ) {
          for( j=0; j<n; j++ ) {
            tmp_con[i][j] = 0.0;
          }
          tmp_b[i] = 0.0;
        }

        tmp_con[0][0] = -1.0;            // -z[0]+z[1] <= 0, i.e. z[0] >= z[1]
        tmp_con[0][1] = 1.0;
        tmp_b[0] = 0.0;

        for(i=1; i<n-1; i++) {       // Concave part - second derivs are negative.
          tmp_con[i][i] = -2.0;          // z[i-1]-2z[i]+z[i+1] <= 0
          tmp_con[i][i-1] = 1.0;
          tmp_con[i][i+1] = 1.0;
          tmp_b[i] = 0.0;
        }

    // z[n-1] >= 0
        tmp_con[n-1][n-1] = -1.0;       // -z[n-1] <= 0
        tmp_b[n-1] = 0.0;

        break;

      case 3:  // Convex function.

// The number of constraints 1+n-2+1+1 = n+1.

        if( *n_con != 0 ) { // Memory was allocated, free it.
          free_matrix( *con, *n_con );
          free_vector( *b );
        }

// Allocate memory.
        *n_con = n+1; // See comments below.
        if( !( tmp_con = matrix( *n_con, n ) ) ) { // Constraints matrix.
          *ierr = 202;
          return;
        }
        if( !( tmp_b = vector( *n_con ) ) ) { // Constraints vector (right-hand side of constraints).
          *ierr = 202;
          free_matrix( tmp_con, *n_con );
          return;
        }

// Just in case - reset the constraints matrix and the right-hand side vector.
        for( i=0; i<*n_con; i++ ) {
          for( j=0; j<n; j++ ) {
            tmp_con[i][j] = 0.0;
          }
          tmp_b[i] = 0.0;
        }

        tmp_con[0][0] = -1.0;            // -z[0]+z[1] <= 0, i.e. z[0] >= z[1]
        tmp_con[0][1] = 1.0;
        tmp_b[0] = 0.0;

        for(i=1; i<n-1; i++) {           // Convex part - second derivs are positive.
          tmp_con[i][i] = 2.0;           // -z[i-1]+2z[i]-z[i+1] <= 0
          tmp_con[i][i-1] = -1.0;
          tmp_con[i][i+1] = -1.0;
          tmp_b[i] = 0.0;
        }

        tmp_con[n-1][n-1] = 1.0;         // z[n-1]-z[n-2] <= 0, i.e. z[n-2]>= z[n-1]
        tmp_con[n-1][n-2] = -1.0;
        tmp_b[n-1] = 0.0;

// z[n-1] >= 0
        tmp_con[n][n-1] = -1.0;       // -z[n-1] <= 0
        tmp_b[n] = 0.0;

        break;

      case 4:  // Concave-convex function.
      
// The number of a priori constraintss is n_con = 1+l-1+(n-2-l)+1+1 = n.

        if( *n_con != 0 ) { // Memory was allocated, free it.
          free_matrix( *con, *n_con );
          free_vector( *b );
        }

// Allocate memory.
        *n_con = n; // See comments below.
        if( !( tmp_con = matrix( *n_con, n ) ) ) { // Constraints matrix.
          *ierr = 202;
          return;
        }
        if( !( tmp_b = vector( *n_con ) ) ) { // Constraints vector (right-hand side of constraints).
          *ierr = 202;
          free_matrix( tmp_con, *n_con );
          return;
        }

// Just in case - reset the constraints matrix and the right-hand side vector.
        for( i=0; i<*n_con; i++ ) {
          for( j=0; j<n; j++ ) {
            tmp_con[i][j] = 0.0;
          }
          tmp_b[i] = 0.0;
        }

        tmp_con[0][0] = -1.0;            // -z[0]+z[1] <= 0, i.e. z[0] >= z[1]
        tmp_con[0][1] = 1.0;
        tmp_b[0] = 0.0;

        for(i=1; i<l; i++) {         // Concave part - second derivs are negative.
          tmp_con[i][i] = -2.0;           // z[i-1]-2z[i]+z[i+1] <= 0
          tmp_con[i][i-1] = 1.0;
          tmp_con[i][i+1] = 1.0;
          tmp_b[i] = 0.0;
        }

        for( i=l+1; i<n-1; i++ ) {   // Convex part - second derivs are positive.
          tmp_con[i-1][i] = 2.0;        // -z[i-1]+2z[i]-z[i+1] <= 0
          tmp_con[i-1][i-1] = -1.0;
          tmp_con[i-1][i+1] = -1.0;
          tmp_b[i-1] = 0.0;
        }

        tmp_con[n-2][n-1] = 1.0;         // z[n-1]-z[n-2] <= 0, i.e. z[n-2]>= z[n-1]
        tmp_con[n-2][n-2] = -1.0;
        tmp_b[n-2] = 0.0;

        tmp_con[n-1][n-1] = -1.0;        // -z[n-1] <= 0
        tmp_b[n-1] = 0.0;

        break;

      case 5: // Non-negative functions. For use with Tikhonov's regularization only.

// The number of constraints = n

        if( *n_con != 0 ) { // Memory was allocated, free it.
          free_matrix( *con, *n_con );
          free_vector( *b );
        }

// Allocate memory.
        *n_con = n;
        if( !( tmp_con = matrix( *n_con, n ) ) ) { // Constraints matrix.
          *ierr = 202;
          return;
        }
        if( !( tmp_b = vector( *n_con ) ) ) { // Constraints vector (right-hand side of constraints).
          *ierr = 202;
          free_matrix( tmp_con, *n_con );
          return;
        }

// Just in case - reset the constraints matrix and the right-hand side vector.
        for( i=0; i<*n_con; i++ ) {
          for( j=0; j<n; j++ ) {
            tmp_con[i][j] = 0.0;
          }
          tmp_b[i] = 0.0;
        }

        for(i=0; i<n; i++) {
          tmp_con[i][i] = -1.0;         // -z[i] <= 0, i.e. z[i] >= 0
          tmp_b[i] = 0.0;
        }

        break;

      default:
        break;

    } // End switch.

  } // End kernel_type=4.

  *con = tmp_con; // Assign local pointers to parameters to return them.
  *b = tmp_b;

  *ierr = 1;
  return;

}
