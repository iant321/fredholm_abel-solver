  /*
  Some common functions for the programs of solving light curves of WR+O binaries and for solving Abel eq,

  kernel1,2 - kernels for Fredholm eqs.
  kernel_abel - kernel for Abel eq.
  kernel_test - kernel for a test case.

  Author: Igor Antokhin (igor@sai.msu.ru)
  Last modification: Sept 2016

*/

#include <stdlib.h>
#include <stdio.h>
#include <malloc.h>
#include <math.h>
#include <string.h>

#define PI M_PI

// Kernels of Fredholm integral equations for the 1st and 2nd minima.
void kernel1( double **a, double *s, double *x1, double rstar, double ax, int n, int m1 );
void kernel2( double **a, double *s, double *x2, double rstar, int n, int m2 ); 

// Kernel of Abel equation.
void kernel_abel( double **a, double *s, double *xi, int n, int m );

// Kernel of the test problem.
void kernel_test( double **a, double *s, double *x, int n, int m );

// Various elliptical integrals needed for kernels 1 and 2.
double cel( double qqc, double pp, double aa, double bb ); // Common elliptical integral
double ellipte( double k );  // Elliptical integral of E-type
double elliptk( double k );  // Elliptical integral of K-type

// Square of x.
double sqr( double x );

void clearstr( char *str, char rem ); // Remove symbol ch from string str.

void kernel1( double **a, double *s, double *x1, double rstar, double ax, int n, int m1 ) {
/*
  Compute the kernel of the fredholm's equation for the 1st minimum.
  
  Vars:

  a[m1][n] - operator matrix
  s[n]     - grid on the WR disk
  x1[m1]   - grid on the distance between the centers of the components
  rstar    - O star radius
  ax       - coefficient of the linear limb darkening for the O star
  n        - size of the grid in s
  m1       - size of the grid in x1
*/

int i, j;
double r1, r2, r3, r4, r5;
// double a[100][100];

  for(j=0; j<n; ++j) {
//    printf( "%le\n", s[j] );
    for(i=0; i<m1; ++i) {
//      if( x1[i] >= 1.0e-9 ) // zero to avoid problems with sqrt(0) ...
      if( x1[i] > 0.0 ) // zero to avoid problems with sqrt(0) ...
        r1 = rstar+x1[i];
      else {
        a[i][j] = 0.0;
        continue;
      }
      if( s[j] >= r1 ) {             // G4 region
        a[i][j] = 0.0;
        continue;
      }
      if( s[j] <= 1.0e-18 ) { // zero to avoid problem with sqrt(0) ...
//        printf( "111111 j=%d i=%d s=%le\n", j, i, s[j] );
        a[i][j] = 0.0;
        continue;
      }
//      printf( "222222 j=%d i=%d\n", j, i );
      r1 = rstar-x1[i];
      if( s[j] <= r1 ) {             // G2 region
        r2 = sqrt(4*s[j]*x1[i]/(rstar*rstar-sqr(s[j]-x1[i]) ));
        r3 = ellipte(r2);
        a[i][j] = 2*PI*(1.0-ax)*s[j]+
                  (4*s[j]/rstar)*ax*sqrt(rstar*rstar-sqr(s[j]-x1[i]) )*r3;
        continue;
      }
      if( s[j] <= -r1 ) {            // G3 region
        a[i][j] = 0.0;
        continue;
      }
      r1 = (s[j]*s[j]+x1[i]*x1[i]-rstar*rstar)/(2*x1[i]*s[j]); // G1 region
      r2 = 1.0-r1*r1;
      if( r2 <= 0.0 )
        r2 = 0.0;
      r3 = sqrt(r2);
      r2 = sqrt((rstar*rstar-sqr(s[j]-x1[i]) )/4/s[j]/x1[i]);
      r4 = ellipte(r2);
      r5 = elliptk(r2);
      r2 = 8*s[j]*ax*sqrt(s[j]*x1[i])*r4/rstar;
      r2 = r2+ax*r5*2*sqrt(s[j])*(rstar*rstar-sqr(s[j]+x1[i]) )/
           rstar/sqrt(x1[i]);
//      if( r1 >= 1.0e-11 ) {
      if( r1 >= 0.0 ) {
        a[i][j] = 2*s[j]*atan(r3/r1)*(1-ax)+r2;
        continue;
      }
//      if( r1 <= -1.0e-12 ) {
      if( r1 < 0.0 ) {
        a[i][j] = 2*s[j]*(PI-atan(-r3/r1))*(1-ax)+r2;
        continue;
      }
      a[i][j] = PI*s[j]*(1-ax)+r2; // When cos=0
    }
  }

  return;

}

void kernel2( double **a, double *s, double *x2, double rstar, int n, int m2 ) {
/*
  Compute the kernel of the fredholm's equation for the 2nd minimum.
  
  Vars:

  a[m2][n] - operator matrix
  s[n]     - grid on the WR disk
  x2[m2]   - grid on the distance between the centers of the components
  rstar    - O star radius
  n        - size of the grid in s
  m2       - size of the grid in x2
*/

int i, j;
double r1, r2, r3;

  for(j=0; j<n; ++j) {
    for(i=0; i<m2; ++i) {
      if( s[j] >= (rstar+x2[i]) ) {   // G4 region
        a[i][j] = 0.0; 
        continue;
      }
      if( s[j] <= 1.0e-19 ) {
        a[i][j] = 0.0;
        continue;
      }
      if( s[j] <= rstar-x2[i] ) {     // G2 region
        a[i][j] = 2*PI*s[j];
        continue;
      }
      if( s[j] <= x2[i]-rstar ) {     // G3 region
        a[i][j] = 0.0;
        continue;
      }
      r1 = (s[j]*s[j]+x2[i]*x2[i]-rstar*rstar)/(2*x2[i]*s[j]);  // G1 region
      r2 = 1-r1*r1;
      if( r2 <= 0 ) 
        r2 = 0.0;
      r3 = sqrt(r2);
//      if( r1 >= 1.0e-12 ) {
      if( r1 > 0.0 ) {
        a[i][j] = 2*s[j]*atan(r3/r1);
        continue;
      }
//      if( r1 < 0.0= -1.0e-12 ) {
      if( r1 < 0.0 ) {
        a[i][j] = 2*s[j]*(PI-atan(-r3/r1));
        continue;
      }
      a[i][j] = PI*s[j]; // When cos=0
    }
  }

  return;

}

double cel( double qqc, double pp, double aa, double bb ) {
// Common elliptical integral.

#define CA   0.0003
#define PIO2 1.5707963268

double a, b, e, f, g;
double em, p, q, qc;
double result;

   if( qqc == 0.0 ) {
      printf("pause in routine CEL\n");
      getchar();
   }
   qc = fabs(qqc);
   a = aa;
   b = bb;
   p = pp;
   e = qc;
   em = 1.0;
   if( p > 0.0 ) {
      p = sqrt(p);
      b = b/p;
   } else {
      f = qc*qc;
      q = 1.0-f;
      g = 1.0-p;
      f = f-p;
      q = q*(b-a*p);
      p = sqrt(f/g);
      a = (a-b)/g;
      b = -q/(g*g*p)+a*p;
   }
beg:
   f = a;
   a = a+b/p;
   g = e/p;
   b = b+f*g;
   b = b+b;
   p = g+p;
   g = em;
   em = qc+em;
   if( fabs(g-qc) > g*CA ) {
      qc = sqrt(e);
      qc = qc+qc;
      e = qc*em;
      goto beg;
   }
   result = PIO2*(b+a*em)/(em*(em+p));
   return result;
}

double ellipte( double k ) {
// Elliptical integral of the E-type.

double kc, a;

  kc = 1-k*k;
  a = cel( kc, 1.0, 1.0, kc*kc );
  return a;
}

double elliptk( double k ) {
// Elliptical integral of the K-type.

double kc, a;

  kc = 1-k*k;
  a = cel(kc, 1.0, 1.0, 1.0);
  return a;
}

void kernel_abel( double **a, double *s, double *xi, int n, int m ) {

/*
 A version for the general use case.

 Kernel of Abel equation, variant 1. Use generalized formula of left rectangles, see my notes.
*/

int i, j;

/*
  printf( "kernel_abel: n= %d m= %d\ns=\n", n, m );
  for( i=0; i<n; i++ ) {
    printf( "%le\n", s[i] );
  }
  printf( "xi=\n" );
  for( i=0; i<m; i++ ) {
    printf( "%le\n", xi[i] );
  }
*/

  for( i = 0; i<m; i++ ) {
    for( j=0; j<n; j++ ) {

// In Abel quadrature formula in this variant, the sum for a given r goes up
// to n-2, at every xi from 0 to m-1.  However, in my function for computing
// Az, the sum is up to n-1.  So to be able to use it, set the corresponding
// row and column in the matrix "a".  At xi=R0 (m=m-1) the interval of
// integration becomes zero, so set corresponding "a" row to 0.
      if( i == m-1 ) {
        a[i][j] = 0.0;
        continue;
      }
      if( j == n-1 ) {
        a[i][j] = 0.0;
        continue;
      }

      if( s[j] < xi[i] ) {
        a[i][j] = 0.0;
      } else {
//        a[i][j] = sqrt(s[j+1]*s[j+1]-xi[i]*xi[i]) - sqrt(s[j]*s[j]-xi[i]*xi[i]);
//        if( s[j] > xi[i] && s[j-1] <= xi[i] ) {
        if( s[j+1] > xi[i] && s[j] <= xi[i] ) {
          a[i][j] = sqrt(s[j+1]*s[j+1]-xi[i]*xi[i]);
        } else {
          a[i][j] = sqrt(s[j+1]*s[j+1]-xi[i]*xi[i]) - sqrt(s[j]*s[j]-xi[i]*xi[i]);
        }      
      }

    }
  }
  return;
}

void kernel_test( double **a, double *s, double *x, int n, int m ) {
// Test kernel identical to the one in the test example in the "brown book".
// and in ../prgrad_orig_test
int i, j;

  for(j=0; j<n; ++j) {
    for(i=0; i<m; ++i) {
      a[i][j] = 1.0/(1.0+100.0*sqr(s[j]-x[i]));
    }
  }

}


double sqr( double x ) {
  return (x)*(x);
}

void clearstr( char *str, char ch ) {

// Clear string "str" from characters "ch".

  char *dst, *src;

  if( !str )
    return;


  dst = strchr( str, ch );
  src = dst;

  if ( dst != NULL ) {
    while ( *dst != '\0' ) {
      while ( *src == ch )
        src++;
      *dst++ = *src++;
    }
  }
  return;
//  return str; May be used if the function type is changed to char *
}
