//
//  Model.cpp
//
//  model name: 3D warped torus
//
//  Adapted by Brando Magnani from Jonathan Goodman's code.
//
/*   This model defines a warped circle in 3D as the intersection
     of two surfaces.  One is a round sphere defined by
          
            | x - c_0 | = r_0
    
     The other surface is an ellipsoid, which is a distorted sphere,
     defined by
     
          sum_k ( x_k - c_{1,k} )^2/s_k^2 = 1
     
*/
#include <iostream>
#include <blaze/Math.h>
#include "model.hpp"
using namespace std;
using namespace blaze;

// DEFAULT Constructor
Model::Model(){}

// PARAMETRIZED Constructor, copy given dimensions into instance variables
Model::Model( int                         d0,    /* dimension of the ambient space */
              int                         m0,    /* number of constraints          */
              double                      r00,   /* radus of sphere                              */
       DynamicVector<double>              s0,    /* dimensional radius numbers for the ellipsoid */
       DynamicMatrix<double, columnMajor> c0){   /* column 0 = cdnter of round sphere            */
                                                 /* column 1 = center of ellipsoid               */
   // building the model object
   d = d0;
   m = m0;
   r = r00;
   s = s0;
   c = c0;
   
   ssq.resize(d); // contains the square of entries of s0, the "radius" parameters of the ellipsoid
   for ( int i = 0; i < d; i++) {
      ssq[i] = s0[i]*s0[i];
   }
}

// COPY Constructor
Model::Model(const Model& M0){
   d   = M0.d;
   m   = M0.m;
   r   = M0.r;
   s   = M0.s;
   c   = M0.c;
   ssq = M0.ssq;
}

DynamicVector<double, columnVector>
Model::q(DynamicVector<double, columnVector> x){
   
   DynamicVector<double, columnVector> disp(d);   // vector from x to c_j (center of sphere j)
   DynamicVector<double, columnVector> q(m);      // constraint function values
   
   //        The sphere
     
   disp = x - column(c,0);
   q[0] = trans(disp)*disp - r*r;
   
   //       The ellipsoid

   double eq = 0.;              // q[1] = eq = q "for the ellipsoid'
   disp = x - column(c,1);
   for ( int j =0; j < d; j++){
      eq += disp[j]*disp[j]/ssq[j];
   }
   q[1] = eq - 1.;
   
   return q;
}

DynamicMatrix<double, columnMajor>
Model::gq(DynamicVector<double, columnVector> x){
   
   DynamicVector<double, columnVector>disp(d);   // displacement from a center
   DynamicMatrix<double, columnMajor> gq(d,m);   // gradient, (dxm) matrix
   
   //        The sphere, first column
     
   disp = x - column(c,0);
   column(gq, 0)  = 2.*disp;
   
   //       The ellipsoid, second column

   disp = x - column(c,1);
   for ( int j =0; j < d; j++){
      gq(j,1) = 2.0*disp[j]/ssq[j];
   }
   
   return gq;
}

// Returns gq(x) augmented to a square matrix (in case d > m), last n=d-m columns are just zeros.
// Needed to have a complete QR decomposition of gq(x), with Q (dxd) matrix
DynamicMatrix<double, columnMajor>
Model::Agq(DynamicMatrix<double, columnMajor> gq){
   
   DynamicMatrix<double, columnMajor> Agq(d,d);   // Augmented gradient, (dxd) matrix
   Agq = 0.;  // initialize to zero
   
   for (int j = 0; j < m; j++){
      column(Agq, j)  = column(gq, j);
   }

   return Agq;  // last n=d-m columns are zeros
}

string Model::ModelName(){                  /* Return the name of the model */
   
   return(" 3D Warped Torus ");
}

//  Compute the (un-normalized) probability density for x1 by integrating over
//  the other two variables.
 
double Model::yzIntegrate( double x, double L, double R, double eps, int n){

//  Use the rectangle rule in the y and z directions with n midpoints in each direction

   double dy  = (R-L)/( (double) n);
   double dz  = dy;
   double y, z;
   double sum = 0.;
   
   DynamicVector<double, columnVector> xv( 3);    // point in 3D
   DynamicVector<double, columnVector> qv( 2);    // values of the constraint functions
   
   xv[0] = x;
   for ( int j = 0; j < n; j++){
      y = L + j*dy + .5*dy;
      xv[1] = y;
      for ( int k = 0; k < n; k++) {
         z = L + k*dz + .5*dz;
         xv[2] = z;
         qv = q(xv);
         sum += exp( - 0.5*(trans(qv)*qv)/(eps*eps) );
      }
   }
   return dy*dz*sum;
}
