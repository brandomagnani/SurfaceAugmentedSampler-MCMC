/*
       Squishy sampler project, the Surface Augmented Sampler approach.
       See main.cpp for more information.
       
       SAS.cpp
*/

#include <iostream>
#include <blaze/Math.h>   // numerical linear algebra and matrix/vector classes
#include <random>         // for the random number generators
#include <cmath>          // defines exp(..), M_PI
#include <string>
#include "model.hpp"
#include "SAS.hpp"
using namespace std;
using namespace blaze;



  
void SASampler(      vector<double> &Schain,         /* Soft Samples output from the MCMC run, pre-allocated, length = d*T */
                     struct SamplerStats *stats,     /* statistics about different kinds of rections                       */
                     size_t T,                       /* number of MCMC steps        */
                     double eps,                     /* squish parameter            */
                     double p_hard,                  /* probability of Hard move    */
                     double p_soft,                  /* probability of Soft move    */
                     DynamicVector<double>& x0,      /* starting point              */
                     Model M,                        /* evaluate q(x) and grad q(x) */
                     double cf,                      /* constant multiplying density of surface measure    */
                     double sh,                      /* isotropic gaussian size for Hard move              */
                     double sn,                      /* isotropic gaussian size for Off move, Vn part      */
                     double st,                      /* isotropic gaussian size for Off move, Vt part      */
                     double son,                     /* isotropic gaussian size for On move                */
                     double ss,                      /* isotropic gaussian size for Soft move              */
                     double neps,                    /* convergence tolerance for Newton projection        */
                     double rrc,                     /* closeness criterion for the reverse check          */
                     int   itm,                      /* maximum number of Newton iterations per projection */
                     int debug_off,                  /* if = 1, then prints data on Metropolis ratio for Off move */
                     int debug_on,                   /* if = 1, then prints data on Metropolis ratio for On  move */
                     mt19937 RG) {                   /* random generator engine, already instantiated      */
   
                     
   DynamicVector<double, columnVector> qDummy = M.q(x0);// qStart is used only to learn m
   int d  = x0.size();      // infer dimensions, d = dimension of ambient space
   int m  = qDummy.size();  // m = number of constraints
   int n  = d - m;          // n = dimension of constraint Surface (ie, dimension of its tangent space)
   
   DynamicVector<double, columnVector> x(d);       // current sample
   DynamicVector<double, columnVector> y(d);       // proposed new sample
   
   DynamicVector<double, columnVector> ys(d);      // ys = projection of Off proposal y onto constraint surface ..
   DynamicMatrix<double, columnMajor> gqys(d,m);   // .. used for reverse move (wrt Off move)
   DynamicMatrix<double, columnMajor> gtysgy(m,m); // .. used for reverse move (wrt Off move)
   DynamicMatrix<double, columnMajor> Qx(d,d);     // Q (orthonormal) in QR decomposition of gq
   DynamicMatrix<double, columnMajor> Rx(d,m);     // R (upper triangular) in QR decomposition of gq
   DynamicMatrix<double, columnMajor> Qy(d,d);     // Q (orthonormal) in QR decomposition of gq
   DynamicMatrix<double, columnMajor> Ry(d,m);     // R (upper triangular) in QR decomposition of gq
   DynamicMatrix<double, columnMajor> Tx(d,n);     // for basis of tangent space
   DynamicMatrix<double, columnMajor> Ty(d,n);     // orthonormal for basis of tangent space
   DynamicMatrix<double, columnMajor> Bx(d,m);     // basis of normal space at x
   DynamicVector<double, columnVector> xs(d);      // projection of current state x onto hard surface, used for On move
   DynamicMatrix<double, columnMajor> gqxs(d,m);   // gq(xs) used for On move
   DynamicMatrix<double, columnMajor> gtxsgx(m,m); // gq(xs)^t gq(x) used for On move
   
   DynamicVector<double, columnVector> qx(m);     // constraint function values at x
   DynamicVector<double, columnVector> qy(m);     // constraint function values at x
   DynamicMatrix<double, columnMajor> gqx(d,m);   // gradient matrix of constraint functions
   DynamicMatrix<double, columnMajor> gqy(d,m);   // for surface sampler
   DynamicMatrix<double, columnMajor> gtg(m,m);   // (grad)^t*grad, surface sampler volume element
   DynamicMatrix<double, columnMajor> L(m,m);     // Cholesky factor of gtg
   DynamicMatrix<double, columnMajor> gtygx(m,m); // grad(q)^t(y)*grad(q)(x), in Newton projection
   
   DynamicVector<double, columnVector> xr(d);     // for reverse check, project back to x and see if you get x
   DynamicMatrix<double, columnMajor> gqxr(d,m);  // gq(xr), like gqy, but for the reverse check
   DynamicMatrix<double, columnMajor> gtxrgy(m,m); // grad(q)^t(xr)*grad(q)(y), in Newton projection for Reverse check
   
   normal_distribution<double>       SN(0.,1.);   // standard normal (mean zero, variance 1)
   uniform_real_distribution<double> SU(0.,1.);   // standard uniform [0,1]
   DynamicVector<double, columnVector> Z(d);      // isotropic mean zero normal in the ambient space
   DynamicVector<double, columnVector> Vn(m);     // Off move step in normal direction, N(0,sn^2)
   DynamicVector<double, columnVector> Vt(n);     // Off move step in tangent direction, N(0,st^2)
   DynamicVector<double, columnVector> b(m);      // gqx^t*Z = component of Z normal to constraint tangent plane
   DynamicVector<double, columnVector> a(m);      // coefficients of grad(q), as in gqx*a
   DynamicVector<double, columnVector> da(m);     // increment of a in Newton's iteration
   DynamicVector<double, columnVector> w(m);      // intermediate vector for LL^ta = b, Lw=b, L^ta = w
   DynamicVector<double, columnVector> v(d);      // proposal, in the tangent space at x, to be projected
   DynamicVector<double, columnVector> r(m);      // residual in Newton's step for surface sampler
   DynamicVector<double, columnVector> Sn(d);     // Off move step in normal direction
   DynamicVector<double, columnVector> St(d);     // Off move step in tangent direction
   DynamicVector<double, columnVector> R(n);      // isotropic Gaussian of size sh used for Hard move

   
   double detTytTx;     // det( T_y^t T_x ), used in Off / On move for Jacobian factor
   double p_ratio;      // ratio or probabilities in Metropolis-Hastings ratio (for Off / On moves)
   
   
   double vxyn;                  // |v|^2 for the v used to go from x to y (we call this vxyn) ..
   double vyxn;                  // .. and |v'|^2 for the v' used to go from y to x (we call this vyxn) in Surface Sampler move
   double A;                     // for the Metropolis ratio
   double Uy;                    // U(y) = |q(y)|^2, used in Metropolis ratio Off move, pdf = (1/Z)e^{-U/2*eps^2}
   double Ux;                    // U(x) = |q(x)|^2, used in Metropolis ratio On move, pdf = (1/Z)e^{-U/2*eps^2}
   double vn;                    // | v |^2,  used in Metropolis ratio Off / On move
   double Vnn;                   // | Vn |^2, used in Metropolis ratio Off / On move
   double Vtn;                   // | Vt |^2, used in Metropolis ratio Off / On move
   double nconst;                // densities' normalizating constants ratio, used in Metropolis ratio Off / On move
   
   /* Given p_hard and p_soft as inputs for the sampler, we need probabilities for the complements */
   double p_off;     // probability of drawing an  Off move (when current point x is on the surface, this move jumps Off surf.)
   double p_on;      // probability of drawing an  On move  (when current point x is off the surface, this move jumps On surf.)
   
   int       ssFlag;                             // a flag describing where in the Surface Sampler move you are
   const int starting_ss             = 1;            // possible values: have not started the y projection
   const int y_proj_worked_ss        = 2;            // the Newton iteration found y on the surface q(y)=q(x)
   const int y_proj_failed_ss        = 3;            // the y projection Newton iteration ended without success
   const int Met_rej_ss              = 4;            // the proposed y was rejected by Metropolis
   const int Met_acc_ss              = 5;            // the proposed y was accepted by Metropolis
   const int reverse_check_failed_ss = 6;            // the reverse check iteration failed or found a point that isn't x
   const int reverse_check_worked_ss = 7;            // the reverse check iteration found x
   const int accept_y_ss             = 8;            // accepted the new point
   
   int       offFlag;                             // a flag describing where in the OFF move step you are
   const int starting_off              = 1;            // possible values: have not started the ys projection
   const int ys_proj_worked_off        = 2;            // the Newton iteration found ys on the surface q(ys)=0
   const int ys_proj_failed_off        = 3;            // the ys projection Newton iteration ended without success
   const int Met_rej_off               = 4;            // the proposed y was rejected by Metropolis
   const int Met_acc_off               = 5;            // the proposed y was accepted by Metropolis
   const int reverse_check_failed_off  = 6;            // the reverse check iteration failed or found a point that isn't x
   const int reverse_check_worked_off  = 7;            // the reverse check iteration found x
   const int accept_y_off              = 8;            // accepted the new point
   
   int       onFlag;                             // a flag describing where in the ON move step you are
   const int starting_on              = 1;            // possible values: have not started the ys projection
   const int xs_proj_worked_on        = 2;            // the Newton iteration found xs on the surface, q(xs)=0
   const int xs_proj_failed_on        = 3;            // the xs projection Newton iteration ended without success
   const int y_proj_worked_on         = 4;            // the Newton iteration found y on the surface, q(y)=0
   const int y_proj_failed_on         = 5;            // the y projection Newton iteration ended without success
   const int Met_rej_on               = 6;            // the proposed y was rejected by Metropolis
   const int Met_acc_on               = 7;            // the proposed y was accepted by Metropolis
   const int accept_y_on              = 8;            // accepted the new point
   
   int       softFlag;                             // a flag describing where in the SOFT move step you are
   const int Met_rej_soft               = 1;            // the proposed y was rejected by Metropolis
   const int Met_acc_soft               = 2;            // the proposed y was accepted by Metropolis
   const int accept_y_soft              = 3;            // accepted the new point
   
   DynamicMatrix<double, columnMajor> Agqys(d,d);   // augmented gq(ys) used for Off move 'full' QR decomposition
   DynamicMatrix<double, columnMajor> Agqx(d,d);    // augmented gq(x) used for Off move 'full' QR decomposition
   DynamicMatrix<double, columnMajor> Agqxs(d,d);   // augmented gq(xs) used for On move 'full' QR decomposition
   DynamicMatrix<double, columnMajor> Agqy(d,d);    // augmented gq(y) used for On move 'full' QR decomposition
   
   DynamicMatrix<double, columnMajor>  U;          // U contains left singular vectors, size (d x d)
   DynamicVector<double, columnVector> s;          // vector contains m singular values, size m
   DynamicMatrix<double, columnMajor>  Vtr;        // V^t where V contains right singular vectors, size (m x m)
   DynamicMatrix<double, columnMajor>  Sinv(m,m);    // Inverse of Reduced Singular Values matrix
   double detx, dety;    // detx = sqrt( det(gtg) ) = det(S) where S is the singular value matrix in reduced SVD of ( gq * gq^t )
   double fx, fy;        // density with respect to surface measure, fx = sqrt( (2*pi)^m sn^2m ) / detx
   
   stats-> HardSample                        = 0;
   stats-> HardSampleAccepted                = 0;
   stats-> HardRejectionFailedProjection_y   = 0;     // Hard move: failed projection from x+v in T_x to y
   stats-> HardRejectionMetropolis           = 0;
   stats-> HardRejectionReverseCheck         = 0;
   
   stats-> OffSample                         = 0;
   stats-> OffSampleAccepted                 = 0;
   stats-> OffRejectionProjection_ys         = 0;     // Off move: failed projection from proposal y to ys (used for rev. prop. dens.ty)
   stats-> OffRejectionMetropolis            = 0;
   stats-> OffRejectionReverseCheck          = 0;
   
   stats-> SoftSample                        = 0;
   stats-> SoftSampleAccepted                = 0;
   stats-> SoftRejectionMetropolis           = 0;
   
   stats-> OnSample                          = 0;
   stats-> OnSampleAccepted                  = 0;
   stats-> OnRejectionFailedProjection_xs    = 0;     // On move: failed projection from current x to xs
   stats-> OnRejectionFailedProjection_y     = 0;     // On move: failed projection from xs + v in T_xs to y on Surface
   stats-> OnRejectionMetropolis             = 0;
   
   int l = -1;               // number of samples coming from Soft moves / Accepted Off moves / Rejected On moves ..
                             // .. l = stats.SoftSample + stats.OffSampleAccepted + (stats.OnSample - stats.OnSampleAccepted) - 1.
                             // "l" is the index used to is used to update Schain
   
   p_off  = 1.0 - p_hard;    // probability of drawing an Off move type. Complement event of drawing a Hard move type.
   p_on   = 1.0 - p_soft;    // probability of drawing an On  move type. Complement event of drawing a Soft move type.
   Sinv   = 0.0;             // initialize to zero Siv = inverse of Reduced Singular Values matrix
   
   
//    Setup for the MCMC iteration: get values at the starting point
   
   // Update these at the end of each move if proposal is accepted
   x   = x0;       // starting point
   qx  = M.q(x);   // constraint function evaluated at starting point
   gqx = M.gq(x);  // gradient of constraint function at starting point
   
   
//    Start MCMC loop
   
   for (unsigned int iter = 0; iter < T; iter++){
      
//    draw Move Type and generate the move
      
      if (norm(qx) < neps){   // if magnitude of qx is zero, we are on hard surface
         
         if ( SU(RG) < p_hard ){  // draw a Hard move with probability p_hard
            
            stats-> HardSample++;

//-------------------------------------------Hard move (Surface Sampler move) -----------------------------------------------------------
            
            ssFlag = starting_ss;
            
         //          Generate the tangent space forward move v.
            for ( unsigned int k = 0; k < n; k++){   // Isotropic Gaussian, not tangent
               R[k] = SN(RG);
            }
            R = sh*R;                // Isotropic, length scale sh
            
         //       Compute Tx =  basis for tangent space at x. To do so, calculate Full SVD of gqx = U * S * V^t. Then,
         //       .. Tx = last d-m columns of U in the Full SVD for gqx
            
            Agqx = M.Agq(gqx);        // add d-n column of zeros to gqx to get full SVD, needed to get Tx = last d-n columns of U
            svd( Agqx, U, s, Vtr);    // Computing the singular values and vectors of gqx
            
            // multiply singular values to get detx = sqrt( det(gqx^t gqx) ) = det(S)
            detx = 1.;
            for ( unsigned long int i = 0; i < m; i++){
               detx *= s[i];    // detx = det(S) = sqrt( det( gqx^t * gqx ) )
            }
            
            //    Build Tx matrix for tangent space basis at x
            for ( unsigned long int i = m; i < d; i++){
               unsigned long int k = i-m;
               column(Tx,k) = column(U,i);
            }      // end of calculation for T_x
            
            v    = Tx * R;                      //   in the x tangent space
            vxyn = sqrNorm(v);                   //   for the Metropolis ratio
            
         //    Generate the proposal, y, by projecting x+v onto the constraint surface
         //    Newton loop to find y = x+v+grad(q)(x)*a with q(y)=q(x)=0
            
            y = x + v;               // initial guess = move in the tangent direction
            gqy = M.gq(y);           // because these are calculated at the end of this loop
            for ( int ni = 0; ni < itm; ni++){
               r     = - M.q(y);              // equation residual
               gtygx = trans( gqy )*gqx;      // Newton Jacobian matrix
               solve( gtygx, da, r);
               y  += gqx*da;                   // take the Newton step
               gqy = M.gq(y);                  // constraint gradient at the new point (for later)
               if ( norm(r) <= neps ) {
                  ssFlag = y_proj_worked_ss;     // record that you found y
                  break;                      // stop the Newton iteration
               }   // end of if ( norm(r) ...
            }       // end of Newton solver loop
            if ( ssFlag == starting_ss ) {        // the Newton iteration failed, or the flag would be: y_proj_worked
               ssFlag = y_proj_failed_ss;         // done with this surface sampler step
               stats->HardRejectionFailedProjection_y++;
            }                                 // end of the proposal generation phase

         //    Do the Metropolis detailed balance check first, then the reverse check
         //    First find the v in the tangent space at y that lies "over" x
            
            if ( ssFlag == y_proj_worked_ss ) {   //  the next step is: find the tangent vector from y
               
            //       Compute Ty =  basis for tangent space at y. To do so, calculate Full SVD of gqy = U * S * V^t. Then,
            //       .. Ty = last d-m columns of U in the Full SVD for gqx
               
               Agqy = M.Agq(gqy);        // add d-n column of zeros to gqx to get full SVD, needed to get Ty = last d-n columns of U
               svd( Agqy, U, s, Vtr);    // Computing the singular values and vectors of gqy
               
               // multiply singular values to get dety = sqrt( det(gqy^t gqy) ) = det(S)
               dety = 1.;
               for ( unsigned long int i = 0; i < m; i++){
                  dety *= s[i];    // dety = det(S) = sqrt( det( gqy^t * gqy ) )
               }
               
               //    Build Ty matrix for tangent space basis at y
               for ( unsigned long int i = m; i < d; i++){
                  unsigned long int k = i-m;
                  column(Ty,k) = column(U,i);
               }      // end of calculation for T_x
               
               v = Ty * (trans(Ty) * (x-y));    //  v in the y tangent space, since x-y = Ty*a + [(x-y) - Ty*a], where a = Ty^t * (x-y)
               vyxn = sqrNorm(v);               //  the other part of the Metropolis ratio
               
               A    = exp( .5*( vxyn - vyxn ) / (sh*sh) );   //  part of the Metropolis ratio
               A   *= (detx/dety);  // since fy/fx = detx/dety
               
               if ( A < SU(RG) ) {                           // Accept with probability A,
                  ssFlag = Met_rej_ss;                       // rejected
                  stats->HardRejectionMetropolis++;
               }
               else{
                  ssFlag = Met_acc_ss;                       // accepted
               }
            }  // Metropolis rejection step done
            
         //    Reverse check solve: does Newton converge to x from y + tangent vector at y?
            
            if ( ssFlag == Met_acc_ss ) {     //  the next step is: find the tangent vector from y

               xr   = y + v;               // initial guess = move in the tangent direction
               gqxr = M.gq(xr);            // because these are calculated at the end of this loop
               for ( int ni = 0; ni < itm; ni++){
                  r     = - M.q(xr);              // equation residual.  qx = qy = 0 from the forward solve
                  gtxrgy = trans( gqxr )*gqy;     // Newton Jacobian matrix, using old storage
                  solve( gtxrgy, da, r);
                  xr  += gqy*da;                  // take the Newton step
                  gqxr = M.gq(xr);               // constraint gradient at the new point (for later)
                  if ( norm(r) <= neps ) {       // If the reverse check Newton iteration converged
                     if ( norm( xr - x ) < rrc ) {      // did it converge to the right point?
                        ssFlag = reverse_check_worked_ss;
                     }
                     else {
                        ssFlag = reverse_check_failed_ss;   // converged to the wrong point -- a failure
                     }
                     break;                              // stop the Newton iteration, it converged
                  }
               }       // end of Newton solver loop
               if ( ssFlag != reverse_check_worked_ss ) {   // If you did all your Newton iterations without converging
                  ssFlag = reverse_check_failed_ss;         // that's another kind of reverse check failure
                  stats->HardRejectionReverseCheck++;
               }
            }
            if ( ssFlag == reverse_check_worked_ss ) {     //  process an accepted proposal
               x   = y;
               gqx = gqy;      // update gradiet
               qx  = M.q(y);   // update constraint function
               ssFlag = accept_y_ss;
               stats-> HardSampleAccepted++;
            }
            else {                                       // process a rejected proposal
            }     // end this surface sampler move
            
         // UN-COMMENT this when running the SoftSampler with p_hard = 1.0
            //l++;
            //for ( int k = 0; k < d; k++){
               //Schain[ k + d*l] = x[k];
            //}
            
         }  // end of Hard move
         
         else{   // draw an Off move with probability p_off

            stats-> OffSample++;
            
//--------------------------------------------Off move (we jump off the constraint surface)----------------------------------------------
            
            offFlag = starting_off;
            
         //       Produce a proposal y = x + ( Bx * Vn + Tx * Vt )
         //       First, draw isotropic mean zero gaussians Vn with scale sn, and Vt with scale st
            
            for (unsigned int k = 0; k < m; k++){  // Sample Isotropic Standard Gaussian, for Normal space move
               Vn[k] = SN(RG);
            }
            Vn = sn*Vn;
            
            for (unsigned int k = 0; k < n; k++){  // Sample Isotropic Standard Gaussian, for Tangent space move
               Vt[k] = SN(RG);
            }
            Vt = st*Vt;
         
         //       Compute Bx =  basis for normal space at x. To do so, calculate Full SVD of gqx = U * S * V^t. Then,
         //       1- Tx = last d-m columns of U in the Full SVD for gqx
         //       2- resize SVD matrices to get Reduced SVD. Then, let Bx = U S^-1 V^t = transposed pseudoinverse of gqx
            
            Agqx = M.Agq(gqx);        // add d-n column of zeros to gqx to get full SVD, needed to get Tx = last d-n columns of U
            svd( Agqx, U, s, Vtr);    // Computing the singular values and vectors of gqx
            
            //    Build S matrix containing singular values, its the reduced version of size (m x m)
            detx = 1.;
            for ( unsigned long int i = 0; i < m; i++){
               Sinv(i,i) = 1.0 / s[i];
               detx *= s[i];    // detx = det(S) = sqrt( det( gqx^t * gqx ) )
            }
            
            //    Build Tx matrix for tangent space basis at x
            for ( unsigned long int i = m; i < d; i++){
               unsigned long int k = i-m;
               column(Tx,k) = column(U,i);
            }      // end of calculation for T_x
            
            //    Produce matrices for reduced SVD for gqx and get Bx, basis for normal space at x
            U.resize(d,m,true);     // reduced U in SVD for gqx
            Vtr.resize(m,m,true);   // reduced V in SVD for gqx
            Bx = U * Sinv * Vtr;    // basis for normal space at x, used for off step in normal direction
            
         // Proposal point y, combination of normal and tangential moves:
            y = x + Bx*Vn + Tx*Vt;
            

         //       Need to evaluate density of proposal for reverse move p_On(x,y) = h(v;ys) |det(T_ys^t T_x)| ..
         //       where 1- ys is the projection of y on the surface in the direction of gq(y),
         //             2- x = ys + v + w, where v in T_ys and w in orth(T_ys)
         //       Thus, to evaluate density of proposal for reverse move need: ys, v, T_ys, T_x

         //       First, find 'ys'. We use Newton method to find 'a' such that q( y + gq(y)a ) = 0, then set ys := y + gq(y)a.
            
            ys = y;                  // initial guess = start from proposal point y ( that is, guess a=0 )
            gqys = M.gq(ys);         // because these are needed in loop
            gqy  = M.gq(y);          // because these are needed in loop (also used when processing accepted proposal)
            for ( int ni = 0; ni < itm; ni++){
               r     = - M.q(ys);             // residual, want this to be = 0
               gtysgy = trans( gqys )*gqy;    // Newton Jacobian matrix
               solve( gtysgy, da, r);         // solve Newton'm system for da: gq(ys)^t gq(y) da = - q(ys)
               ys = ys + gqy*da;              // take the Newton step
               gqys = M.gq(ys);               // constraint gradient at the new point ys (for later)
               if ( norm(r) <= neps ){
                  offFlag = ys_proj_worked_off;   // record that you found ys
                  break;                      // stop the Newton iteration
               }     // end of if
            }     // end of Newton solver loop
            if ( offFlag == starting_off ) {        // the Newton iteration failed, or the flag would be: ys_proj_worked_off
               offFlag = ys_proj_failed_off;
               stats-> OffRejectionProjection_ys++;
            }                        // end of the ys projection
            
         //       Do the Metropolis detailed balance check, only if Newton successfully found ys
            
            if ( offFlag == ys_proj_worked_off ) {
            
            //    Calculate T_ys: last n=d-m columns of Qy in the QR decomposition of gq(ys) = Qy Ry
               
               Agqys = M.Agq(gqys);      // augment with column of zeros to get full QR
               qr( Agqys, Qy, Ry );      // QR decomposition of gq(ys) (where Qy orthonorm, Ry upp triang)

               for ( unsigned long int i = m; i < d; i++){
                  unsigned long int k = i-m;
                  column(Ty,k) = column(Qy,i);
               }      // end of calculation for T_ys
               
            //    Calculate reverse tangent step v:    x-ys = Ty * a + [ x-ys - Ty * a ], where a = Ty^t * (x-ys) ..
            //    .. so we set v = Ty [ Ty^t (x-ys) ]     (where Ty is the basis matrix for T_ys)
               
               v = Ty * ( trans(Ty) * (x-ys) );
               
         //    Do the Metropolis detailed balance check here
               
               qy = M.q(y);                          // evaluate q(y), also used when processing accepted proposal
               detTytTx = det(trans(Ty)*Tx);         // det( T_ys^t T_x)
               p_ratio  = p_on / p_off;              // ratio of probabilities
               Uy       = sqrNorm(qy);               // |q(y)|^2
               vn       = sqrNorm(v);                // | v |^2
               Vnn      = sqrNorm(Vn);               // | Vn |^2 =  (Bx*Vn)^t * gqx * gqx * (Bx*Vn) = Vn^t * V * V^t * Vn
               Vtn      = sqrNorm(Vt);               // | Vt |^2
               fx = cf / detx;              // surface density at x
               
               nconst   = ( ( pow( 2.0*M_PI, ((double) m) / 2.0 ) *
                            pow( sn, (double) m ) * pow( st / son, (double) n) ) ) / detx; // nconst is the ratio of normalizing ..
                                                                                           // .. constants for densities of proposals
               
               A  = exp( 0.5*( -(Uy / (eps*eps)) - (vn / (son*son)) + (Vnn / (sn*sn)) + (Vtn / (st*st)) ) )
                        * nconst * p_ratio * (abs(detTytTx)) * (1.0 / fx);  //  Metropolis ratio (where we use the ..
                                                                            // .. uniform measure on the hard surface)
               
               if ( debug_off == 1 ) {
                  cout << "--------------------------------------" << endl;
                  cout << " OFF MOVE : " << endl;
                  cout << " exp      = " << exp(-(0.5*Uy/(eps*eps))-(0.5*vn/(son*son))+(0.5*Vnn/(sn*sn))+(0.5*Vtn/(st*st))) << endl;
                  cout << " exp - Uy / 2 eps^2     = " << exp( -(0.5*Uy/(eps*eps)) ) << endl;
                  cout << " exp -|v|^2 / 2 son^2   = " << exp( - (0.5*vn/(son*son)) ) << endl;
                  cout << " exp  |Vn|^2 / 2 sn^2   = " << exp( (0.5*Vnn/(sn*sn)) ) << endl;
                  cout << " exp  |Vt|^2 / 2 st^2   = " << exp( (0.5*Vtn/(st*st)) ) << endl;
                  cout << " nconst / fx            = " << nconst / fx              << endl;
                  cout << " nconst * p_ratio / fx  = " << (nconst * p_ratio) / fx  << endl;
                  cout << " p_ratio        = " << p_ratio       << endl;
                  cout << " det(Tys^t Tx ) = " << abs(detTytTx) << endl;
                  cout << " detSx          = " << detx << endl;
                  cout << " "                          << endl;
                  cout << " Acceptance Pr  = " << A    << endl;
                  cout << " "                          << endl;
                  cout << " y[0]           = " << y[0] << endl;
                  cout << " "                          << endl;
                  cout << "--------------------------------------" << endl;
               }
               
               if ( SU(RG) > A ){      // Accept with probability A,
                  offFlag = Met_rej_off;    // rejected
                  stats-> OffRejectionMetropolis++;
               }
               else{
                  offFlag = Met_acc_off;    // accepted
               }
            }
            
         //    Do the reverse check as last step (more expensive): does ys + v converge to x ? (v is tangent space move from ys)
                  
            if ( offFlag == Met_acc_off ) {  //  the next step is: project back to the manifold, initial guess is ys + v
               xr = ys + v;                //  v was updated above, it is now the tangent space move starting from ys
               gqxr = M.gq(xr);            //  because these are calculated at the end of this loop
               for ( int ni = 0; ni < itm; ni++){
                  r      = - M.q(xr);              // residual, want this to be = 0
                  gtxrgy = trans( gqxr )*gqys;     // Newton Jacobian matrix
                  solve( gtxrgy, da, r);           // solve Newton's system for da: gq(xr)^t gq(ys) da = - q(xr)
                  xr = xr + gqys*da;               // take the Newton step
                  gqxr = M.gq(xr);                 // constraint gradient at the new point (for later)
                  if ( norm(r) <= neps ) {         // If the reverse check Newton iteration converged
                     if ( norm( xr - x ) < rrc ) {      // did it converge to the right point?
                        offFlag = reverse_check_worked_off;
                     }
                     else{
                        offFlag = reverse_check_failed_off;   // converged to the wrong point -- a failure
                     }
                     break;
                  }
               }    // end of Newton solver loop
               if ( offFlag != reverse_check_worked_off ) {   // If you did all your Newton iterations without converging
                  offFlag = reverse_check_failed_off;         // that's another kind of reverse check failure
                  stats-> OffRejectionReverseCheck++;
               }
            }     // Done with reverse check
            if ( offFlag == reverse_check_worked_off ) {     //  process an accepted proposal
               x   = y;
               gqx = gqy;      // update gradiet
               qx  = qy;       // update constraint function
               offFlag = accept_y_off;
               stats-> OffSampleAccepted++;
               
               l++;            // Off sample accepted --> increment index for Schain[]
               for ( int k = 0; k < d; k++){    // since Off sample is accepted, add it to Schain here
                  Schain[ k + d*l]  = x[k];    // value of x[k] where x is l-th Soft / Acc.Off / Rej.On sample 
               }
               
            }
            else {                                       // process a rejected proposal
            }
            
         }   // end of Off move
      }  // end of Hard / Off moves section
      
      else{          // else we are off the Surface
         
         
         if ( SU(RG) < p_soft ){  // draw a Soft move with probability p_soft
            
            stats-> SoftSample++;
            
//------------------------------Soft move (we move in ambient space), simple Gauss. Metropolis step in R^d-------------------------------
            
         // Draw proposal y: isotropic gaussian ( std = ss ) in ambient space
            
            for (unsigned int k = 0; k < d; k++){  // Sample Isotropic Standard Gaussian
               Z[k] = SN(RG);
            }
            y = x + ss*Z;      // Proposal: Isotropic gaussian with mean zero and covariance ss^2*Id
            
         // Do the metropolis detail balance check
            
            Ux = sqrNorm( M.q(x) );       // |q(x)|^2
            qy  = M.q(y);                 // evaluate q(y) (also used when processing accepted proposal)
            Uy = sqrNorm( qy );           // |q(y)|^2
            
            A = exp( 0.5*((Ux - Uy) / (eps*eps)) );   // Metropolis ratio (gaussian proposal simplifies in the ratio, probabilities too)
            
            if ( SU(RG) > A ){      // Accept with probability A,
               softFlag = Met_rej_soft;    // rejected
               stats-> SoftRejectionMetropolis++;
            }
            else{
               softFlag = Met_acc_soft;    // accepted
            }
            
            if ( softFlag == Met_acc_soft ) {     //  process an accepted proposal
               x   = y;
               gqx = M.gq(y);     // update gradiet
               qx  = qy;          // update constraint function
               softFlag = accept_y_soft;
               stats-> SoftSampleAccepted++;
            }
            else {                                       // process a rejected proposal
            }
            
            l++;   // we drew a Soft sample --> increment index for Schain[]
            for ( int k = 0; k < d; k++){    // add sample to Schain here
               Schain[ k + d*l]  = x[k];    // value of x[k] where x is l-th Soft / Acc.Off / Rej.On sample
            }
            
         } // end of Soft move
         
         else{   // draw an On move with probability p_on

            stats-> OnSample++;
            
//--------------------------------------- On move (we jump back onto the surface constraint)---------------------------------------------
         
            onFlag = starting_on;
            
         //    Find xs on constraint surface first. That is, use Newton find 'a' such that q( x +gq(x)a ) = 0
            
            xs = x;                  // initial guess = start from current state x ( that is, guess a=0 )
            gqxs  = M.gq(xs);        // because these are needed in loop
            for ( int ni = 0; ni < itm; ni++){
               r     = - M.q(xs);             // residual, want this to be = 0
               gtxsgx = trans( gqxs )*gqx;    // Newton Jacobian matrix
               solve( gtxsgx, da, r);         // solve Newton'm system for da: gq(xs)^t gq(x) da = - q(xs)
               xs = xs + gqx*da;              // take the Newton step
               gqxs = M.gq(xs);               // constraint gradient at the new point xs (for later)
               if ( norm(r) <= neps ){
                  onFlag = xs_proj_worked_on;   // record that you found xs
                  break;                       // stop the Newton iteration
               }     // end of if
            }     // end of Newton solver loop
            
            
            if ( onFlag == starting_on ) {        // the Newton iteration failed, or the flag would be: xs_proj_worked_off
               onFlag = xs_proj_failed_on;
               stats-> OnRejectionFailedProjection_xs++;
            }                                     // end of the xs projection
            
         //    If xs was found, do a surface sampler move, starting from xs. We produce a proposal y = xs + v + w ..
         //    .. where v in T_xs, w in orth(T_xs).
            
            if ( onFlag == xs_proj_worked_on ){
               
               //    First, find v by drawing gaussian R (scale = son) in R^n and then take v = Tx * R where Tx is an orthonormal ..
               //    .. basis matrix for T_xs ( found as last d-m columns of Qx where gqx = Qx Rx is the QR decomp. for gq(xs) )
               
               for (unsigned int k = 0; k < n; k++){  // Sample Isotropic Standard Gaussian
                  R[k] = SN(RG);
               }
               R = son*R;    // isotropic gaussian, scale son
               
               Agqxs = M.Agq(gqxs);             // augment with column of zeros to get full QR  for gqxs
               qr( Agqxs, Qx, Rx );             // QR  decomposition of gq(xs) (where Qx orthonorm, Rx upp triang)
               
               //    Calculate T_xs: last d-m columns of Qx
               for ( unsigned long int i = m; i < d; i++){
                  unsigned long int k = i-m;
                  column(Tx,k) = column(Qx,i);
               }      // end of calculation for T_xs
               
               v = Tx * R;
               vn   = sqrNorm(v);                 // | v |^2, used in the Metropolis check later ...
               
               //   Second, construct proposal y. Use Newton to find 'a' such that q( xs + v + gq(xs)a ) = 0. Then, set ..
               //   .. y = xs + v + gq(xs)a.
               
               y = xs + v;               // initial guess = move in the tangent direction ( that is, guess a=0 )
               gqy = M.gq(y);            // because these are calculated at the end of this loop
               for ( int ni = 0; ni < itm; ni++){
                  r     = - M.q(y);              // residual, want this to be = 0
                  gtygx = trans( gqy )*gqxs;     // Newton Jacobian matrix gq(y)^t gq(xs) (we store it as 'gtygx', to save memory)
                  solve( gtygx, da, r);          // solve Newton'm system for da: gq(y)^t gq(xs) da = - q(y)
                  y = y + gqxs*da;               // take the Newton step
                  gqy = M.gq(y);                 // constraint gradient at the new point (for later)
                  if ( norm(r) <= neps ){
                     onFlag = y_proj_worked_on;     // record that you found y
                     break;                     // stop the Newton iteration
                  }     // end of if
               }     // end of Newton solver loop
               if ( onFlag == starting_on ) {         // the Newton iteration failed, or the flag would be: y_proj_worked_on
                  onFlag = y_proj_failed_on;          // done with this surface sampler step
                  stats-> OnRejectionFailedProjection_y++;
               }                                      // end of the proposal generation phase
            }     // end of y proposal generation
            
         //    Do the Metropolis detail balance check, only if y proposal was found
            
            if ( onFlag == y_proj_worked_on ){
               
               //   First, calculate T_y: last n=d-m columns of U in Full SVD for gq(y) = U * S * Vtr.
               //   Then, also calculate N_y (basis for normal space at y): first m columns of Qy
               
               Agqy  = M.Agq(gqy);     // augment with column of zeros to get full SVD for gqy
               svd( Agqy, U, s, Vtr );            // full SVD decomposition of gq(y)
               
               //    Compute dety = det(S) = sqrt( det( gqy^t * gqy ) )
               dety = 1.;
               for ( unsigned long int i = 0; i < m; i++){
                  dety *= s[i];       // dety = det(S) = sqrt( det( gqy^t * gqy ) )
               }
               
               //    Calculate T_y: last d-m columns of U
               for ( unsigned long int i = m; i < d; i++){
                  unsigned long int k = i-m;
                  column(Ty,k) = column(U ,i);
               }      // end of calculation for T_y
               
               //    Need to find Sn, St for the reverse (off) move from y to x:  x-y = v' = Sn + St, where v' = x-y, Sn in Ny, St in Ty.
               //    For this, we have   x-y = Ty*a + [ x-y - Ty*a ],  where a = Ty^t * (x-y) so that St = Ty * [ Ty^t * (x-y) ]
               
               v  = x-y;                         // reverse Off move proposal in R^d
               St = Ty * ( trans(Ty) * v );      // reverse Off move tangential component
               Sn = v - St;                      // reverse Off move normal component
               Vtn = trans(St) * St;                      // | Vt |^2 , where St = Ty * Vt
               Vnn = trans(Sn) * gqy * trans(gqy) * Sn;   // | Vn |^2 , where Sn = By * Vn
               
               p_ratio  = p_off / p_on;                   // ratio of probabilities
               detTytTx = det( trans(Ty)*Tx );            // det( T_y^t T_xs ) = det( T_xs^t T_y )
               Ux       = sqrNorm(M.q(x));                // |q(x)|^2
               fy       = cf / dety;              // surface density at y
               
               nconst   = ( pow( 1.0/(2.0*M_PI) , ((double) m) / 2.0 ) *
                         pow( 1.0/sn , (double) m ) * pow( son / st, (double) n ) ) * dety; // nconst is the ratio of normalizing ..
                                                                                            // .. constants for densities of proposals
               
               A  = exp( 0.5*( (Ux/(eps*eps)) + (vn/(son*son)) - (Vnn/(sn*sn)) - (Vtn/(st*st)) ) )
                                        * nconst * p_ratio * (1.0/abs(detTytTx)) * fy;   //  Metropolis ratio (where we use the..
                                                                                        //   .. uniform measure on the hard surface)
               
               if ( debug_on == 1 ) {
                  cout << "--------------------------------------" << endl;
                  cout << " ON MOVE : " << endl;
                  cout << " exp     = " << exp((0.5*Ux/(eps*eps)) + (0.5*vn/(son*son)) - (0.5*Vnn/(sn*sn)) - (0.5*Vtn/(st*st))) << endl;
                  cout << " exp   Ux/ 2 eps^2      = " << exp( (0.5*Ux/(eps*eps)) ) << endl;
                  cout << " exp  |v|^2 / son^2     = " << exp( (0.5*vn/(son*son)) ) << endl;
                  cout << " exp -|Vn|^2 / sn^2     = " << exp( -(0.5*Vnn/(sn*sn)) ) << endl;
                  cout << " exp -|Vt|^2 / st^2     = " << exp( -(0.5*Vtn/(st*st)) ) << endl;//
                  cout << " nconst * fy            = " << nconst * fy               << endl;
                  cout << " nconst * p_ratio * fy  = " << nconst * p_ratio * fy     << endl;
                  cout << " p_ratio         = " << p_ratio << endl;
                  cout << " 1/det(Ty^t Txs) = " << (1.0/abs(detTytTx)) << endl;
                  cout << " detSy           = " << dety << endl;
                  cout << " "                           << endl;
                  cout << " Acceptance Pr   = " << A    << endl;
                  cout << " "                           << endl;
                  cout << " y[0]            = " << y[0] << endl;
                  cout << " "                           << endl;
                  cout << "--------------------------------------" << endl;
               }
               
               if ( SU(RG) > A ){      // Accept with probability A,
                  onFlag = Met_rej_on;    // rejected
                  stats-> OnRejectionMetropolis++;
               }
               else{
                  onFlag = Met_acc_on;    // accepted
               }
            }  // end of Metropolis detail balance check
            
         //    There is NO reverse check for the On move.

            if ( onFlag == Met_acc_on ) {     //  process an accepted proposal
               x   = y;
               gqx = gqy;      // update gradiet
               qx  = M.q(y);   // update constraint function
               onFlag = accept_y_on;
               stats-> OnSampleAccepted++;
            }
            else {    // process a rejected proposal
               
               l++;   // we rejected an On sample --> increment index for Schain[]
               for ( int k = 0; k < d; k++){    // add sample to Schain here
                  Schain[ k + d*l]  = x[k];    // value of x[k] where x is l-th Soft / Acc.Off / Rej.On sample
               }
               
            }

         }  // end of On move
      }  // end of Soft / On moves section
      
      
   } // end of for loop
   
}  // end of SASampler















