/*
       Squishy sampler project, the Surface Augmented Sampler approach.
       See main.cpp for more information.
       
       Some sections of the code are adapted from
       Jonathan Goodman's code main.cpp

*/
//
//  3D Warped Torus model
//
//  main.cpp
//
#include <iostream>
#include <fstream>                // write the output to a file
#include <blaze/Math.h>
#include <random>                 // contains the random number generator
#include <cmath>                  // defines exp(..), M_PI
#include <chrono>
#include "SAS.hpp"
#include "model.hpp"
using namespace std;
using namespace blaze;



int main(int argc, char** argv){
   
   const int d = 3;       // dimension of ambient space
   const int m = 2;       // number of constraint functions

   
   //  Set up the specific model problem -- a surface defined by intersection of m spheres
   
   DynamicVector<double, columnVector> ck(d);         // Center of sphere k
   DynamicMatrix<double, columnMajor>  c(d,m);        // Column k is the center of sphere k
   DynamicVector<double, columnVector> s(d);          // dimensional radius numbers for the ellipsoid

   double r;                                          // r = radius of sphere
   
   ck[0] = 0.;                   // first center: c_0 = ( 0, 0, 1)
   ck[1] = 0.;
   ck[2] = 1.;
   column( c, 0UL) = ck;         // 0UL means 0, as an unsigned long int
   
   ck[0] = 0.;                   // second center: c_0 = ( 0,-1, 0, )
   ck[1] = -1.;
   ck[2] =  0.;
   column( c, 1UL) = ck;
   
   r = sqrt(2.);                // The distance between centers is sqrt(2)
   s[0] = sqrt(2.);
   s[1] = sqrt(3.);
   s[2] = sqrt(5.);
   
   // copy the parameters into the instance model
   
   
   Model M( d, m, r, s, c);   // instance of model " 3D Warped Torus "


   
   cout << "--------------------------------------" << endl;
   cout << "\n  Running model: " << M.ModelName() << "\n" << endl;
   
   
//        Inputs for the SoftSampler
   
   
   DynamicVector<double, columnVector> x( d); // starting point for sampler
   
   //x[0] = 0.;                   // give value to starting point, off Hard constraint
   //x[1] = 0.;
   //x[2] = 0.;
   
   //x[0] = 1.20105824111462;                   // give value to starting point, on 3D Simple Torus model constraint ..
   //x[1] = -0.669497937230318;                 // // .. r1 = sqrt(2), r2 = sqrt(2)
   //x[2] = 0.669497937230317;
   
   x[0] = 0.3780902941558512;                 // give value to starting point, on 3D Warped Torus model constraint ..
   x[1] = -0.7586573478136572;                // .. r = sqrt(2), sx = sqrt(2), sy = sqrt(3), sz = sqrt(5)
   x[2] = 2.132027719657734;



   size_t T      = 20000000;     // number of MCMC steps
   double neps   = 1.e-10;       // convergence tolerance for Newton projection
   double rrc    = 1.e-8;        // closeness criterion for the reverse check
   int itm       = 6;            // maximum number of Newtons iterations
   
   int debug_off = 0;                // if = 1, then prints data on Metropolis ratio for Off move
   int debug_on  = 0;                // if = 1, then prints data on Metropolis ratio for On  move
   
   int integrate = 1;                // if = 1, then it does the integration to find marginal density for x[0]
   
   
// --------------------------------------------KEY PARAMETERS FOR SAMPLER---------------------------------------------------------
   
   
   double beta   = 1000.0;                      // squish parameter, beta = 1 / 2*eps^2
   double eps    = 1.0 / sqrt(2.0*beta);        // squish parameter
   
   double p_hard   = 0.8;          // probability of drawing Hard move
   double p_soft   = 0.2;          // probability of drawing Soft move
   double p_off    = 1.0 - p_hard;
   double p_on     = 1.0 - p_soft;
   double p_ratio  = p_on / p_off;
   
   double kt  = 1.0;               // factor for st  (st = scale of Vt in Off proposal)   * should be similar to kon !! *
   double kon = 1.0;               // factor for son (son = scale of On move tang. step)  * should be similar to kt !! *
   double kn  = 1.0;               // factor for sn  (sn = scale of Vn in Off proposal)
   
   double kc  = 1.0;               // proportionality constant in density w.r.t. surface measure
   
   double ks  = 0.7;                // factor for Soft proposal size


   double st   = kt*eps;           // scale for Off proposal, Vt part    * should similar to son !! *
   double son  = kon*eps;          // scale for On  proposal             * should similar to st !!  *
   double sn   = kn*eps;           // scale for Off proposal, Vn part
   
   double sc   = sn;
   double cf = p_ratio * sqrt( pow(2.0*M_PI, (double) m) *  // constant for density wrt surface measure ..
                              pow(sc, (double) 2.0*m) );    // .. * f1 = normalizing constant for density of Vn *
                                                         

   double sh     = 1.0;          // scale for Hard proposal
   double ss     =ks*eps;        // scale for Soft proposal
   
// -------------------------------------------------------------------------------------------------------------------------------
   
   
   size_t T_Schain = 0.5 * T;         // length of Schain: we will have about 0.25*T samples from Soft / Acc. Off / Rej. On moves
   vector<double> Schain(d*T_Schain); // Schain[k+d*l] is the value of x[k] where x is l-th Soft / Acc. Off sample / Rej. On moves
   unsigned seed = 17U;     // seed for the random number generator -- 17 is the most random number?
   mt19937 RG(seed);        // Mersenne twister random number generator
   SamplerStats stats;
   
   
   auto start = chrono::steady_clock::now();
   SASampler(Schain, &stats, T, eps, p_hard, p_soft, x, M, cf,
                  sh, sn, st, son, ss, neps, rrc, itm, debug_off, debug_on, RG);
   auto end = chrono::steady_clock::now();
   
   double Th   = stats.HardSample;    // number of Hard samples proposed
   double Toff = stats.OffSample;     // number of off  samples proposed
   double Ts   = stats.SoftSample;    // number of Soft samples proposed
   double Ton  = stats.OnSample;      // number of On   samples proposed
   
   double Ah   = 0.;      // Pr of Acceptance of Hard sample
   double Aoff = 0.;      // Pr of Acceptance of Off sample
   double As   = 0.;      // Pr of Acceptance of Soft sample
   double Aon  = 0.;      // Pr of Acceptance of On sample
   
   if (stats.HardSample > 0) {   // Set Pr of Acceptance of Hard sample
      Ah    = (double) stats.HardSampleAccepted / (double) stats.HardSample;
   }
   
   if (stats.OffSample > 0) {   // Set Pr of Acceptance of Off sample
      Aoff    = (double) stats.OffSampleAccepted / (double) stats.OffSample;
   }
   
   if (stats.SoftSample > 0) {  // Set Pr of Acceptance of Soft sample
      As    = (double) stats.SoftSampleAccepted / (double) stats.SoftSample;
   }
   
   if (stats.OnSample > 0) {   // Set Pr of Acceptance of On sample
      Aon  = (double) stats.OnSampleAccepted / (double) stats.OnSample;
   }
   
   
   int Ns   = stats.SoftSample + stats.OffSampleAccepted          // number of Soft / Accepted Off / Rejected On samples ..
                               + (Ton - stats.OnSampleAccepted);  // .. produced by SoftSampler(). These are stored in  ..
                                                                  // .. Schain[] and will be used for analysis.
   
   //int Ns   = stats.HardSample;

   
   //int Ns   = stats.SoftSample;
   

   cout << " beta = " << beta << endl;
   cout << " eps = " << eps << endl;
   cout << " Elapsed time : " << chrono::duration_cast<chrono::seconds>(end - start).count() << " sec" << endl;
   cout << " "              << endl;
   cout << " Hard move reverse check failures : " << stats.HardRejectionReverseCheck << endl;
   cout << " Off move reverse check failures  : " << stats.OffRejectionReverseCheck  << endl;
   cout << " "              << endl;
   cout << " Hard move y projection failures : " << stats.HardRejectionFailedProjection_y << endl;
   cout << " Off move ys projection failures : " << stats.OffRejectionProjection_ys<< endl;
   cout << " " << endl;
   cout << " Number of Hard samples proposed = " << Th    << endl;
   cout << " Number of Off  samples proposed = " << Toff  << endl;
   cout << " Number of Soft samples proposed = " << Ts    << endl;
   cout << " Number of On   samples proposed = " << Ton   << endl;
   cout << " " << endl;
   cout << " Hard sample Acceptance Pr = " << Ah   << endl;
   cout << " Off  sample Acceptance Pr = " << Aoff << endl;
   cout << " Soft sample Acceptance Pr = " << As   << endl;
   cout << " On   sample Acceptance Pr = " << Aon  << endl;
   cout << " " << endl;
   cout << " T  = " << T  << endl;
   cout << " " << endl;
   cout << " Ns = " << Ns << endl;
   cout << " " << endl;
   cout << " Ns / T  = " << ((double) Ns) / ((double) T)   << endl;
   cout << " " << endl;
   cout << "--------------------------------------" << endl;

   
//  setup for data analysis:
   
   int bin;
   char OutputString[200];
   int  StringLength;
   
//   NO ANGLE CHECK
   
   
//   Histogram of the x coordinates x[0]=x, x[1]=y, x[2]=z
   
   int nx   = 100;             // number of x1 values for the PDF and number of x1 bins
   vector<double> Ratio(nx);   // contains the random variables Ri_N

   if ( integrate == 1 ) {
      int ni   = 500;     // number of integration points in each direction
      double L = -3.0;
      double R =  3.0;
      double x1    = .5;
      vector<double> fl(nx);  // approximate (un-normalized) true pdf for x[0]=x, compute by integrating out x,z variables
      vector<int>    Nxb(nx); // vector counting number of samples in each bin
      for ( bin = 0; bin < nx; bin++) Nxb[bin] = 0;
      double dx = ( R - L )/( (double) nx);
      for ( int i=0; i<nx; i++){
         x1 = L + dx*i + .5*dx;
         fl[i]= M.yzIntegrate( x1, L, R, eps, ni);
      }
               
      int outliers = 0;       //  number of samples outside the histogram range
      for ( unsigned int iter = 0; iter < Ns; iter++){
         x1 = Schain[ d*iter ];  // same as before, but with k=0 as we need x[0] of iter-th Soft/Off sample
         bin = (int) ( ( x1 - L )/ dx);
         if ( ( bin >= 0 ) && ( bin < nx ) ){
            Nxb[bin]++;
         }
         else{
            outliers++;
         }
      }        // end of analysis loop
     
      double Z;
      cout << " " << endl;
      cout << "   bin    center      count      pdf          1/Z" << endl;
      for ( bin = 0; bin < nx; bin++){
         if ( Nxb[bin] > 0 ) {
            x1 = L + dx*bin + .5*dx;
            Z = ( (double) Nxb[bin]) / ( (double) Ns*dx*fl[bin]);
            Ratio[bin] = Z;
            StringLength = sprintf( OutputString, " %4d   %8.3f   %8d   %9.3e    %9.3e", bin, x1, Nxb[bin], fl[bin], Z);
            cout << OutputString << endl;
         }
      }
      cout << " " << endl;
      cout << " Number of outliers : " << outliers << endl;
      cout << " " << endl;
   }
   

   ofstream OutputFile ( "ChainOutput.py");
   OutputFile << "# data output file from an MCMC code\n" << endl;
   OutputFile << "import numpy as np" << endl;
   StringLength = sprintf( OutputString, "eps = %10.5e", eps);
   OutputFile << OutputString << endl;
   StringLength = sprintf( OutputString, "kt = %10.5e", kt);
   OutputFile << OutputString << endl;
   StringLength = sprintf( OutputString, "kon = %10.5e", kon);
   OutputFile << OutputString << endl;
   StringLength = sprintf( OutputString, "kn = %10.5e", kn);
   OutputFile << OutputString << endl;
   StringLength = sprintf( OutputString, "ks = %10.5e", ks);
   OutputFile << OutputString << endl;
   StringLength = sprintf( OutputString, "kc = %10.5e", kc);
   OutputFile << OutputString << endl;
   StringLength = sprintf( OutputString, "ss = %10.5e", ss);
   OutputFile << OutputString << endl;
   StringLength = sprintf( OutputString, "sh = %10.5e", sh);
   OutputFile << OutputString << endl;
   StringLength = sprintf( OutputString, "sn = %10.5e", sn);
   OutputFile << OutputString << endl;
   StringLength = sprintf( OutputString, "st = %10.5e", st);
   OutputFile << OutputString << endl;
   StringLength = sprintf( OutputString, "son = %10.5e", son);
   OutputFile << OutputString << endl;
   StringLength = sprintf( OutputString, "p_h = %10.5e", p_hard);
   OutputFile << OutputString << endl;
   StringLength = sprintf( OutputString, "p_s = %10.5e", p_soft);
   OutputFile << OutputString << endl;
   StringLength = sprintf( OutputString, "T = %10d", (int)T);
   OutputFile << OutputString << endl;
   StringLength = sprintf( OutputString, "Th = %10d", (int)Th);
   OutputFile << OutputString << endl;
   StringLength = sprintf( OutputString, "Toff = %10d", (int)Toff);
   OutputFile << OutputString << endl;
   StringLength = sprintf( OutputString, "Ts = %10d", (int)Ts);
   OutputFile << OutputString << endl;
   StringLength = sprintf( OutputString, "Ton = %10d", (int)Ton);
   OutputFile << OutputString << endl;
   StringLength = sprintf( OutputString, "Ah = %6.3f", Ah);
   OutputFile << OutputString << endl;
   StringLength = sprintf( OutputString, "Aoff = %6.3f", Aoff);
   OutputFile << OutputString << endl;
   StringLength = sprintf( OutputString, "As = %6.3f", As);
   OutputFile << OutputString << endl;
   StringLength = sprintf( OutputString, "Aon = %6.3f", Aon);
   OutputFile << OutputString << endl;
   StringLength = sprintf( OutputString, "Ns = %10d", Ns);
   OutputFile << OutputString << endl;
   StringLength = sprintf( OutputString, "d = %10d", d);
   OutputFile << OutputString << endl;
   OutputFile << "ModelName = \"" << M.ModelName() << "\"" << endl;
   OutputFile.close();
   
/*
   OutputFile << "\n" << endl;
   OutputFile << "# X contains Soft Samples + Accepted Off samples, used for analysis\n" << endl;
   StringLength = sprintf( OutputString, "X = np.ndarray([%5d,%5d], dtype=np.float64)", (int)Ns, d);
   OutputFile << OutputString << endl;
   for ( int iter = 0; iter < Ns; iter++){
      for (int k = 0; k < d; k++){
         StringLength = sprintf( OutputString, "X[%5d,%5d] = %10.5e", iter, k, Schain[ k + d*iter]);
         OutputFile << OutputString << endl;
       }
    }

   OutputFile.close();
*/


// Write Schain in binary format (in a file called "Schain.bin"), much faster for Python to read when Schain is very long
   size_t size1 = Ns*d;
   ofstream ostream1("Schain.bin", ios::out | ios::binary);
   ostream1.write((char*)&Schain[0], size1 * sizeof(double));
   ostream1.close();


   
}  // end of main
