/*
       Squishy sampler project, the Surface Augmented Sampler approach.
       See main.cpp for more information.
       
       Some sections of the code are adapted from
       Jonathan Goodman's code FoliationSampler.hpp

*/
//
//  SAS.hpp
//

#ifndef SAS_hpp
#define SAS_hpp

#include <iostream>
#include <blaze/Math.h>
#include <random>
#include <string>
#include "model.hpp"
using namespace std;
using namespace blaze;




struct SamplerStats{                     /* Acceptance/rejection data from the sampler */
   int HardSample;
   int HardSampleAccepted;
   int HardRejectionFailedProjection_y; // Hard move: failed projection from x+v in T_x to y
   int HardRejectionMetropolis;
   int HardRejectionReverseCheck;
   int OffSample;
   int OffSampleAccepted;
   int OffRejectionProjection_ys;  // Off move: failed projection from proposal y to ys, used for reverse proposal
   int OffRejectionMetropolis;
   int OffRejectionReverseCheck;
   int SoftSample;
   int SoftSampleAccepted;
   int SoftRejectionMetropolis;
   int OnSample;
   int OnSampleAccepted;
   int OnRejectionFailedProjection_xs;  // On move: failed projection from current x to xs
   int OnRejectionFailedProjection_y;   // On move: failed projection from xs+v in T_xs to y on Surface
   int OnRejectionMetropolis;
};
 

void SASampler(    vector<double> &Schain,         /* Soft Samples output from the MCMC run, pre-allocated, length = d*T */
                   struct SamplerStats *stats,     /* statistics about different kinds of rejections                     */
                   size_t T,                       /* number of MCMC steps        */
                   double eps,                     /* squish parameter            */
                   double p_hard,                  /* probability of Hard move    */
                   double p_soft,                  /* probability of Soft move    */
                   DynamicVector<double>& x0,      /* starting point              */
                   Model M,                        /* evaluate q(x) and grad q(x) */
                   double cf,                       /* constant multiplying density of surface measure    */
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
                   mt19937 RG);                    /* random generator engine, already instantiated      */





#endif /* SAS.hpp */

