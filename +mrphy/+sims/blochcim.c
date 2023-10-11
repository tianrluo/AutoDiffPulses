/*
 * blochcim.c
 */

#include <mex.h>
#if defined(_WIN32)
  #ifndef M_PI
    #define M_PI 3.14159265358979323846
  #endif
#else
  #include <math.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h> /* common macros: size_t, ptrdiff_t, etc. */

struct rPara
{
  double niBmag;   /* negative inverse magnitude of this Beff */
  double ct;       /* cos of theta caused by Beff magnitude   */
  double stbn[3];  /* (sin of the theta) times niBmag         */
  double ctbn1[3]; /* (1 - ct) times niBmag                   */
};

struct Bxyz
{
  const double *x; /* Beff's x component */
  const double *y; /* Beff's y component */
  const double *z; /* Beff's z component */
};

inline void getRotatePara(
  const double        Bx,
  const double        By,
  const double        Bz,
  const double        gambar, /* 2pi*Hz/G */
  const double        dt,     /* Sec */
        struct rPara *rP_p)
{
  double Bmag  = sqrt(Bx*Bx + By*By + Bz*Bz);
  double theta = Bmag * gambar * dt;
  rP_p->niBmag   = Bmag? -1/Bmag : 0; /* '-' for correct cross-prod direction */
  rP_p->ct       = cos(theta);

  double stn  = sin(theta)   * rP_p->niBmag;
  double ct1n = (1 - rP_p->ct) * rP_p->niBmag;

  rP_p->stbn[0] =stn *Bx; rP_p->stbn[1] =stn *By; rP_p->stbn[2] =stn *Bz;
  rP_p->ctbn1[0]=ct1n*Bx; rP_p->ctbn1[1]=ct1n*By; rP_p->ctbn1[2]=ct1n*Bz;
}

inline void doRotate(
  const size_t       nSpins,       /* number of spins                     */
  const double       dt,           /* (Sec) simumation temporal step size */
  const int          isNotScalarB, /* is B not the same for all spin      */
  const struct Bxyz *B_p,          /* (Gauss) ptr to Beff                 */
  const int          isNotScalarG, /* is gambar the same for all spin     */
  const double      *gambar_p,     /* (Hz/Gauss * 2PI) gyro-freq ratio    */
  const double      *M0_p,         /* spin state before rotation          */
        double      *M1_p)         /* spin state after rotation           */
{
  const double *Mx0_p = M0_p, *My0_p = M0_p+nSpins, *Mz0_p = M0_p+2*nSpins;
  double       *Mx1_p = M1_p, *My1_p = M1_p+nSpins, *Mz1_p = M1_p+2*nSpins;

  struct rPara rP;

  const double *Bx_p = B_p->x, *By_p = B_p->y, *Bz_p = B_p->z;
  double M0x = 0, M0y = 0, M0z = 0;
  double ip = 0; /* inner product */
  size_t iSpins = 0;

  for (iSpins = 0; iSpins < nSpins; ++iSpins)
  {
    if ( *Bx_p || *By_p || *Bz_p )
    {
      getRotatePara(*Bx_p, *By_p, *Bz_p, *gambar_p, dt, &rP);

      M0x = *Mx0_p++; M0y = *My0_p++; M0z = *Mz0_p++;
      ip = (M0x * *Bx_p + M0y * *By_p + M0z * *Bz_p)*rP.niBmag;
      *Mx1_p++ = rP.ct     *M0x -rP.stbn[2]*M0y +rP.stbn[1]*M0z +ip*rP.ctbn1[0];
      *My1_p++ = rP.stbn[2]*M0x +rP.ct     *M0y -rP.stbn[0]*M0z +ip*rP.ctbn1[1];
      *Mz1_p++ =-rP.stbn[1]*M0x +rP.stbn[0]*M0y +rP.ct     *M0z +ip*rP.ctbn1[2];
    }
    else
    {
      *Mx1_p++ = *Mx0_p++; *My1_p++ = *My0_p++; *Mz1_p++ = *Mz0_p++;
    }
    if (isNotScalarB) { Bx_p++; By_p++; Bz_p++; }
    gambar_p += isNotScalarG;
  }
}

inline void doDecay(
  const size_t  nSpins,
  const double  dt,
  const int     isNotScalarE1,
  const double *E1_p,
  const int     isNotScalarE2,
  const double *E2_p,
        double *M_p)
{
  double *Mx_p = M_p, *My_p = M_p+nSpins, *Mz_p = M_p+2*nSpins;
  size_t iSpins = 0;

  for (iSpins = 0; iSpins < nSpins; ++iSpins)
  {
    *Mx_p++ *= *E2_p;
    *My_p++ *= *E2_p;
    *Mz_p = *Mz_p * *E1_p + (1-*E1_p);
    Mz_p++;

    E1_p += isNotScalarE1;
    E2_p += isNotScalarE2;
  }
}

/* blochcim, the outer for-loop of nSteps is intentionaly left in mexFunction */
/*
 * Mo = blochcim(Mi, Beff, T1, T2, dt, gam)
 * INPUTS:
 * - Mi (nSpins, nDim): input spins' magnetizations
 * - Beff , Gauss:
 *     (1, nDim, nSteps), global
 *     (nSpins, nDim, nSteps), spin-wise
 * - T1 & T2, Sec: globally or spin-wisely defined T1 and T2
 *     (1,), global
 *     (nSpins, 1), spin-wise
 * - dt (1,) Sec: temporal simulation step size.
 * - gam, Hz/G, gyro frequency ratio:
 *     (1,), global
 *     (nSpins, 1), spin-wise
 * OUTPUTS:
 * - Mo   (nSpins, nDim, nSteps): output spins' magnetizations
 * - Mhst (nSpins, nDim, nSteps): simulated spins' trajectories history
 */

void mexFunction(
  int nlhs,       mxArray *plhs[],
  int nrhs, const mxArray *prhs[])
{
  const mwSize *nd_M = mxGetDimensions(prhs[0]);
  const mwSize *nd_B = mxGetDimensions(prhs[1]);

  const size_t  nSpM = (size_t)(nd_M[0]); /* nSpins */
  const size_t  nDim = (size_t)(nd_M[1]);

  const size_t  nSpB = (size_t)nd_B[0];
  const size_t  nSt  = (size_t)nd_B[2];   /* nSteps */

  const mwSize  nd_Mh[3] = {nSpM, nDim, nSt}; /* size of Mhst */

  /* a bit sanity check */
  if (!mxIsDouble(prhs[0])) {mexErrMsgTxt("blochcim: Mi should be double");}
  if (!mxIsDouble(prhs[1])) {mexErrMsgTxt("blochcim: Beff should be double");}
  if (!mxIsDouble(prhs[2])) {mexErrMsgTxt("blochcim: T1 should be double");}
  if (!mxIsDouble(prhs[3])) {mexErrMsgTxt("blochcim: T2 should be double");}
  if (!mxIsDouble(prhs[4])) {mexErrMsgTxt("blochcim: dt should be double");}
  if (!mxIsDouble(prhs[5])) {mexErrMsgTxt("blochcim: gam should be double");}

  /* I/O */
  const int     doHist = (nlhs > 1); /* Record history */
  const double *Mi_p   = mxGetPr(prhs[0]);
  const double *Beff_p = mxGetPr(prhs[1]);
  const double *T1_p   = mxGetPr(prhs[2]);
  const double *T2_p   = mxGetPr(prhs[3]);
  const double  dt     = mxGetScalar(prhs[4]);
  const double *gam_p  = mxGetPr(prhs[5]);

  struct Bxyz B = {.x = Beff_p, .y = Beff_p+nSpB, .z = Beff_p+2*nSpB};

  plhs[0] = mxCreateNumericArray(2, nd_M, mxDOUBLE_CLASS, mxREAL);
  if (doHist) {plhs[1] = mxCreateNumericArray(3, nd_Mh, mxDOUBLE_CLASS,mxREAL);}
  double *Mo_p = mxGetPr(plhs[0]), *Mhst_p = doHist?mxGetPr(plhs[1]):NULL;
  double *M1_p = doHist?Mhst_p:Mo_p;

  /*
  mexPrintf("nSpM: %d; nDim: %d; nSt: %d; nSpB: %d\n", nSpM, nDim, nSt, nSpB);
  mexPrintf("doHist: %d;\n", doHist);

  return;
  */

  const double *M0_p = Mi_p;

  /* pre-process */
  const mwSize nT1  = mxGetNumberOfElements(prhs[2]);
  const mwSize nT2  = mxGetNumberOfElements(prhs[3]);
  const mwSize nGam = mxGetNumberOfElements(prhs[5]);
  const int    isNotScalarB = !(1==nSpB);
  const int    isNotScalarG = !(1==nGam);
  const int    isNotScalarE1 = !(1==nT1);
  const int    isNotScalarE2 = !(1==nT2);

  double *E1_p     = mxCalloc(nT1,  sizeof(double));
  double *E2_p     = mxCalloc(nT2,  sizeof(double));
  double *gambar_p = mxCalloc(nGam, sizeof(double));
  double TAU = 2 * M_PI;

  size_t iT1 = 0, iT2 = 0, iGam = 0;
  for (iT1  = 0; iT1  < nT1;  ++iT1)  { *E1_p++ = exp(-dt / *T1_p++); }
  for (iT2  = 0; iT2  < nT2;  ++iT2)  { *E2_p++ = exp(-dt / *T2_p++); }
  for (iGam = 0; iGam < nGam; ++iGam) { *gambar_p++ = TAU * *gam_p++; }
  E1_p -= nT1;
  E2_p -= nT2;
  gambar_p -= nGam;

  /* simulation */
  size_t iSt = 0, nSpM_nDim = 0, nSpB_nDim = nSpB*nDim;

  if (doHist) /* M1_p == Mhst_p */
  {
    nSpM_nDim = nSpM*nDim;
    for (iSt = 0; iSt < nSt; ++iSt)
    {
      doRotate(nSpM, dt, isNotScalarB, &B, isNotScalarG, gambar_p, M0_p, M1_p);
      doDecay( nSpM, dt, isNotScalarE1, E1_p, isNotScalarE2, E2_p, M1_p);

      B.x += nSpB_nDim;
      B.y += nSpB_nDim;
      B.z += nSpB_nDim;
      M0_p = M1_p;
      M1_p += nSpM_nDim; /* Will be out-of-range at return */
    }
    memcpy((void *)Mo_p, (void *)M0_p, nSpM_nDim*sizeof(double));
  }
  else        /* M1_p == Mo_p */
  {
    for (iSt = 0; iSt < nSt; ++iSt)
    {
      doRotate(nSpM, dt, isNotScalarB, &B, isNotScalarG, gambar_p, M0_p, M1_p);
      doDecay( nSpM, dt, isNotScalarE1, E1_p, isNotScalarE2, E2_p, M1_p);

      B.x += nSpB_nDim;
      B.y += nSpB_nDim;
      B.z += nSpB_nDim;
      M0_p = M1_p;
    }
  }
}

