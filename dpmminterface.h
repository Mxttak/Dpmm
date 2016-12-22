/*
A wrapper class for DPMM class. Contains extraction and storage functions for
interacting with Matlab.

Copyright (C) 2016 Maxim Dolgov: m<dot>dolgov<at>web<dot>de
No warranty, no commercial use.

Definitely contains errors; use with caution.
*/

#ifndef _DPMMINTERFACE_H_
#define _DPMMINTERFACE_H_

#include "mex.h"
#include "dpmm.h"

/* ---------------------------------------------------------------------- */
template <typename Mat>
void printmat(const Mat mat) {
  mexPrintf("[ ");
  for (int i = 0; i < mat.rows(); i++) {
    for (int j = 0; j < mat.cols(); j++) {
      mexPrintf("%f ", mat(i, j));
    }
  }
  mexPrintf("]");
}

/* ---------------------------------------------------------------------- */
class DPMMInterface {
 public:
  DPMM dp;

 public:
  DPMMInterface(){};
  template <typename Mat>
  DPMMInterface(const mxArray* input, size_t structNum, Mat& samples) {
    ReadParams(input, structNum);
    dp.samples = samples;
  }
  template <typename Mat>
  DPMMInterface(const Mat& samples) {
    dp.InitDefault(samples);
  };
  void ReadParams(const mxArray* input, size_t structNum);
  void StoreParams(mxArray* output, size_t structNum);
  void PrintParams(void);
};  // END: class DPMMInterface

/* ---------------------------------------------------------------------- */
void DPMMInterface::ReadParams(const mxArray* input, size_t structNum) {
  // get the scalar parameters
  dp.alpha = *((double*)mxGetData(
      mxGetFieldByNumber(input, structNum, mxGetFieldNumber(input, "alpha"))));
  if (dp.alpha <= 0) {
    throw std::out_of_range("Input parameter <alpha> must be positive.");
  }
  dp.kappa = *((double*)mxGetData(
      mxGetFieldByNumber(input, structNum, mxGetFieldNumber(input, "kappa"))));
  if (dp.kappa <= 0) {
    throw std::out_of_range("Input parameter <kappa> must be positive.");
  }
  dp.nu = *((double*)mxGetData(
      mxGetFieldByNumber(input, structNum, mxGetFieldNumber(input, "nu"))));
  if (dp.nu <= 0) {
    throw std::out_of_range("Input parameter <nu> must be positive.");
  }

  // get initcov
  mxArray* pInitcov =
      mxGetFieldByNumber(input, structNum, mxGetFieldNumber(input, "initcov"));
  size_t numDimsInitcov = mxGetNumberOfDimensions(pInitcov);
  const size_t* dimsInitcov = mxGetDimensions(pInitcov);
  if (numDimsInitcov > 2 || (dimsInitcov[0] != dimsInitcov[1])) {
    throw std::out_of_range(
        "Input parameter <initcov> must be a square matrix.");
  }
  dp.initcov = Eigen::Map<Eigen::MatrixXd>((double*)mxGetData(pInitcov),
                                           dimsInitcov[0], dimsInitcov[1]);

  // set dimSamples
  dp.dimSamples = dimsInitcov[0];

  // get initmean
  mxArray* pInitmean =
      mxGetFieldByNumber(input, structNum, mxGetFieldNumber(input, "initmean"));
  size_t numDimsInitmean = mxGetNumberOfDimensions(pInitmean);
  const size_t* dimsInitmean = mxGetDimensions(pInitmean);
  if (numDimsInitmean > 2 || (dimsInitmean[0] != 1 && dimsInitmean[1] != 1)) {
    throw std::out_of_range("Input parameter <initmean> must be a vector.");
  }
  dp.initmean =
      Eigen::Map<Eigen::VectorXd>((double*)mxGetData(pInitmean), dp.dimSamples);

  dp.num_clusters = (*(double*)mxGetData(mxGetFieldByNumber(
      input, structNum, mxGetFieldNumber(input, "num_classes"))));
  if (dp.num_clusters < 0) {
    throw std::out_of_range(
        "Input parameter <num_clusters> cannot be negative.");
  }

  // get counts
  mxArray* pMxCounts =
      mxGetFieldByNumber(input, structNum, mxGetFieldNumber(input, "counts"));
  double* pCounts = (double*)mxGetData(pMxCounts);
  dp.counts = std::vector<double>(
      (double*)mxGetData(pMxCounts),
      (double*)mxGetData(pMxCounts) + (size_t)dp.num_clusters);

  // get pointer to sums
  mxArray* sumsNumber =
      mxGetFieldByNumber(input, structNum, mxGetFieldNumber(input, "sums"));
  size_t numDimsSums = mxGetNumberOfDimensions(sumsNumber);
  const size_t* dimsSums = mxGetDimensions(sumsNumber);
  double* pSums = (double*)mxGetData(sumsNumber);

  // get pointer to cholSSE
  mxArray* cholSseNumber =
      mxGetFieldByNumber(input, structNum, mxGetFieldNumber(input, "cholSSE"));
  size_t numDimsCholSSE = mxGetNumberOfDimensions(cholSseNumber);
  const size_t* dimsCholSSE = mxGetDimensions(cholSseNumber);
  double* pCholSSE = (double*)mxGetData(cholSseNumber);

  // get cholSSE and sums
  dp.sums.reserve(dp.num_clusters);
  dp.cholSSE.reserve(dp.num_clusters);
  dp.counts.reserve(dp.num_clusters);
  for (size_t iterClasses = 0; iterClasses < dp.num_clusters; iterClasses++) {
    // sums.push_back(Eigen::Map<Eigen::VectorXd>(pSums + iterClasses *
    // dimSamples,  dimSamples));
    dp.sums.push_back(Eigen::Map<Eigen::VectorXd>(
        pSums + iterClasses * dp.dimSamples, dp.dimSamples));
    dp.cholSSE.push_back(Eigen::Map<Eigen::MatrixXd>(
        pCholSSE + iterClasses * dp.dimSamples * dp.dimSamples, dp.dimSamples,
        dp.dimSamples));
    // counts.push_back((double)pCounts[iterClasses]);
  }

  // get cluster assignments
  mxArray* pClasses =
      mxGetFieldByNumber(input, structNum, mxGetFieldNumber(input, "classes"));
  size_t numDimsClasses = mxGetNumberOfDimensions(pClasses);
  const size_t* dimsClasses = mxGetDimensions(pClasses);
  if (numDimsClasses > 2 || (dimsClasses[0] != 1 && dimsClasses[1] != 1)) {
    throw std::out_of_range("Input parameter <classes> must be a vector.");
  }
  dp.clusters = std::vector<double>(
      (double*)mxGetData(pClasses),
      (double*)mxGetData(pClasses) + dimsClasses[0] * dimsClasses[1]);

  // set numSamples;
  dp.numSamples = dimsClasses[0] * dimsClasses[1];
}  // END: void ReadParams(DPMM& dp)

/* ---------------------------------------------------------------------- */
void DPMMInterface::StoreParams(mxArray* output, size_t structNum) {
  // alpha
  size_t tmpDims[] = {1, 1, 0};
  mxArray* pAlpha = mxCreateNumericArray(2, tmpDims, mxDOUBLE_CLASS, mxREAL);
  memcpy(mxGetData(pAlpha), &dp.alpha, sizeof(double));
  mxSetField(output, structNum, "alpha", pAlpha);

  // kappa
  mxArray* pKappa = mxCreateNumericArray(2, tmpDims, mxDOUBLE_CLASS, mxREAL);
  memcpy(mxGetData(pKappa), &dp.kappa, sizeof(double));
  mxSetField(output, structNum, "kappa", pKappa);

  // nu
  mxArray* pNu = mxCreateNumericArray(2, tmpDims, mxDOUBLE_CLASS, mxREAL);
  memcpy(mxGetData(pNu), &dp.nu, sizeof(double));
  mxSetField(output, structNum, "nu", pNu);

  // initmean
  tmpDims[0] = dp.dimSamples;
  mxArray* pInitmean = mxCreateNumericArray(2, tmpDims, mxDOUBLE_CLASS, mxREAL);
  Eigen::Map<Eigen::VectorXd>((double*)mxGetData(pInitmean), dp.dimSamples) =
      dp.initmean;
  mxSetField(output, structNum, "initmean", pInitmean);

  // initcov
  tmpDims[1] = dp.dimSamples;
  mxArray* pInitcov = mxCreateNumericArray(2, tmpDims, mxDOUBLE_CLASS, mxREAL);
  Eigen::Map<Eigen::MatrixXd>((double*)mxGetData(pInitcov), dp.dimSamples,
                              dp.dimSamples) = dp.initcov;
  mxSetField(output, structNum, "initcov", pInitcov);

  // num_classes
  tmpDims[0] = 1;
  tmpDims[1] = 1;
  mxArray* pNumClasses = mxCreateDoubleMatrix(1, 1, mxREAL);
  mxCreateNumericArray(2, tmpDims, mxDOUBLE_CLASS, mxREAL);
  memcpy(mxGetData(pNumClasses), &dp.num_clusters, sizeof(double));
  mxSetField(output, structNum, "num_classes", pNumClasses);

  // counts
  tmpDims[1] = (size_t)dp.num_clusters;
  mxArray* pCounts = mxCreateNumericArray(2, tmpDims, mxDOUBLE_CLASS, mxREAL);
  memcpy(mxGetData(pCounts), dp.counts.data(),
         ((size_t)dp.num_clusters) * sizeof(double));
  mxSetField(output, structNum, "counts", pCounts);

  tmpDims[1] = (size_t)dp.num_clusters;
  tmpDims[0] = dp.dimSamples;
  mxArray* pSums = mxCreateNumericArray(2, tmpDims, mxDOUBLE_CLASS, mxREAL);

  tmpDims[1] = dp.dimSamples;
  tmpDims[2] = (size_t)dp.num_clusters;
  mxArray* pCholSSE = mxCreateNumericArray(3, tmpDims, mxDOUBLE_CLASS, mxREAL);

  for (size_t iterClasses = 0; iterClasses < (size_t)dp.num_clusters;
       iterClasses++) {
    Eigen::Map<Eigen::VectorXd>(
        (double*)mxGetData(pSums) + iterClasses * dp.dimSamples,
        dp.dimSamples) = dp.sums.at(iterClasses);
    Eigen::Map<Eigen::MatrixXd>((double*)mxGetData(pCholSSE) +
                                    iterClasses * dp.dimSamples * dp.dimSamples,
                                dp.dimSamples, dp.dimSamples) =
        dp.cholSSE.at(iterClasses);
  }
  mxSetField(output, structNum, "sums", pSums);
  mxSetField(output, structNum, "cholSSE", pCholSSE);

  // classes
  tmpDims[0] = dp.numSamples;
  tmpDims[1] = 1;
  mxArray* pClasses = mxCreateNumericArray(2, tmpDims, mxDOUBLE_CLASS, mxREAL);
  memcpy(mxGetData(pClasses), dp.clusters.data(),
         dp.numSamples * sizeof(double));
  mxSetField(output, structNum, "classes", pClasses);

}  // END: void StoreParams(DPMM& dp, mxArray* output, size_t structNum)

/* ---------------------------------------------------------------------- */
void DPMMInterface::PrintParams(void) {
  mexPrintf("\nalpha = %f\nkappa = %f\nnu = %f\nnum_clusters = %d", dp.alpha,
            dp.kappa, dp.nu, (size_t)dp.num_clusters);
  mexPrintf("\ninitmean = ");
  printmat(dp.initmean);
  mexPrintf("\ninitcov = ");
  printmat(dp.initcov);
  for (size_t iter = 0; iter < (size_t)dp.sums.size(); iter++) {
    mexPrintf("\ncounts.at(%d) = %d", iter, (size_t)dp.counts.at(iter));
    mexPrintf("\nsums.at(%d) = ", iter);
    printmat(dp.sums.at(iter));
    mexPrintf("\ncholSSE.at(%d) = ", iter);
    printmat(dp.cholSSE.at(iter));
  }

  mexPrintf("\nclusters = [ ");
  for (size_t iter = 0; iter < dp.numSamples; iter++)
    mexPrintf("%d ", (size_t)dp.clusters.at(iter));
  mexPrintf("]\n");
}  // END: void PrintParams(DPMM& dp)

#endif  // _DPMMINTERFACE_H_
