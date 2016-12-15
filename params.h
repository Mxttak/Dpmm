/*
Class that contains the parameters of the DPGMM.

Copyright (C) 2016 Maxim Dolgov: m<dot>dolgov<at>web<dot>de
No warranty, no commercial use.

Definitely contains errors; use with caution.
*/

#ifndef __PARAMS_H__
#define __PARAMS_H__

#include "mex.h"
#include "miscmatlab.h"
#include "Eigen/Eigen"

#include <vector>
#include <stdexcept>

// a class for managing parameters for DPMM codecvt
// sums and samples must be column vectors when used

/* ---------------------------------------------------------------------- */
template <typename MAT>
void printmat(MAT& mat) {
  mexPrintf("[ ");
  for (int i = 0; i < mat.rows(); i++) {
    for (int j = 0; j < mat.cols(); j++) {
      mexPrintf("%f ", mat(i, j));
    }
  }
  mexPrintf("]");
}

/* ---------------------------------------------------------------------- */
class ParamsStruct {
 public:
  double alpha;
  double kappa;
  double nu;
  Eigen::MatrixXd initmean;
  Eigen::MatrixXd initcov;
  double num_classes;  // don't forget to convert to size_t when using
  double num_emptyclasses;
  std::vector<double> counts;  // don't forget to convert to size_t when using
  std::vector<Eigen::MatrixXd> sums;
  std::vector<Eigen::MatrixXd> cholSSE;
  std::vector<double> classes;  // don't forget to convert to size_t when using

  size_t dimSamples;
  size_t numSamples;

 public:
  ParamsStruct(){};
  ParamsStruct(const mxArray* input, size_t structNum);
  void Init(const mxArray* input, size_t structNum);
  void PrintParams(void);
  void LogParams(void);
  void DumpToOutput(mxArray* output, size_t structNum);
  void RemoveEmptyClasses(void);
  void AddNewClass(void);
  double CountEmptyClasses(void);
};  // END: class ParamsStruct

/* ---------------------------------------------------------------------- */
ParamsStruct::ParamsStruct(const mxArray* input, size_t structNum) {
  Init(input, structNum);
}

/* ---------------------------------------------------------------------- */
void ParamsStruct::Init(const mxArray* input, size_t structNum) {
  // get the scalar parameters
  alpha = *((double*)mxGetData(
      mxGetFieldByNumber(input, structNum, mxGetFieldNumber(input, "alpha"))));
  if (alpha <= 0) {
    throw std::out_of_range("Input parameter <alpha> must be positive.");
  }
  kappa = *((double*)mxGetData(
      mxGetFieldByNumber(input, structNum, mxGetFieldNumber(input, "kappa"))));
  if (kappa <= 0) {
    throw std::out_of_range("Input parameter <kappa> must be positive.");
  }
  nu = *((double*)mxGetData(
      mxGetFieldByNumber(input, structNum, mxGetFieldNumber(input, "nu"))));
  if (nu <= 0) {
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
  initcov = Eigen::Map<Eigen::MatrixXd>((double*)mxGetData(pInitcov),
                                        dimsInitcov[0], dimsInitcov[1]);

  // set dimSamples
  dimSamples = dimsInitcov[0];

  // get initmean
  mxArray* pInitmean =
      mxGetFieldByNumber(input, structNum, mxGetFieldNumber(input, "initmean"));
  size_t numDimsInitmean = mxGetNumberOfDimensions(pInitmean);
  const size_t* dimsInitmean = mxGetDimensions(pInitmean);
  if (numDimsInitmean > 2 || (dimsInitmean[0] != 1 && dimsInitmean[1] != 1)) {
    throw std::out_of_range("Input parameter <initmean> must be a vector.");
  }
  initmean =
      Eigen::Map<Eigen::MatrixXd>((double*)mxGetData(pInitmean), dimSamples, 1);

  // get num_classes
  num_classes = (*(double*)mxGetData(mxGetFieldByNumber(
      input, structNum, mxGetFieldNumber(input, "num_classes"))));
  if (num_classes < 0) {
    throw std::out_of_range(
        "Input parameter <num_classes> cannot be negative.");
  }

  // get counts
  mxArray* pMxCounts =
      mxGetFieldByNumber(input, structNum, mxGetFieldNumber(input, "counts"));
  double* pCounts = (double*)mxGetData(pMxCounts);
  counts =
      std::vector<double>((double*)mxGetData(pMxCounts),
                          (double*)mxGetData(pMxCounts) + (size_t)num_classes);

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
  sums.reserve(num_classes);
  cholSSE.reserve(num_classes);
  counts.reserve(num_classes);
  for (size_t iterClasses = 0; iterClasses < num_classes; iterClasses++) {
    // sums.push_back(Eigen::Map<Eigen::VectorXd>(pSums + iterClasses *
    // dimSamples,  dimSamples));
    sums.push_back(Eigen::Map<Eigen::MatrixXd>(pSums + iterClasses * dimSamples,
                                               dimSamples, 1));
    cholSSE.push_back(Eigen::Map<Eigen::MatrixXd>(
        pCholSSE + iterClasses * dimSamples * dimSamples, dimSamples,
        dimSamples));
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
  classes = std::vector<double>(
      (double*)mxGetData(pClasses),
      (double*)mxGetData(pClasses) + dimsClasses[0] * dimsClasses[1]);

  // set numSamples;
  numSamples = dimsClasses[0] * dimsClasses[1];

  num_emptyclasses = CountEmptyClasses();
}  // END: void ParamsStruct::Init(const mxArray* input, size_t structNum)

/* ---------------------------------------------------------------------- */
void ParamsStruct::PrintParams(void) {
  // debugging function that prints the params to matlab

  mexPrintf("\nalpha = %f\nkappa = %f\nnu = %f\nnum_classes = %d\n", alpha,
            kappa, nu, (size_t)num_classes);
  mexPrintf("intimean = [ ");
  printmat(initmean);
  mexPrintf("]\ninitcov = [ ");
  printmat(initcov);
  mexPrintf("]\ncounts = [ ");
  for (size_t iterClasses = 0; iterClasses < (size_t)num_classes;
       iterClasses++) {
    mexPrintf("%f ", counts.at(iterClasses));
  }
  mexPrintf("]\nsums: \n");
  for (size_t iterClasses = 0; iterClasses < (size_t)num_classes;
       iterClasses++) {
    mexPrintf("class %d,\n\tsums = [ ", iterClasses);
    printmat(sums.at(iterClasses));
    mexPrintf("]\n\tcholSSE = [ ");
    printmat(cholSSE.at(iterClasses));
    mexPrintf("]\n");
  }
  mexPrintf("num_emptyclasses = %d\n\tClasses = [ ", (size_t)num_emptyclasses);
  for (size_t iterSamples = 0; iterSamples < numSamples; iterSamples++) {
    mexPrintf("%.1f ", classes.at(iterSamples));
  }
  mexPrintf("]\n");
}  // END: ParamsStruct::PrintParams(void)

/* ---------------------------------------------------------------------- */
void ParamsStruct::DumpToOutput(mxArray* output, size_t structNum) {
  // stores the parameters in the output structure determined by <structNum>

  // mxCreateNumericArray(mwSize ndim, const mwSize *dims, mxClassID classid,
  // mxComplexity ComplexFlag)

  // void mxSetField(mxArray *pm, mwIndex index, const char *fieldname, mxArray
  // *pvalue);

  // alpha
  size_t tmpDims[] = {1, 1, 0};
  mxArray* pAlpha = mxCreateNumericArray(2, tmpDims, mxDOUBLE_CLASS, mxREAL);
  memcpy(mxGetData(pAlpha), &alpha, sizeof(double));
  mxSetField(output, structNum, "alpha", pAlpha);

  // kappa
  mxArray* pKappa = mxCreateNumericArray(2, tmpDims, mxDOUBLE_CLASS, mxREAL);
  memcpy(mxGetData(pKappa), &kappa, sizeof(double));
  mxSetField(output, structNum, "kappa", pKappa);

  // nu
  mxArray* pNu = mxCreateNumericArray(2, tmpDims, mxDOUBLE_CLASS, mxREAL);
  memcpy(mxGetData(pNu), &nu, sizeof(double));
  mxSetField(output, structNum, "nu", pNu);

  // initmean
  tmpDims[0] = dimSamples;
  mxArray* pInitmean = mxCreateNumericArray(2, tmpDims, mxDOUBLE_CLASS, mxREAL);
  Eigen::Map<Eigen::MatrixXd>((double*)mxGetData(pInitmean), dimSamples, 1) =
      initmean;
  mxSetField(output, structNum, "initmean", pInitmean);

  // initcov
  tmpDims[1] = dimSamples;
  mxArray* pInitcov = mxCreateNumericArray(2, tmpDims, mxDOUBLE_CLASS, mxREAL);
  Eigen::Map<Eigen::MatrixXd>((double*)mxGetData(pInitcov), dimSamples,
                              dimSamples) = initcov;
  mxSetField(output, structNum, "initcov", pInitcov);

  // num_classes
  tmpDims[0] = 1;
  tmpDims[1] = 1;
  mxArray* pNumClasses =
      mxCreateNumericArray(2, tmpDims, mxDOUBLE_CLASS, mxREAL);
  memcpy(mxGetData(pNumClasses), &num_classes, sizeof(double));
  mxSetField(output, structNum, "num_classes", pNumClasses);

  // counts
  tmpDims[1] = (size_t)num_classes;
  mxArray* pCounts = mxCreateNumericArray(2, tmpDims, mxDOUBLE_CLASS, mxREAL);
  memcpy(mxGetData(pCounts), counts.data(),
         ((size_t)num_classes) * sizeof(double));
  mxSetField(output, structNum, "counts", pCounts);

  tmpDims[1] = (size_t)num_classes;
  tmpDims[0] = dimSamples;
  mxArray* pSums = mxCreateNumericArray(2, tmpDims, mxDOUBLE_CLASS, mxREAL);

  tmpDims[1] = dimSamples;
  tmpDims[2] = (size_t)num_classes;
  mxArray* pCholSSE = mxCreateNumericArray(3, tmpDims, mxDOUBLE_CLASS, mxREAL);

  for (size_t iterClasses = 0; iterClasses < (size_t)num_classes;
       iterClasses++) {
    Eigen::Map<Eigen::MatrixXd>(
        (double*)mxGetData(pSums) + iterClasses * dimSamples, dimSamples, 1) =
        sums.at(iterClasses);
    Eigen::Map<Eigen::MatrixXd>(
        (double*)mxGetData(pCholSSE) + iterClasses * dimSamples * dimSamples,
        dimSamples, dimSamples) = cholSSE.at(iterClasses);
  }
  mxSetField(output, structNum, "sums", pSums);
  mxSetField(output, structNum, "cholSSE", pCholSSE);

  // classes
  tmpDims[0] = numSamples;
  tmpDims[1] = 1;
  mxArray* pClasses = mxCreateNumericArray(2, tmpDims, mxDOUBLE_CLASS, mxREAL);
  memcpy(mxGetData(pClasses), classes.data(), numSamples * sizeof(double));
  mxSetField(output, structNum, "classes", pClasses);
}  // END: void ParamsStruct::DumpToOutput(mxArray* output, size_t
// structNum)

/* ---------------------------------------------------------------------- */
double ParamsStruct::CountEmptyClasses(void) {
  double num = 0;
  for (size_t iter = 0; iter < (size_t)num_classes; iter++) {
    if ((size_t)counts.at(iter) == 0) num += 1;
  }
  return num;
}

/* ---------------------------------------------------------------------- */
void ParamsStruct::RemoveEmptyClasses(void) {
  // removes empty classes from the parameters

  /* updates:
  double num_classes;          // don't forget to convert to size_t when using
  std::vector<double> counts;  // don't forget to convert to size_t when using
  std::vector<Eigen::VectorXd> sums;
  std::vector<Eigen::MatrixXd> cholSSE;
  */
  size_t maxiter = (size_t)num_classes;
  for (size_t iter = 0; iter < maxiter; iter++) {
    if ((size_t)counts.at(iter) < 1) {
      // empty cluster

      counts.erase(counts.begin() + iter);
      sums.erase(sums.begin() + iter);
      cholSSE.erase(cholSSE.begin() + iter);
      // decrease cluster numbers for samples
      for (size_t i = 0; i < numSamples; i++) {
        if ((size_t)classes.at(i) > iter) classes.at(i) -= 1;
      }
      num_classes -= 1;
      maxiter -= 1;
      iter -= 1;
    }
  }

  num_emptyclasses = CountEmptyClasses();
}  // END: void ParamsStruct::RemoveEmptyClasses(void)

/* ---------------------------------------------------------------------- */
void ParamsStruct::AddNewClass(void) {
  // attaches a new empty class to the params
  /* updates
  double num_classes;          // don't forget to convert to size_t when using
  std::vector<double> counts;  // don't forget to convert to size_t when using
  std::vector<Eigen::VectorXd> sums;
  std::vector<Eigen::MatrixXd> cholSSE;
  */
  num_classes += 1;
  counts.push_back(0.0);
  sums.push_back(kappa * initmean);
  Eigen::LLT<Eigen::MatrixXd> lltOfA(initcov);
  cholSSE.push_back(nu * ((Eigen::MatrixXd)lltOfA.matrixU()));

  num_emptyclasses = CountEmptyClasses();
}  // END: ParamsStruct::AddNewClass(void)

#endif