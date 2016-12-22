/*
Mex function for a Dirichlet Process Gaussian Mixture Model (DPGMM) parameter
estimation according to implementation by Jacob Eisenstein in pure Matlab.

Copyright (C) 2016 Maxim Dolgov: m<dot>dolgov<at>web<dot>de
No warranty, no commercial use.

Definitely contains errors; use with caution.
*/

#include "mex.h"
#include "dpmminterface.h"

#include <stdexcept>

/* ---------------------------------------------------------------------- */
void mxCopyStruct(mxArray* destination, size_t destStruct,
                  const mxArray* source, size_t sourceStruct) {
  /*
  copies a numeric structure at <destStruct> from <destination> into a
  preallocated structure at <sourceStruct> in <source>

  Example:
  mxCopyStruct(plhs[0],0,prhs[0],2);
  copies the data of the third structure from input data at 0 to the first
  structure in outputs at 0
  */

  // get the number of fields in each structure
  size_t numFields = mxGetNumberOfFields(source);

  // get IDs of stored data (double, int, etc)
  mxClassID* classIDflags = (mxClassID*)mxCalloc(numFields, sizeof(mxClassID));

  // iterate through all fields
  for (size_t iterFields = 0; iterFields < numFields; iterFields++) {
    // get dimensions of the current field
    size_t numDims = mxGetNumberOfDimensions(
        mxGetFieldByNumber(source, sourceStruct, iterFields));
    const size_t* dims =
        mxGetDimensions(mxGetFieldByNumber(source, sourceStruct, iterFields));

    // create current element in the output structure
    mxArray* pOut = mxCreateNumericArray(
        numDims, dims,
        mxGetClassID(mxGetFieldByNumber(source, (int)sourceStruct, iterFields)),
        mxREAL);
    void* pData = mxGetData(pOut);

    // determine how many bytes to copy
    size_t copySize = 1;
    for (size_t j = 0; j < numDims; j++) {
      copySize *= dims[j];
    }
    copySize *=
        mxGetElementSize(mxGetFieldByNumber(source, sourceStruct, iterFields));

    // copy data
    memcpy(pData,
           mxGetData(mxGetFieldByNumber(source, sourceStruct, iterFields)),
           copySize);
    // set field in output
    mxSetFieldByNumber(destination, destStruct, iterFields, pOut);
  }  // END: iterate through all fields
}

/* ---------------------------------------------------------------------- */
void mexFunction(int nOnputs, mxArray* outputs[],         // output arguments
                 int nInputs, const mxArray* inputs[]) {  // input arguments

  const mxArray* pSamples = inputs[1];
  const mxArray* pParams = inputs[0];
  const mxArray* pNumIter = inputs[2];

  size_t numIter = (size_t)(*((double*)mxGetData(pNumIter)));

  double* samples = (double*)mxGetData(pSamples);

  size_t numSamples = mxGetN(pSamples);
  size_t dimSamples = mxGetM(pSamples);

  // if (nInputs > 2) {

  size_t numFields =
      mxGetNumberOfFields(pParams);  // number of fields in each structure
  size_t numStructs =
      mxGetNumberOfElements(pParams);  // number of structures in the list

  // extract field names and class ids
  const char** fieldNames;
  fieldNames = (const char**)mxCalloc(numFields, sizeof(*fieldNames));
  mxClassID* classIDflags = (mxClassID*)mxCalloc(numFields, sizeof(mxClassID));

  for (size_t iterField = 0; iterField < numFields; iterField++) {
    fieldNames[iterField] = mxGetFieldNameByNumber(pParams, iterField);
    classIDflags[iterField] =
        mxGetClassID(mxGetFieldByNumber(pParams, 0, iterField));
  }

  // create output
  outputs[0] =
      mxCreateStructMatrix(1, numStructs + numIter, numFields, fieldNames);

  // copy the provided parameters into output
  for (size_t iterStructs = 0; iterStructs < numStructs; iterStructs++) {
    mxCopyStruct(outputs[0], iterStructs, inputs[0], iterStructs);
  }

  DPMMInterface dpInterface(
      pParams, numStructs - 1,
      Eigen::Map<Eigen::MatrixXd>(
          samples, dimSamples, numSamples));  // initialize with default params

  for (size_t iter = 0; iter < numIter; iter++) {
    dpInterface.dp.gibbsSampling(1);

    dpInterface.StoreParams(outputs[0], numStructs + iter);
  }
}
