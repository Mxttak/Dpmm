/*
Mex function to perform a provided number of Gibbs sampling iterations for
parameter estimation of a DPGMM.

I tried to design it such that the Dpmm class (can perform a single Gibbs
iteration) can be easily used by other functions. However, it is necessary to
separate the creation of the Dpmm class and initialization of the parms (will be
implemented sometimes).

Adaptive Rejection Sampling of the parameter alpha uses the digamma function by
Richard J. Mathar that is available at
http://www2.mpia-hd.mpg.de/~mathar/progs/digamma.c (for more info, there is a
header in myars.h)

Copyright (C) 2016 Maxim Dolgov: m<dot>dolgov<at>web<dot>de
No warranty, no commercial use.

Definitely contains errors; use with caution.
*/

#include "mex.h"
#include "dpmm.h"

/*
params = gibbsSampling(params,samples,numIter)
params includes the old parameters and the new
*/

/* ---------------------------------------------------------------------- */
void mexFunction(int nOnputs, mxArray* outputs[],         // output arguments
                 int nInputs, const mxArray* inputs[]) {  // input arguments

  // just
  const mxArray* pParams = inputs[0];
  const mxArray* pSamples = inputs[1];
  const mxArray* pNumIter = inputs[2];

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

  // determine the number of iterations
  size_t numIter = (size_t)(*((double*)mxGetData(pNumIter)));

  // create output
  outputs[0] =
      mxCreateStructMatrix(1, numStructs + numIter, numFields, fieldNames);

  // copy the provided parameters into output
  for (size_t iterStructs = 0; iterStructs < numStructs; iterStructs++) {
    mxCopyStruct(outputs[0], iterStructs, inputs[0], iterStructs);
  }

  // extract params and create DP class
  Dpmm dp(pParams, numStructs - 1, pSamples);  // I will separate this step
                                               // sometimes to get independent
                                               // of the Matlab part

  // perform Gibbs Iterations
  for (size_t gibbsIter = 0; gibbsIter < numIter; gibbsIter++) {
    // perform single run
    dp.SingleGibbsIteration();

    // save parameters after the run
    dp.params.DumpToOutput(outputs[0], numStructs + gibbsIter);
  }
}
