/*
Class with functions for parameter estimation of a DPGMM using Gibbs sampling.

Copyright (C) 2016 Maxim Dolgov: m<dot>dolgov<at>web<dot>de
No warranty, no commercial use.

Definitely contains errors; use with caution.
*/

#ifndef __DPMM_H__
#define __DPMM_H__

#include "mex.h"
#include "params.h"
#include "myars.h"
#include "Multinomial_double.hpp"
#include "Eigen/Eigen"

#include <vector>
#include <random>
#include <ctime>
#include <chrono>
#include <stdexcept>
#include <numeric>  // std::iota

// samples and sums in params must be column vectors

// todo: remove iterations over samples in hideObservation and unhideObservation
// -> write an extra function for computaiton of initial parameters

// todo: remove dependencies on Matlab for debugging

// todo: include the sampler for alpha into class properties (the randomness
// will be improved because the random number generator won't be reinitialized
// at each sampling)

// todo: merge cholupdate and icholupdate into a single function using Eigen's
// rankUpdate function for adjointselfView of the matrix.

/* ---------------------------------------------------------------------- */
class Dpmm {
 public:
  ParamsStruct params;
  Eigen::MatrixXd samples;
  AlphaSampler alphaSampler;

  std::vector<size_t> indexes;

  std::mt19937 gen;  // mersenne twister random number generator
  std::uniform_real_distribution<double> dis;  // distribution

 public:
  Dpmm(){};
  Dpmm(const mxArray* mxParams, size_t structnum, const mxArray* mxSamples) {
    Init(mxParams, structnum, mxSamples);
  };
  void Dpmm::Init(const mxArray* mxParams, size_t structnum,
                  const mxArray* mxSamples);

  template <typename Mat>
  double SampleClass(const Mat& classprobs);

  void Dpmm::UpdateAlpha(void);

  void Dpmm::SingleGibbsIteration(void);

  template <typename Vec, typename Mat>
  void Dpmm::ComputeClassProbs(const Vec& data, Mat& classprobs);

  template <typename veca, typename vecb, typename mat>
  double Dpmm::normpdfln(const veca& x, const vecb& m, const mat& cov);

  template <typename Mat>
  void Dpmm::unhideObservation(size_t classnum, const Mat& data);

  template <typename Mat>
  void Dpmm::hideObservation(size_t classnum, const Mat& data);

  template <typename VEC, typename MAT>
  void Dpmm::cholupdate(const VEC& x, MAT& cov);

  template <typename VEC, typename MAT>
  void Dpmm::icholupdate(const VEC& x, MAT& cov);

};  // END: class Dpmm

/* ---------------------------------------------------------------------- */
void Dpmm::Init(const mxArray* mxParams, size_t structnum,
                const mxArray* mxSamples) {
  clock_t t = clock();
  // map samples
  samples = Eigen::Map<Eigen::MatrixXd>((double*)mxGetData(mxSamples),
                                        mxGetM(mxSamples), mxGetN(mxSamples));

  // map parameters
  params.Init(mxParams, structnum);

  // initialize random number generator and the distribution
  gen.seed(clock() - t + time(0));
  dis.param(std::uniform_real_distribution<double>::param_type(0.0, 1.0));

  // create vector of sample indexes
  indexes = std::vector<size_t>(params.numSamples);
  std::iota(indexes.begin(), indexes.end(), 0);
  std::random_shuffle(indexes.begin(), indexes.end());
}  // END: void Dpmm::Init(mxArray* mxParams, mxArray* mxSamples)

/* ---------------------------------------------------------------------- */
void Dpmm::UpdateAlpha(void) {
  alphaSampler = AlphaSampler(params.num_classes, params.numSamples);
  params.alpha = alphaSampler.drawSample();
}  // END: void Dpmm::UpdateAlpha(void)

/* ---------------------------------------------------------------------- */
void Dpmm::SingleGibbsIteration(void) {
  // A single iteration of Gibbs sampling.

  // random permutation of sample indexes
  std::random_shuffle(indexes.begin(), indexes.end());

  size_t pos;
  double newClass, curClass;

  // iterate through samples
  for (size_t iter = 0; iter < params.numSamples; iter++) {
    // which sample to process
    pos = indexes.at(iter);

    // which cluster is the current sample assigned to
    curClass = params.classes.at(pos);

    // get the sample out of its cluster and recompute the relevant cluster
    // parameters (mean, covariance, counts)
    hideObservation((size_t)(curClass - 1), samples.col(pos));

    // remove empty classes if any
    params.RemoveEmptyClasses();

    // create a new empty class
    params.AddNewClass();

    // compute the probabilities of the current sample to belong to a particular
    // cluster
    Eigen::MatrixXd classprobs((size_t)params.num_classes, 1);
    ComputeClassProbs(samples.col(pos), classprobs);

    // sample new cluster according to these probabilities
    newClass = SampleClass(classprobs);

    params.classes.at(pos) = newClass;

    // put the sample into the cluster and update the relevant cluster
    // parameters (mean, covariance, counts)
    unhideObservation((size_t)(newClass - 1.0), samples.col(pos));

  }  // END: iterate through samples

  // update alpha (see Rasmussen, "The Infinite Gaussian Mixture Model", 2000,
  // Eqn. 15)
  UpdateAlpha();
}  // END: void Dpmm::SingleGibbsIteration(void)

/* ---------------------------------------------------------------------- */
template <typename Mat>
double Dpmm::SampleClass(const Mat& classprobs) {
  // Samples from a multinomial by sampling from the Uniform(0.0,1.0) and then
  // checking into which bin of the multinomial it falls.

  // draw a sample from Uniform(0,1)
  double randnum = dis(gen);
  // gen.seed((int)(randnum * 1000) + clock());
  randnum = dis(gen);

  // compute cumulative sum
  std::vector<double> cumsum((size_t)classprobs.rows());
  cumsum.at(0) = classprobs(0);

  if (randnum <= cumsum.at(0)) return 1.0;

  for (size_t iter = 0; iter < (size_t)classprobs.rows() - 1; iter++) {
    cumsum.at(iter + 1) = cumsum.at(iter) + classprobs(iter + 1);

    if (randnum <= cumsum.at(iter + 1)) return (iter + 2.0);
  }

  // return negative value that indicates an error
  return -1.0;
}  // END: double Dpmm::SampleClass(const Mat& classprobs)

/* ---------------------------------------------------------------------- */
template <typename Vec, typename Mat>
void Dpmm::ComputeClassProbs(const Vec& data, Mat& classprobs) {
  // Computes the loglikelihood of the current sample for belonging to each
  // cluster. Normalized likelihood values correspond to the probability that
  // the current sample belongs to the particular cluster.

  Eigen::MatrixXd log_p_obs((size_t)params.num_classes, 1);
  Eigen::MatrixXd p_prior((size_t)params.num_classes, 1);
  double kappabar;

  // iterate over all clusters
  for (size_t iter = 0; iter < (size_t)params.num_classes; iter++) {
    kappabar = params.kappa + params.counts.at(iter);
    log_p_obs(iter) =
        normpdfln(data, params.sums.at(iter) / (kappabar),
                  params.cholSSE.at(iter) *
                      sqrt((kappabar + 1) /
                           (kappabar * (params.nu + params.counts.at(iter) -
                                        data.rows() - 1))));
    if ((size_t)params.counts.at(iter) == 0) {
      p_prior(iter) = params.alpha;
    } else {
      p_prior(iter) = (double)params.counts.at(iter);
    }
  }

  // remove offset to increase numeric stability
  log_p_obs = (Eigen::MatrixXd)(log_p_obs.array() - log_p_obs.maxCoeff()).exp();

  classprobs = p_prior.array() * log_p_obs.array();

  // normalize
  classprobs /= classprobs.sum();

}  // END: void Dpmm::ComputeClassProbs(const Vec& data)

/* ---------------------------------------------------------------------- */
template <typename veca, typename vecb, typename mat>
double Dpmm::normpdfln(const veca& x, const vecb& m, const mat& cov) {
  // Computes the loglikelihood that the current sample belongs to the Gaussian
  // with provided mean and covariance.

  const double log2pi = 1.83787706640935;
  Eigen::VectorXd tmp = cov.transpose().inverse() * (x - m);
  double out = -log(cov.diagonal().prod()) -
               0.5 * (x.rows() * log2pi + (tmp.transpose() * tmp));
  if (out != out) {  // detects whether the output is NaN
    throw std::out_of_range("Non-real value in normpdfln.");
  }
  return out;
}  // END: double Dpmm::normpdfln(const veca& x, const vecb& m, const mat& cov)

/* ---------------------------------------------------------------------- */
template <typename Mat>
void Dpmm::hideObservation(size_t classnum, const Mat& data) {
  // Removes the current sample from the cluster and updates the relevant
  // parameters: mean, covariance, counts.

  //  old_count = params.counts(new_class) + params.kappa;
  // old_sum = params.sums(new_class,:);
  // params.cholSSE(:,:,new_class) =
  // cholupdate(params.cholSSE(:,:,new_class),old_sum' / sqrt(old_count));
  cholupdate(params.sums.at(classnum) /
                 sqrt(params.kappa + params.counts.at(classnum)),
             params.cholSSE.at(classnum));

  for (size_t iter = 0; iter < (size_t)data.cols(); iter++) {
    // params.counts(new_class) = params.counts(new_class) - 1;
    params.counts.at(classnum) -= 1.0;
    if (params.counts.at(classnum) < 0)
      mexPrintf("\nunhideObservations: params.counts.at(%d) < 0, iter = %d",
                classnum, iter);

    // params.sums(new_class,:) = params.sums(new_class,:) - data(i,:);
    params.sums.at(classnum) -= data.col(iter);

    // params.cholSSE(:,:,new_class) =
    // cholupdate(params.cholSSE(:,:,new_class),data(i,:)','-');
    icholupdate(data.col(iter), params.cholSSE.at(classnum));
  }

  // new_count = params.counts(new_class) + params.kappa;
  // params.cholSSE(:,:,new_class) =
  // cholupdate(params.cholSSE(:,:,new_class),params.sums(new_class,:)' /
  // sqrt(new_count),'-');
  icholupdate(params.sums.at(classnum) /
                  sqrt(params.kappa + params.counts.at(classnum)),
              params.cholSSE.at(classnum));

}  // END: void Dpmm::unhideObservation(size_t classnum, const Mat& data)

/* ---------------------------------------------------------------------- */
template <typename Mat>
void Dpmm::unhideObservation(size_t classnum, const Mat& data) {
  // Puts the current sample into the cluster and updates the relevant
  // parameters: mean, covariance, counts.

  //  old_count = params.kappa + params.counts(new_class);
  // params.cholSSE(:,:,new_class) =
  // cholupdate(params.cholSSE(:,:,new_class),params.sums(new_class,:)' /
  // sqrt(old_count));

  cholupdate(params.sums.at(classnum) /
                 sqrt(params.kappa + params.counts.at(classnum)),
             params.cholSSE.at(classnum));

  // params.sums(new_class,:) = params.sums(new_class,:) + sum(data,1);
  params.sums.at(classnum) += data.rowwise().sum();

  // params.counts(new_class) = params.counts(new_class) + size(data,1);
  params.counts.at(classnum) += data.cols();

  for (size_t iter = 0; iter < (size_t)data.cols(); iter++) {
    // params.cholSSE(:,:,new_class) =
    // cholupdate(params.cholSSE(:,:,new_class),data(i,:)');
    cholupdate(data.col(iter), params.cholSSE.at(classnum));
  }

  // new_count = params.kappa + params.counts(new_class);
  // params.cholSSE(:,:,new_class) =
  // cholupdate(params.cholSSE(:,:,new_class),params.sums(new_class,:)' /
  // sqrt(new_count),'-');
  icholupdate(params.sums.at(classnum) /
                  sqrt(params.kappa + params.counts.at(classnum)),
              params.cholSSE.at(classnum));

}  // END: void Dpmm::unhideObservation(size_t classnum, Mat& data)

/* ---------------------------------------------------------------------- */
template <typename VEC, typename MAT>
void Dpmm::cholupdate(const VEC& x, MAT& cov) {
  // Performs rank 1 update of the covariance.

  Eigen::MatrixXd tmpx = x * x.transpose();
  Eigen::MatrixXd tmpCov = cov.transpose() * cov;

  if (tmpCov.rows() != cov.rows() || tmpCov.cols() != cov.rows()) {
    throw std::length_error("Dimensions of the covariance are wrong.");
  }
  if (tmpx.rows() != cov.rows() || tmpx.cols() != cov.rows()) {
    throw std::length_error("Dyadic product x*x\' has wrong dimensions.");
  }

  Eigen::LLT<Eigen::MatrixXd> lltOfA(tmpx + tmpCov);
  cov = lltOfA.matrixU();

  if ((cov.diagonal().array() < 0).any())
    throw std::out_of_range("Negative entries on covariance matrix diagonal");

}  // END: void Dpmm::cholupdate(VEC x, MAT cov)

/* ---------------------------------------------------------------------- */
template <typename VEC, typename MAT>
void Dpmm::icholupdate(const VEC& x, MAT& cov) {
  // Performs rank 1 update of the covariance.

  Eigen::MatrixXd tmpx = x * x.transpose();
  Eigen::MatrixXd tmpCov = cov.transpose() * cov;

  if (tmpCov.rows() != cov.rows() || tmpCov.cols() != cov.rows()) {
    throw std::length_error("Dimensions of the covariance are wrong.");
  }
  if (tmpx.rows() != cov.rows() || tmpx.cols() != cov.rows()) {
    throw std::length_error("Dyadic product x*x\' has wrong dimensions.");
  }

  Eigen::LLT<Eigen::MatrixXd> lltOfA(tmpCov - tmpx);
  cov = lltOfA.matrixU();

  if ((cov.diagonal().array() < 0).any())
    throw std::out_of_range("Negative entries on covariance matrix diagonal");
}  // END: void Dpmm::icholupdate(VEC x, MAT cov)

#endif