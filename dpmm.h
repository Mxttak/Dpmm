/*
Implementation of the Dirichlet Process Gaussian Mixture Model (DPGMM) according
to the Matlab implementation by Jacob Eisenstein.

Adaptive rejection sampling of the parameter alpha uses the digamma function by
Richard J. Mathar that is available at
http://www2.mpia-hd.mpg.de/~mathar/progs/digamma.c (for more info, there is a
header in myars.h)

ToDo: make a separate class for adaptive rejection sampling

Copyright (C) 2016 Maxim Dolgov: m<dot>dolgov<at>web<dot>de
No warranty, no commercial use.

Definitely contains errors; use with caution.
*/

#ifndef _DPMM_H_
#define _DPMM_H_

#include "Eigen/Eigen"

#include "time.h"

#include <vector>
#include <stdexcept>
#include <random>
#include <numeric>    // std::iota
#include <cmath>      // lgamma
#include <algorithm>  // std::sort

#include "mex.h"

// some macros
#ifndef M_PIl
/** The constant Pi in high precision */
#define M_PIl 3.1415926535897932384626433832795029L
#endif
#ifndef M_GAMMAl
/** Euler's constant in high precision */
#define M_GAMMAl 0.5772156649015328606065120900824024L
#endif
#ifndef M_LN2l
/** the natural logarithm of 2 in high precision */
#define M_LN2l 0.6931471805599453094172321214581766L
#endif
// END: some macros

/* ---------------------------------------------------------------------- */
class DPMM {
 public:
  double alpha;
  double kappa;
  double nu;
  Eigen::VectorXd initmean;
  Eigen::MatrixXd initcov;
  double num_clusters;
  std::vector<double> counts;            // numbers of samples per cluster
  std::vector<Eigen::VectorXd> sums;     // pseudo means (see Sudderth)
  std::vector<Eigen::MatrixXd> cholSSE;  // pseudo covariances (see Sudderth)
  std::vector<double> clusters;          // assignments to clusters

  Eigen::MatrixXd samples;

  size_t numSamples;
  size_t dimSamples;

  std::vector<size_t> indexes;

  std::mt19937 gen;  // mersenne twister random number generator
  std::uniform_real_distribution<double> dis;  // distribution

 public:
  DPMM() {
    // set random number generators
    gen.seed((size_t)time(0));
    dis.param(std::uniform_real_distribution<double>::param_type(0.0, 1.0));
  };
  template <typename Mat>
  void InitDefault(const Mat& smpls);
  void RemoveEmptyClusters(void);
  void AddNewCluster(void);
  void popSample(size_t cluster, size_t sampleNum);
  void putSample(size_t cluster, size_t sampleNum);
  void gibbsSampling(size_t numGibbsIter);
  template <typename Mat>
  void ComputeClusterProbs(size_t pos, Mat& clusterprobs);
  template <typename Vec>
  size_t SampleCluster(const Vec& clusterprobs);
  double SampleAlpha(void);

 private:
  template <typename Mat, typename Vec1, typename Vec2>
  double lognormpdf(const Vec1& x, const Vec2& m, Mat& C);
  void logalphapdf(double alpha, double k, double n, double& h, double& hprime);
  void sortarsparams(std::vector<double>& x, std::vector<double>& h,
                     std::vector<double>& hprime);
  void computehusccu(std::vector<double>& x, std::vector<double>& h,
                     std::vector<double>& hprime, std::vector<double>& hu,
                     std::vector<double>& sc, double& cu, double zmin);
  template <typename Mat, typename Vec>
  void rankupdate(Mat& C, const Vec& x, double a);
  long double digammal(long double x);
};  // END: DPMM

/* ---------------------------------------------------------------------- */
void DPMM::AddNewCluster(void) {
  // appends an empty cluster to the list of clusters
  num_clusters += 1;
  counts.push_back(0);
  sums.push_back(kappa * initmean);
  Eigen::MatrixXd tmp =
      (nu * initcov).selfadjointView<Eigen::Upper>().llt().matrixU();
  cholSSE.push_back(tmp);
}  // END: void DPMM::AddNewCluster(void)

/* ---------------------------------------------------------------------- */
void DPMM::RemoveEmptyClusters(void) {
  size_t maxiter = (size_t)num_clusters;
  for (size_t iter = 0; iter < maxiter; iter++) {
    if (counts.at(iter) < 1) {
      // empty cluster
      counts.erase(counts.begin() + iter);
      sums.erase(sums.begin() + iter);
      cholSSE.erase(cholSSE.begin() + iter);
      // decrease cluster numbers for samples
      for (size_t i = 0; i < numSamples; i++) {
        if ((size_t)clusters.at(i) > iter) clusters.at(i) -= 1;
      }
      num_clusters -= 1;
      maxiter -= 1;
      iter -= 1;
    }
  }

}  // END: void DPMM::RemoveEmptyClusters(void)

/* ---------------------------------------------------------------------- */
template <typename Mat>
void DPMM::InitDefault(const Mat& smpls) {
  numSamples = smpls.cols();
  dimSamples = smpls.rows();

  samples = smpls;

  initmean = samples.rowwise().mean();
  initcov = 0.1 * (samples.colwise() - initmean) *
            ((samples.colwise() - initmean).adjoint()) / (samples.cols() - 1.0);

  // default parameters taken from Jacob Eisenstein's code
  alpha = (double)numSamples / 50;
  nu = 6.0;
  kappa = .1;
  num_clusters = 0;

  clusters = std::vector<double>(numSamples, 1);

  AddNewCluster();

  cholSSE.at(0) = cholSSE.at(0).transpose() * cholSSE.at(0);

  rankupdate(cholSSE.at(0), sums.at(0), 1.0 / kappa);
  sums.at(0) += samples.rowwise().sum();
  counts.at(0) = numSamples;
  for (size_t iterSamples = 0; iterSamples < (size_t)numSamples; iterSamples++)
    rankupdate(cholSSE.at(0), samples.col(iterSamples), 1.0);

  rankupdate(cholSSE.at(0), sums.at(0), -1.0 / (kappa + numSamples));
  cholSSE.at(0) = cholSSE.at(0).selfadjointView<Eigen::Upper>().llt().matrixU();

}  // END: void DPMM::InitDefault(const Vec& m, const Mat& C)

/* ---------------------------------------------------------------------- */
void DPMM::putSample(size_t cluster, size_t sampleNum) {
  // puts the sample into the cluster (unhideObservation))

  cholSSE.at(cluster) = cholSSE.at(cluster).transpose() * cholSSE.at(cluster);

  rankupdate(cholSSE.at(cluster), sums.at(cluster),
             1.0 / (kappa + counts.at(cluster)));
  rankupdate(cholSSE.at(cluster), samples.col(sampleNum), 1.0);

  sums.at(cluster) += samples.col(sampleNum);
  counts.at(cluster) += 1;

  rankupdate(cholSSE.at(cluster), sums.at(cluster),
             -1.0 / (kappa + counts.at(cluster)));

  cholSSE.at(cluster) =
      cholSSE.at(cluster).selfadjointView<Eigen::Upper>().llt().matrixU();

}  // END: void DPMM::popSample(size_t cluster, Vec& sample)

/* ---------------------------------------------------------------------- */
void DPMM::popSample(size_t cluster, size_t sampleNum) {
  // remove sample from cluster (hideObservation)
  cholSSE.at(cluster) = cholSSE.at(cluster).transpose() * cholSSE.at(cluster);

  if ((size_t)counts.at(cluster) < 1)
    throw std::out_of_range("Trying to pop sample from empty cluster");

  rankupdate(cholSSE.at(cluster), sums.at(cluster),
             1.0 / (kappa + counts.at(cluster)));

  counts.at(cluster) -= 1;
  if (counts.at(cluster) < 0) throw std::out_of_range("Negative sample count.");
  sums.at(cluster) -= samples.col(sampleNum);

  rankupdate(cholSSE.at(cluster), samples.col(sampleNum), -1.0);

  rankupdate(cholSSE.at(cluster), sums.at(cluster),
             -1.0 / (kappa + counts.at(cluster)));

  cholSSE.at(cluster) =
      cholSSE.at(cluster).selfadjointView<Eigen::Upper>().llt().matrixU();
}  // END: void DPMM::popSample(size_t cluster, Vec& sample)

/* ---------------------------------------------------------------------- */
void DPMM::gibbsSampling(size_t numGibbsIter) {
  // create vector of sample indexes
  indexes = std::vector<size_t>(numSamples);
  std::iota(indexes.begin(), indexes.end(), 0);
  std::random_shuffle(indexes.begin(), indexes.end());

  for (size_t gibbsIter = 0; gibbsIter < numGibbsIter; gibbsIter++) {
    // shuffle sample indexes
    std::random_shuffle(indexes.begin(), indexes.end());

    size_t pos, curCluster, newCluster;

    // iterate through samples
    for (size_t iter = 0; iter < numSamples; iter++) {
      // get index of current sample
      pos = indexes.at(iter);

      // get the index of cluster that will be updated
      curCluster = (size_t)clusters.at(pos);

      // remove sample from cluster and update its parameters
      popSample(curCluster - 1, pos);

      // remove empty clusters
      RemoveEmptyClusters();

      // add a new clusters
      AddNewCluster();

      // compute the likelihood values for all clusters
      Eigen::VectorXd clusterprobs((size_t)num_clusters);
      ComputeClusterProbs(pos, clusterprobs);

      // sample a new cluster for the current sample
      newCluster = SampleCluster(clusterprobs);

      // update parameters of this cluster
      clusters.at(pos) = (double)newCluster + 1;
      putSample(newCluster, pos);

    }  // for: iter: 0..numSamples-1: iteration through samples during a single
       // Gibbs run

    // sample a new alpha
    alpha = SampleAlpha();

  }  // for: gibbsIter: 0..numGibbsIter-1: Gibbs sampling with a prespecified
     // number of iterations
}  // END: void DPMM::gibbsSampling(size_t numGibbsIter)

/* ---------------------------------------------------------------------- */
template <typename Mat>
void DPMM::ComputeClusterProbs(size_t pos, Mat& clusterprobs) {
  Eigen::VectorXd log_p_obs((size_t)num_clusters);
  double kappabar;

  // parallelization is slower on my machine...
  // #pragma omp parallel for // change iter to int (openmp want a signed
  // integer)
  for (size_t iter = 0; iter < (size_t)num_clusters; iter++) {
    kappabar = counts.at(iter) + kappa;
    log_p_obs(iter) =
        lognormpdf(samples.col(pos), sums.at(iter) / kappabar,
                   sqrt((kappabar + 1) /
                        (kappabar * (nu + counts.at(iter) - dimSamples - 1))) *
                       cholSSE.at(iter));
    clusterprobs(iter) = counts.at(iter);
    if ((size_t)counts.at(iter) == 0) clusterprobs(iter) += alpha;
  }
  clusterprobs = clusterprobs.array() *
                 (log_p_obs.array() - log_p_obs.maxCoeff()).exp().array();

  clusterprobs /= clusterprobs.sum();
}  // END: void DPMM::ComputeClusterProbs(size_t pos, Mat& clusterprobs)

/* ---------------------------------------------------------------------- */
template <typename Vec>
size_t DPMM::SampleCluster(const Vec& clusterprobs) {
  // draw a sample from Uniform(0,1)
  double randnum = dis(gen);
  double cum = clusterprobs(0);

  // look into which cluster the number falls
  for (size_t iter = 0; iter < clusterprobs.size() - 1; iter++) {
    if (randnum < cum) return iter;
    cum += clusterprobs(iter + 1);
  }
  return ((size_t)clusterprobs.size() - 1);
}  // END: size_t DPMM::SampleCluster(const Vec& clusterprobs)

/* ---------------------------------------------------------------------- */
double DPMM::SampleAlpha(void) {
  // adaptive rejection sampling of the alpha parameter

  // initialize
  std::vector<double> x(2);
  std::vector<double> h(2);
  std::vector<double> hprime(2);
  std::vector<double> hu;
  std::vector<double> sc;
  double cu;

  double offset, u[2], xt, ht, hpt, hut;

  x.at(0) = 2 / ((double)numSamples - num_clusters + 1.5);  // deriv_up
  x.at(1) = num_clusters * (double)numSamples /
            ((double)numSamples - num_clusters + 1.0);  // deriv_down

  double zmin = x.at(0);
  std::sort(x.begin(), x.end());
  for (size_t i = 0; i < 2; i++)
    logalphapdf(x.at(i), num_clusters, (double)numSamples, h.at(i),
                hprime.at(i));

  if (hprime.at(0) < 0 || hprime.at(1) > 0)
    throw std::out_of_range("Starting points do not enclose the mode.");

  offset = std::max(h.at(0), h.at(1));
  // h-offset
  std::transform(h.begin(), h.end(), h.begin(),
                 [offset](double a) { return a - offset; });

  // repeat until a sample is found
  while (1) {
    // draw 2 random numbers from Uniform(0,1)
    u[0] = dis(gen);
    u[1] = dis(gen);

    // find the largest z such that sc(z) < u
    size_t index;
    computehusccu(x, h, hprime, hu, sc, cu, zmin);
    for (index = 1; index < sc.size(); index++) {
      if (sc.at(index) / cu >= u[0]) {
        index--;
        break;
      }
    }

    // Figure out the x in that segment that u corresponds to
    xt = x.at(index) +
         (-h.at(index) + log(hprime.at(index) * (cu * u[0] - sc.at(index)) +
                             exp(hu.at(index)))) /
             hprime.at(index);
    logalphapdf(xt, num_clusters, (double)numSamples, ht, hpt);
    ht -= offset;

    // Figure out what h_u(xt) is a dumb way, uses assumption that the log pdf
    // is concave
    hut = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < x.size(); i++) {
      hut = std::min(hut, hprime.at(i) * (xt - x.at(i)) + h.at(i));
    }

    // Decide whether to keep the sample
    if (u[1] < exp(ht - hut)) return xt;

    // update vectors
    x.push_back(xt);
    h.push_back(ht);
    hprime.push_back(hpt);

    sortarsparams(x, h, hprime);
  }  // while: repeat until a sample is found

}  // END: double DPMM::SampleAlpha(void)

/* ---------------------------------------------------------------------- */
template <typename Mat, typename Vec1, typename Vec2>
double DPMM::lognormpdf(const Vec1& x, const Vec2& m, Mat& C) {
  // const double log2pi = 1.83787706640935;
  Eigen::VectorXd t = C.transpose().inverse() * (x - m);
  double res = -log(C.diagonal().prod()) -
               0.5 * (x.rows() * 1.83787706640935 + t.transpose() * t);
  if (res != res) {
    throw std::out_of_range("Result of lognormpdf is NaN.");
  }
}  // END: double DPMM::lognormpdf(const Vec& x, const Vec& m, Mat& C)

/* ---------------------------------------------------------------------- */
void DPMM::logalphapdf(double alpha, double k, double n, double& h,
                       double& hprime) {
  if (alpha <= 0) throw std::out_of_range("Non-positive parameter alpha.");

  h = (k - 1.5) * log(alpha) - 1 / (2 * alpha) + lgamma(alpha) -
      lgamma(n + alpha);
  hprime = (k - 1.5) / alpha + 1 / (2 * alpha * alpha) + digammal(alpha) -
           digammal(n + alpha);
}  // END: void DPMM::logalphapdf(double alpha, double k, double n)

/* ---------------------------------------------------------------------- */
template <typename Mat, typename Vec>
void DPMM::rankupdate(Mat& C, const Vec& x, double a) {
  C = (C.selfadjointView<Eigen::Upper>().rankUpdate(x, a));
  //  .llt().matrixU();
}  // END: void DPMM::rankupdate(Mat& Cnew, Mat& C, const Vec& x, double a)

/*************************************
 An ANSI-C implementation of the digamma-function for real arguments based
 on the Chebyshev expansion proposed in appendix E of
 http://arXiv.org/abs/math.CA/0403344 . This is identical to the implementation
 by Jet Wimp, Math. Comp. vol 15 no 74 (1961) pp 174 (see Table 1).
 For other implementations see
 the GSL implementation for Psi(Digamma) in
 http://www.gnu.org/software/gsl/manual/html_node/Psi-_0028Digamma_0029-Function.html

Richard J. Mathar, 2005-11-24
**************************************/
long double DPMM::digammal(long double x) {
  /* force into the interval 1..3 */
  if (x < 0.0L)
    return digammal(1.0L - x) +
           M_PIl / tanl(M_PIl * (1.0L - x)); /* reflection formula */
  else if (x < 1.0L)
    return digammal(1.0L + x) - 1.0L / x;
  else if (x == 1.0L)
    return -M_GAMMAl;
  else if (x == 2.0L)
    return 1.0L - M_GAMMAl;
  else if (x == 3.0L)
    return 1.5L - M_GAMMAl;
  else if (x > 3.0L)
    /* duplication formula */
    return 0.5L * (digammal(x / 2.0L) + digammal((x + 1.0L) / 2.0L)) + M_LN2l;
  else {
    /* Just for your information, the following lines contain
    * the Maple source code to re-generate the table that is
    * eventually becoming the Kncoe[] array below
    * interface(prettyprint=0) :
    * Digits := 63 :
    * r := 0 :
    *
    * for l from 1 to 60 do
    * 	d := binomial(-1/2,l) :
    * 	r := r+d*(-1)^l*(Zeta(2*l+1) -1) ;
    * 	evalf(r) ;
    * 	print(%,evalf(1+Psi(1)-r)) ;
    *o d :
    *
    * for N from 1 to 28 do
    * 	r := 0 :
    * 	n := N-1 :
    *
    *	for l from iquo(n+3,2) to 70 do
    *		d := 0 :
    *		for s from 0 to n+1 do
    *		 d := d+(-1)^s*binomial(n+1,s)*binomial((s-1)/2,l) :
    *		od :
    *		if 2*l-n > 1 then
    *		r := r+d*(-1)^l*(Zeta(2*l-n) -1) :
    *		fi :
    *	od :
    *	print(evalf((-1)^n*2*r)) ;
    *od :
    *quit :
    */
    static long double Kncoe[] = {.30459198558715155634315638246624251L,
                                  .72037977439182833573548891941219706L,
                                  -.12454959243861367729528855995001087L,
                                  .27769457331927827002810119567456810e-1L,
                                  -.67762371439822456447373550186163070e-2L,
                                  .17238755142247705209823876688592170e-2L,
                                  -.44817699064252933515310345718960928e-3L,
                                  .11793660000155572716272710617753373e-3L,
                                  -.31253894280980134452125172274246963e-4L,
                                  .83173997012173283398932708991137488e-5L,
                                  -.22191427643780045431149221890172210e-5L,
                                  .59302266729329346291029599913617915e-6L,
                                  -.15863051191470655433559920279603632e-6L,
                                  .42459203983193603241777510648681429e-7L,
                                  -.11369129616951114238848106591780146e-7L,
                                  .304502217295931698401459168423403510e-8L,
                                  -.81568455080753152802915013641723686e-9L,
                                  .21852324749975455125936715817306383e-9L,
                                  -.58546491441689515680751900276454407e-10L,
                                  .15686348450871204869813586459513648e-10L,
                                  -.42029496273143231373796179302482033e-11L,
                                  .11261435719264907097227520956710754e-11L,
                                  -.30174353636860279765375177200637590e-12L,
                                  .80850955256389526647406571868193768e-13L,
                                  -.21663779809421233144009565199997351e-13L,
                                  .58047634271339391495076374966835526e-14L,
                                  -.15553767189204733561108869588173845e-14L,
                                  .41676108598040807753707828039353330e-15L,
                                  -.11167065064221317094734023242188463e-15L};

    register long double Tn_1 = 1.0L;   /* T_{n-1}(x), started at n=1 */
    register long double Tn = x - 2.0L; /* T_{n}(x) , started at n=1 */
    register long double resul = Kncoe[0] + Kncoe[1] * Tn;

    x -= 2.0L;

    for (int n = 2; n < sizeof(Kncoe) / sizeof(long double); n++) {
      const long double Tn1 =
          2.0L * x * Tn -
          Tn_1; /* Chebyshev recursion, Eq. 22.7.4 Abramowitz-Stegun */
      resul += Kncoe[n] * Tn1;
      Tn_1 = Tn;
      Tn = Tn1;
    }
    return resul;
  }
}

/* ---------------------------------------------------------------------- */
void DPMM::sortarsparams(std::vector<double>& x, std::vector<double>& h,
                         std::vector<double>& hprime) {
  if (x.size() != h.size() || x.size() != hprime.size())
    throw std::length_error("Lengthes of input arrays are not equal.");

  // iterate and swap (x is already presorted -> only one element must be moved
  // => linear runtime)
  for (unsigned int i = x.size() - 1; i > 0; i--) {
    if (x.at(i) < x.at(i - 1)) {
      std::swap(x.at(i), x.at(i - 1));
      std::swap(h.at(i), h.at(i - 1));
      std::swap(hprime.at(i), hprime.at(i - 1));
    }
  }
}  // END: void DPMM::sortarsparams(std::vector<double>& x, std::vector<double>&
// h, std::vector<double>& hprime)

/* ---------------------------------------------------------------------- */
void DPMM::computehusccu(std::vector<double>& x, std::vector<double>& h,
                         std::vector<double>& hprime, std::vector<double>& hu,
                         std::vector<double>& sc, double& cu, double zmin) {
  size_t len = x.size();
  hu = std::vector<double>(len + 1);
  sc = std::vector<double>(len + 1);
  double z, tmp;

  hu.at(0) = hprime.at(0) * (zmin - x.at(0)) + h.at(0);
  tmp = exp(hu.at(0));
  sc.at(0) = 0;
  for (size_t i = 1; i < len; i++) {
    // (hprime(2:end).*diff(x)-diff(h)) ./ diff(hprime)
    z = (hprime.at(i) * (x.at(i) - x.at(i - 1)) + h.at(i - 1) - h.at(i)) /
        (hprime.at(i) - hprime.at(i - 1));

    hu.at(i) = hprime.at(i - 1) * z + h.at(i - 1);

    sc.at(i) = (exp(hu.at(i)) - tmp) / hprime.at(i - 1);
  }

  hu.at(len) = hprime.at(len - 1) * std::numeric_limits<double>::infinity() +
               h.at(len - 1);

  sc.at(len) = -tmp / hprime.at(len - 1);

  cu = sc.at(len);
}  // END: void DPMM::computehusccu(std::vector<double>& x, std::vector<double>&
   // h, std::vector<double>& hprime, std::vector<double>& hu,
   // std::vector<double>& sc, double cu)

#endif  // _DPMM_H_
