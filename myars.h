/*
Adaptive rejection sampling to draw an update of the parameter alpha of a DPGMM.

Adapted from code by Michael Mandel that is included in the DPMM code by Jacob
Eisenstein.

I use the digamma function by Richard J. Mathar that is available at
http://www2.mpia-hd.mpg.de/~mathar/progs/digamma.c. See the header below for
more info.

Copyright (C) 2016 Maxim Dolgov: m<dot>dolgov<at>web<dot>de
No warranty, no commercial use.

Definitely contains errors; use with caution.
*/

#ifndef _MY_ARS_H_
#define _MY_ARS_H_

#include <cmath>
#include <stdexcept>
#include <ctime>
#include <random>
#include <vector>
#include <algorithm>  // std::sort
#include <numeric>    // std::iota
#include <utility>    // std::swap

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
long double digammal(long double x) {
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
/* ---------------------------------------------------------------------- */
class AlphaSampler {
 public:
  std::vector<double> x;
  std::vector<double> h;
  std::vector<double> hprime;
  std::vector<double> hu;
  std::vector<double> sc;
  double k, n, cu, zmin, ht, hpt, xt, offset;

  std::mt19937 gen;  // mersenne twister random number generator
  std::uniform_real_distribution<double> dis;

 public:
  AlphaSampler(){};
  AlphaSampler(double k, double n);
  void Init(double kt, double nt);
  void logpdf(const double& x, double& h, double& hprime);
  void computehusccu(void);
  void sortx(void);
  double drawSample(void);
};  // END: class AlphaSampler

/* ---------------------------------------------------------------------- */
AlphaSampler::AlphaSampler(double kt, double nt) {
  // k: number of clusters
  // n: number of samples

  Init(kt, nt);
}  // END: AlphaSampler::AlphaSampler(double k, double n)

/* ---------------------------------------------------------------------- */
void AlphaSampler::Init(double kt, double nt) {
  // k: number of clusters
  // n: number of samples

  clock_t t = clock();

  k = kt;
  n = nt;

  // initialize first points
  x.push_back(2 / (n - k + 1.5));
  x.push_back(k * n / (n - k + 1));
  zmin = x.at(0);
  std::sort(x.begin(), x.end());

  // initialize random number generator
  dis.param(std::uniform_real_distribution<double>::param_type(0.0, 1.0));
  gen.seed(clock() - t + time(0));

  // compute h and hprime of the first points
  logpdf(x.at(0), ht, hpt);
  h.push_back(ht);
  hprime.push_back(hpt);
  logpdf(x.at(1), ht, hpt);
  h.push_back(ht);
  hprime.push_back(hpt);

  if (h.at(0) != h.at(0)) throw std::out_of_range("Infinite h.");
  if (hprime.at(0) < 0 || hprime.at(1) > 0)
    throw std::out_of_range("hprime is out of range");

  // remove offset because the envelope and pdf are only proportional to the
  // true pdf
  offset = std::max(h.at(0), h.at(1));
  h.at(0) -= offset;
  h.at(1) -= offset;
}  // END: void AlphaSampler::Init(double kt, double nt)

/* ---------------------------------------------------------------------- */
void AlphaSampler::logpdf(const double& x, double& h, double& hprime) {
  if (x <= 0) throw std::out_of_range("Negative alpha");

  h = ((k - 1.5) * log(x) - 1 / (2 * x) + lgamma(x) - lgamma(x + n));
  hprime =
      ((k - 1.5) / x + 1 / (2 * x * x) + (digammal(x)) - (digammal(x + n)));
}  // END: void AlphaSampler::logpdf(const double& x, double& h, double& hprime)

/* ---------------------------------------------------------------------- */
void AlphaSampler::computehusccu(void) {
  unsigned int len = x.size();
  double z;
  hu = std::vector<double>(len + 1);
  sc = std::vector<double>(len + 1);

  sc.at(0) = 0;

  hu.at(0) = hprime.at(0) * (zmin - x.at(0)) + h.at(0);
  hu.at(len) = hprime.at(len - 1) *
                   (std::numeric_limits<double>::infinity() - x.at(len - 1)) +
               h.at(len - 1);
  for (unsigned int i = 0; i < len - 1; i++) {
    // z = [support(1) x(1:end-1)+(-diff(h)+hprime(2:end).*diff(x)) ./ ...
    // diff(hprime) support(end)]
    z = x.at(i) +
        (h.at(i) - h.at(i + 1) + hprime.at(i + 1) * (x.at(i + 1) - x.at(i))) /
            (hprime.at(i + 1) - hprime.at(i));

    // hu = [hprime(1) hprime] .* (z - [x(1) x]) + [h(1) h];
    hu.at(i + 1) = hprime.at(i) * (z - x.at(i)) + h.at(i);

    // sc = cumsum(diff(exp(hu));
    sc.at(i + 1) =
        (exp(hu.at(i + 1)) - exp(hu.at(i))) / hprime.at(i) + sc.at(i);
  }
  sc.at(len) = (exp(hu.at(len)) - exp(hu.at(len - 1))) / hprime.at(len - 1) +
               sc.at(len - 1);

  cu = sc.at(len);
}  // END: void AlphaSampler::computehusccu(void)

/* ---------------------------------------------------------------------- */
void AlphaSampler::sortx(void) {
  if (x.size() != h.size() || x.size() != hprime.size())
    throw std::length_error("Lengthes of input arrays are not equal.");

  // iterate and swap (x is already presorted -> only one element must be moved)
  for (unsigned int i = x.size() - 1; i > 0; i--) {
    if (x.at(i) < x.at(i - 1)) {
      std::swap(x.at(i), x.at(i - 1));
      std::swap(h.at(i), h.at(i - 1));
      std::swap(hprime.at(i), hprime.at(i - 1));
    }
  }
}  // std::vector<int> mysort(std::vector<double>& x)

/* ---------------------------------------------------------------------- */
double AlphaSampler::drawSample(void) {
  while (1) {
    computehusccu();

    // draw two random numbers from Uniform(0,1)
    double u[2];
    u[0] = dis(gen);
    u[1] = dis(gen);

    // find the largest index z such that sc(z) < u[0]
    double idx;
    u[0] *= cu;
    for (idx = 0; idx < sc.size(); idx++) {
      if (sc.at(idx) > u[0]) break;
    }
    idx -= 1;
    if (idx < 0 || (unsigned int)idx >= x.size()) {
      throw std::out_of_range("Index is either negative or out of range.");
    }

    // determine the corresponding x
    xt = x.at(idx) +
         (-h.at(idx) +
          log(hprime.at(idx) * (u[0] - sc.at(idx)) + exp(hu.at(idx)))) /
             hprime.at(idx);

    logpdf(xt, ht, hpt);
    ht -= offset;

    // determine h_u(xt)
    double hut = hprime.at(0) * (xt - x.at(0)) + h.at(0);
    double hutt;
    for (unsigned int i = 1; i < x.size(); i++) {
      hutt = hprime.at(i) * (xt - x.at(i)) + h.at(i);
      if (hutt < hut) hut = hutt;
    }
    // decide whether to keep the samples
    if (u[1] < exp(ht - hut)) return xt;

    // otherwise update the vectors
    x.push_back(xt);
    h.push_back(ht);
    hprime.push_back(hpt);
    sortx();
  }
}  // END: double AlphaSampler::drawSample(void)

#endif