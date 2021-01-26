#include <stan/driver.hpp>
#include <functor/regression.hpp>

namespace adb {

struct RegressionFunc: RegressionFuncBase
{
  template <typename T>
  auto operator()(const T& x) const
  {
    using namespace stan::math;
    size_t N = (x.size() - 2);
    const auto& w = x.head(N);
    const auto& b = x(N);
    const auto& sigma = x(N + 1);
    return normal_lpdf(y, add(multiply(X, w), b), sigma) +
      normal_lpdf(w, 0., 1.) +
      normal_lpdf(b, 0., 1.) -
      uniform_lpdf(sigma, 0.1, 10.);
  }
};

using varmat = stan::math::var_value<Eigen::VectorXd>;
using matvar = Eigen::Matrix<stan::math::var, Eigen::Dynamic, 1>;
BENCHMARK_TEMPLATE(BM_stan, RegressionFunc, matvar)
    -> RangeMultiplier(2) -> Range(1, 1 << 14);
BENCHMARK_TEMPLATE(BM_stan, RegressionFunc, varmat)
    -> RangeMultiplier(2) -> Range(1, 1 << 14);

} // namespace adb
