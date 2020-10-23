#include <stan/driver.hpp>
#include <functor/sum.hpp>

namespace adb {

struct SumFunc: SumFuncBase
{
  template <typename T>
  stan::math::var operator()(const T& x) const
  {
    return stan::math::sum(x);
  }
};

using varmat = stan::math::var_value<Eigen::VectorXd>;
using matvar = Eigen::Matrix<stan::math::var, Eigen::Dynamic, 1>;
BENCHMARK_TEMPLATE(BM_stan, SumFunc, varmat)
    -> RangeMultiplier(2) -> Range(1, 1 << 14);
BENCHMARK_TEMPLATE(BM_stan, SumFunc, matvar)
    -> RangeMultiplier(2) -> Range(1, 1 << 14);

} // namespace adb
