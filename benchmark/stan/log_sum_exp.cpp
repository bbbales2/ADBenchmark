#include <stan/driver.hpp>
#include <functor/log_sum_exp.hpp>

namespace adb {

struct LogSumExpFunc: LogSumExpFuncBase
{
    stan::math::var operator()(const Eigen::Matrix<stan::math::var, Eigen::Dynamic, 1>& x) const
    {
        return stan::math::log_sum_exp(x);
    }
};

using matvar = Eigen::Matrix<stan::math::var, Eigen::Dynamic, 1>;
  BENCHMARK_TEMPLATE(BM_stan, LogSumExpFunc, matvar)
    -> RangeMultiplier(2) -> Range(1, 1 << 14);

} // namespace adb
