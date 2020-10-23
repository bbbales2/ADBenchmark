#include <stan/driver.hpp>
#include <functor/sum_iter.hpp>

namespace adb {

struct SumIterFunc: SumIterFuncBase
{};

using matvar = Eigen::Matrix<stan::math::var, Eigen::Dynamic, 1>;
BENCHMARK_TEMPLATE(BM_stan, SumIterFunc, matvar)
    -> RangeMultiplier(2) -> Range(1, 1 << 14);

} // namespace adb
