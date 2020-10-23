#include <stan/driver.hpp>
#include <functor/prod_iter.hpp>

namespace adb {

struct ProdIterFunc: ProdIterFuncBase
{};

using matvar = Eigen::Matrix<stan::math::var, Eigen::Dynamic, 1>;
BENCHMARK_TEMPLATE(BM_stan, ProdIterFunc, matvar)
  -> RangeMultiplier(2) -> Range(1, 1 << 14);

} // namespace adb
