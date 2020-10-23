#include <stan/driver.hpp>
#include <functor/prod.hpp>

namespace adb {

struct ProdFunc: ProdFuncBase
{};

using matvar = Eigen::Matrix<stan::math::var, Eigen::Dynamic, 1>;
BENCHMARK_TEMPLATE(BM_stan, ProdFunc, matvar)
    -> RangeMultiplier(2) -> Range(1, 1 << 14);

} // namespace adb
