#include <cppad/driver.hpp>
#include <functor/stochastic_volatility.hpp>

namespace adb {

struct StochasticVolatilityFunc: StochasticVolatilityFuncBase
{};

BENCHMARK_TEMPLATE(BM_cppad, StochasticVolatilityFunc)
    -> RangeMultiplier(2) -> Range(1, 1 << 14);

} // namespace adb

