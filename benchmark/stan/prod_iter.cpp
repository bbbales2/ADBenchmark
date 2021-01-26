#include <stan/driver.hpp>
#include <functor/prod_iter.hpp>

namespace adb {

struct ProdIterFunc: ProdIterFuncBase
{
  template <class T>
  stan::math::var operator()(const T& x) const {
    using namespace stan::math;

    stan::math::var res = 1.0;
    for(size_t i = 0; i < x.size(); i++) {
      res *= x.coeffRef(i);
    }

    return res;
  }
};

using varmat = stan::math::var_value<Eigen::VectorXd>;
using matvar = Eigen::Matrix<stan::math::var, Eigen::Dynamic, 1>;
BENCHMARK_TEMPLATE(BM_stan, ProdIterFunc, matvar)
  -> RangeMultiplier(2) -> Range(1, 1 << 14);
BENCHMARK_TEMPLATE(BM_stan, ProdIterFunc, varmat)
  -> RangeMultiplier(2) -> Range(1, 1 << 14);

} // namespace adb
