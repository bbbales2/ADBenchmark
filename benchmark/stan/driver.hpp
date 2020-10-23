#include <benchmark/benchmark.h>
#include <stan/math.hpp>
#include <util/check_gradient.hpp>

namespace adb {

template <class F, class V>
static void BM_stan(benchmark::State& state)
{
    F f;
    size_t N = state.range(0);

    Eigen::VectorXd x(N);
    f.fill(x);
    double fx;
    Eigen::VectorXd grad_fx(x.size());

    state.counters["N"] = x.size();

    int i = 0;
    for (auto _ : state) {
      V x_var(x);
      stan::math::var fx_var = f(x_var);

      fx_var.grad();
      if(i == 0)
	grad_fx = x_var.adj();
      stan::math::recover_memory();
      i++;
    }

    // sanity-check that output gradient is good
    Eigen::VectorXd expected(grad_fx.size());
    f.derivative(x, expected);
    check_gradient(grad_fx, expected, "stan-" + f.name());
}

} // namespace adb
