#include "utils.hpp"

namespace IB_utils {
  
  double err(const Row<std::size_t>& yhat, const Row<std::size_t>& y) {
    return accu(yhat != y) * 100. / y.n_elem;
  }

}
