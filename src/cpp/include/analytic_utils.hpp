#ifndef __ANALYTIC_UTILS_HPP__
#define __ANALTYIC_UTILS_HPP__

#include <mlpack/core.hpp>

using namespace arma;

namespace ANALYTIC_utils {

  template<bool B>
  using bool_constant = std::integral_constant<bool, B>;

  template<typename DataType>
  struct _is_double : bool_constant<std::is_same<double, typename std::remove_cv<DataType>::type>::value> {};

  using arma_dvec_type = rowvec;
  using arma_fvec_type = frowvec;
  using arma_dmat_type = dmat;
  using arma_fmat_type = fmat;

  template<typename DataType>
  struct arma_vec_type {
    typedef arma_fvec_type value;
  };

  template<>
  struct arma_vec_type<double> {
    typedef arma_dvec_type value;
  };

  template<typename DataType>
  struct arma_mat_type {
    typedef arma_fmat_type value;
  };

  template<>
  struct arma_mat_type<double> {
    typedef arma_dmat_type value;
  };

  template<typename DataType>
  struct arma_types {
    typedef typename arma_vec_type<DataType>::value vectype;
    typedef typename arma_mat_type<DataType>::value mattype;
  };

} // namespace ANALYTIC_utils

#endif
