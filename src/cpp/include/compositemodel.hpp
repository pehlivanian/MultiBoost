#ifndef __COMPOSITE_MODEL_HPP__
#define __COMPOSITE_MODEL_HPP__

#include <tuple>
#include <memory>
#include <vector>

#include <mlpack/core.hpp>

#include "utils.hpp"
#include "DP.hpp"
#include "score2.hpp"
#include "classifier.hpp"
#include "model.hpp"
#include "model_traits.hpp"

using namespace arma;

using namespace ModelContext;
using namespace Model_Traits;
using namespace IB_utils;

template<typename ModelType>
class CompositeModel {
  using model = typename model_traits<ModelType>::model;

public:
  CompositeModel() = default;


};

#include "compositemodel_impl.hpp"

#endif
