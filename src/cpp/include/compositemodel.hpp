#ifndef __COMPOSITE_MODEL_HPP__
#define __COMPOSITE_MODEL_HPP__

#include <memory>
#include <mlpack/core.hpp>
#include <tuple>
#include <vector>

#include "DP.hpp"
#include "classifier.hpp"
#include "model.hpp"
#include "model_traits.hpp"
#include "score2.hpp"
#include "utils.hpp"

using namespace arma;

using namespace ModelContext;
using namespace Model_Traits;
using namespace IB_utils;

template <typename ModelType>
class CompositeModel {
  using model = typename model_traits<ModelType>::model;

public:
  CompositeModel() = default;
};

#include "compositemodel_impl.hpp"

#endif
