#ifndef __RECURSIVEMODEL_HPP__
#define __RECURSIVEMODEL_HPP__

#include <tuple>
#include <memory>
#include <vector>
#include <unordered_map>
#include <optional>

#include <boost/filesystem.hpp>

#include <mlpack/core.hpp>

#include "utils.hpp"
#include "DP.hpp"
#include "score2.hpp"
#include "constantclassifier.hpp"
#include "classifier.hpp"
#include "regressor.hpp"
#include "model_traits.hpp"

using namespace arma;

using namespace ModelContext;
using namespace Model_Traits;

template<typename DataType, typename ModelType>
class RecursiveModel {
public:

  using IntegralLabelType	= typename model_traits<ModelType>::integrallabeltype;
  using ModelInstT		= typename model_traits<ModelType>::model;
  using ModelList		= typename std::vector<std::unique_ptr<Model<DataType>>>;

  RecursiveModel() = default;

  auto _constantLeaf() -> Row<DataType> const;
  auto _constantLeaf(double val) -> Row<DataType> const;
  auto _randomLeaf() -> Row<DataType> const;
  void updateModels(std::unique_ptr<Model<DataType>>&&,
		    Row<DataType> prediction);

  ModelList models_;
  // Row<DataType> latestPrediction_;

  

};


#include "recursivemodel_impl.hpp"

#endif
