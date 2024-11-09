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

  using Leaves			= Row<DataType>;
  using Prediction		= Row<DataType>;
  using PredictionList		= std::vector<Prediction>;

  RecursiveModel() = default;

  RecursiveModel(const Row<DataType>& latestPrediction) :
    latestPrediction_{latestPrediction},
    hasInitialPrediction_{true},
    reuseColMask_{false}
  {}
  RecursiveModel(const Row<DataType>& latestPrediction, const uvec& colMask) :
    latestPrediction_{latestPrediction},
    colMask_{colMask},
    hasInitialPrediction_{true},
    reuseColMask_{true}
  {}
  RecursiveModel(const uvec& colMask) :
    colMask_{colMask},
    hasInitialPrediction_{false},
    reuseColMask_{true}
  {}
    
		 

  auto _constantLeaf() -> Row<DataType> const;
  auto _constantLeaf(double val) -> Row<DataType> const;
  auto _randomLeaf() -> Row<DataType> const;
  void updateModels(std::unique_ptr<Model<DataType>>&&, Row<DataType>&);

  ModelList getModels() const { return models_; }
  Row<DataType> getLatestPrediction() const { return latestPrediction_; }

  // void fit_step(int);


protected:

  Row<DataType> latestPrediction_;
  uvec colMask_;

  ModelList models_;
  PredictionList predictions_;

  bool reuseColMask_;
  bool hasInitialPrediction_;
  
  

};


#include "recursivemodel_impl.hpp"

#endif
