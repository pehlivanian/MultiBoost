#ifndef __GRADIENTBOOSTREGRESSOR_HPP__
#define __GRADIENTBOOSTREGRESSOR_HPP__

#include <cereal/types/polymorphic.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/archives/portable_binary.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/archives/xml.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/map.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/access.hpp>

#include <mlpack/core.hpp>

#include "compositeregressor.hpp"

using namespace arma;

template<typename RegressorType>
class GradientBoostRegressor : public CompositeRegressor<RegressorType> {
public:
  
  GradientBoostRegressor() = default;

  // 1
  // mat	: arma::Mat<double>
  // labels	: arma::Row<double>
  // context	: ModelContext::Context
  GradientBoostRegressor(const mat& dataset,
		     const Row<double>& labels,
		     Context context) :
    CompositeRegressor<RegressorType>(dataset, labels, context) {}

  // 2
  // mat		: arma::Mat<double>
  // labels		: arma::Row<double>
  // dataset_oos	: arma::Mat<double>
  // labels_oos		: arma::Row<std::double>
  // context		: ModelContext::Context
  GradientBoostRegressor(const mat& dataset,
		     const Row<double>& labels,
		     const mat& dataset_oos,
		     const Row<double> labels_oos,
		     Context context) :
    CompositeRegressor<RegressorType>(dataset, labels, dataset_oos, labels_oos, context) {}


  // 3
  // mat		: arma::Mat<double>
  // labels		: arma::Row<double>
  // latestPrediction	: arma::Mat<double>
  // colMask		: uvec
  // context		: ModelContext::Context
  GradientBoostRegressor(const mat& dataset,
			  const Row<double>& labels,
			  const Row<double>& latestPrediction,
			  const uvec& colMask,
			  Context context) :
    CompositeRegressor<RegressorType>(dataset, labels, latestPrediction, colMask, context) {}

  // 4
  // mat		: arma::Mat<double>
  // labels		: arma::Row<double>
  // latestPrediction	: arma::Mat<double>
  // context		: ModelContext::Context
  GradientBoostRegressor(const mat& dataset,
			  const Row<double>& labels,
			  const Row<double>& latestPrediction,
			  Context context) :
    CompositeRegressor<RegressorType>(dataset, labels, latestPrediction, context) {}

  
  // 5
  // mat		: arma::Mat<double>
  // labels		: arma::Row<double>
  // dataset_oos	: arma::Mat<double>
  // labels_oos		: Row<double>
  // latestPrediction	: arma::Mat<double>
  // colMask		: uvec
  // context		: ModelContext::Context
  GradientBoostRegressor(const mat& dataset,
			  const Row<double>& labels,
			  const mat& dataset_oos,
			  const Row<double>& labels_oos,
			  const Row<double>& latestPrediction,
			  const uvec& colMask,
			  Context context) :
    CompositeRegressor<RegressorType>(dataset, labels, dataset_oos, labels_oos, latestPrediction, colMask, context) {}

  // 6
  // mat		: arma::Mat<double>
  // labels		: arma::Row<double>
  // dataset_oos	: arma::Mat<double>
  // labels_oos		: Row<double>
  // latestPrediction	: arma::Mat<double>
  // context		: ModelContext::Context
  GradientBoostRegressor(const mat& dataset,
			  const Row<double>& labels,
			  const mat& dataset_oos,
			  const Row<double>& labels_oos,
			  const Row<double>& latestPrediction,
			  Context context) :
    CompositeRegressor<RegressorType>(dataset, labels, dataset_oos, labels_oos, latestPrediction, context) {}

  
};

# include "gradientboostregressor_impl.hpp"

using CompositeRegressorDTRR = CompositeRegressor<DecisionTreeRegressorRegressor>;

using GradientBoostRegressorDTRR = GradientBoostRegressor<DecisionTreeRegressorRegressor>;

CEREAL_REGISTER_TYPE(GradientBoostRegressorDTRR);
CEREAL_REGISTER_TYPE(CompositeRegressorDTRR);

CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelDTRR, CompositeRegressorDTRR);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelDTRR, GradientBoostRegressorDTRR);

#endif
