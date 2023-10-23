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
  GradientBoostRegressor(const Mat<typename CompositeRegressor<RegressorType>::DataType>& dataset,
			 const Row<typename CompositeRegressor<RegressorType>::DataType>& labels,
			 Context context,
			 const std::string& folderName=std::string{}) :
    CompositeRegressor<RegressorType>(dataset, labels, context, folderName) {}

  // 2
  // mat		: arma::Mat<double>
  // labels		: arma::Row<double>
  // dataset_oos	: arma::Mat<double>
  // labels_oos		: arma::Row<std::double>
  // context		: ModelContext::Context
  GradientBoostRegressor(const Mat<typename CompositeRegressor<RegressorType>::DataType>& dataset,
			 const Row<typename CompositeRegressor<RegressorType>::DataType>& labels,
			 const Mat<typename CompositeRegressor<RegressorType>::DataType>& dataset_oos,
			 const Row<typename CompositeRegressor<RegressorType>::DataType> labels_oos,
			 Context context,
			 const std::string& folderName=std::string{}) :
    CompositeRegressor<RegressorType>(dataset, labels, dataset_oos, labels_oos, context, folderName) {}


  // 3
  // mat		: arma::Mat<double>
  // labels		: arma::Row<double>
  // latestPrediction	: arma::Mat<double>
  // colMask		: uvec
  // context		: ModelContext::Context
  GradientBoostRegressor(const Mat<typename CompositeRegressor<RegressorType>::DataType>& dataset,
			 const Row<typename CompositeRegressor<RegressorType>::DataType>& labels,
			 const Row<typename CompositeRegressor<RegressorType>::DataType>& latestPrediction,
			 const uvec& colMask,
			 Context context,
			 const std::string& folderName=std::string{}) :
    CompositeRegressor<RegressorType>(dataset, labels, latestPrediction, colMask, context, folderName) {}

  // 4
  // mat		: arma::Mat<double>
  // labels		: arma::Row<double>
  // latestPrediction	: arma::Mat<double>
  // context		: ModelContext::Context
  GradientBoostRegressor(const Mat<typename CompositeRegressor<RegressorType>::DataType>& dataset,
			 const Row<typename CompositeRegressor<RegressorType>::DataType>& labels,
			 const Row<typename CompositeRegressor<RegressorType>::DataType>& latestPrediction,
			 Context context,
			 const std::string& folderName=std::string{}) :
    CompositeRegressor<RegressorType>(dataset, labels, latestPrediction, context, folderName) {}

  
  // 5
  // mat		: arma::Mat<double>
  // labels		: arma::Row<double>
  // dataset_oos	: arma::Mat<double>
  // labels_oos		: Row<double>
  // latestPrediction	: arma::Mat<double>
  // colMask		: uvec
  // context		: ModelContext::Context
  GradientBoostRegressor(const Mat<typename CompositeRegressor<RegressorType>::DataType>& dataset,
			 const Row<typename CompositeRegressor<RegressorType>::DataType>& labels,
			 const Mat<typename CompositeRegressor<RegressorType>::DataType>& dataset_oos,
			 const Row<typename CompositeRegressor<RegressorType>::DataType>& labels_oos,
			 const Row<typename CompositeRegressor<RegressorType>::DataType>& latestPrediction,
			 const uvec& colMask,
			 Context context,
			 const std::string& folderName=std::string{}) :
    CompositeRegressor<RegressorType>(dataset, labels, dataset_oos, labels_oos, latestPrediction, colMask, context, folderName) {}

  // 6
  // mat		: arma::Mat<double>
  // labels		: arma::Row<double>
  // dataset_oos	: arma::Mat<double>
  // labels_oos		: Row<double>
  // latestPrediction	: arma::Mat<double>
  // context		: ModelContext::Context
  GradientBoostRegressor(const Mat<typename CompositeRegressor<RegressorType>::DataType>& dataset,
			 const Row<typename CompositeRegressor<RegressorType>::DataType>& labels,
			 const Mat<typename CompositeRegressor<RegressorType>::DataType>& dataset_oos,
			 const Row<typename CompositeRegressor<RegressorType>::DataType>& labels_oos,
			 const Row<typename CompositeRegressor<RegressorType>::DataType>& latestPrediction,
			 Context context,
			 const std::string& folderName=std::string{}) :
    CompositeRegressor<RegressorType>(dataset, labels, dataset_oos, labels_oos, latestPrediction, context, folderName) {}

  
};

# include "gradientboostregressor_impl.hpp"

using CompositeRegressorDTRR = CompositeRegressor<DecisionTreeRegressorRegressor>;

using GradientBoostRegressorDTRR = GradientBoostRegressor<DecisionTreeRegressorRegressor>;

CEREAL_REGISTER_TYPE(GradientBoostRegressorDTRR);

CEREAL_REGISTER_TYPE(CompositeRegressorDTRR);

CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelDTRRD, CompositeRegressorDTRR);
// CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelDTRRF, CompositeRegressorDTRR);

CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelDTRRD, GradientBoostRegressorDTRR);
// CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelDTRRF, GradientBoostRegressorDTRR);

#endif
