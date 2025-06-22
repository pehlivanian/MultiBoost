#ifndef __GRADIENTBOOSTREGRESSOR_HPP__
#define __GRADIENTBOOSTREGRESSOR_HPP__

#include <cereal/access.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/archives/portable_binary.hpp>
#include <cereal/archives/xml.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/map.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/vector.hpp>
#include <mlpack/core.hpp>

#include "compositeregressor.hpp"
#include "model_traits.hpp"
#include "regressor.hpp"
#include "regressors.hpp"

using namespace arma;

template <typename RegressorType>
class GradientBoostRegressor : public CompositeRegressor<RegressorType> {
public:
  GradientBoostRegressor() = default;

  // Common constructor patterns - explicit to avoid template overhead
  GradientBoostRegressor(
      const Mat<typename CompositeRegressor<RegressorType>::DataType>& dataset,
      const Row<typename CompositeRegressor<RegressorType>::DataType>& labels,
      Context context,
      const std::string& folderName = std::string{})
      : CompositeRegressor<RegressorType>(dataset, labels, context, folderName) {}

  GradientBoostRegressor(
      const Mat<typename CompositeRegressor<RegressorType>::DataType>& dataset,
      const Row<typename CompositeRegressor<RegressorType>::DataType>& labels,
      const Mat<typename CompositeRegressor<RegressorType>::DataType>& dataset_oos,
      const Row<typename CompositeRegressor<RegressorType>::DataType>& labels_oos,
      Context context,
      const std::string& folderName = std::string{})
      : CompositeRegressor<RegressorType>(
            dataset, labels, dataset_oos, labels_oos, context, folderName) {}

  // Template fallback for other patterns
  template <typename... Args>
  explicit GradientBoostRegressor(Args&&... args)
      : CompositeRegressor<RegressorType>(std::forward<Args>(args)...) {}
};

#include "gradientboostregressor_impl.hpp"

using CompositeRegressorDTRR = CompositeRegressor<DecisionTreeRegressorRegressor>;

using GradientBoostRegressorDTRR = GradientBoostRegressor<DecisionTreeRegressorRegressor>;

CEREAL_REGISTER_TYPE(GradientBoostRegressorDTRR);

CEREAL_REGISTER_TYPE(CompositeRegressorDTRR);

CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelD, CompositeRegressorDTRR);

CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelD, GradientBoostRegressorDTRR);

#endif
