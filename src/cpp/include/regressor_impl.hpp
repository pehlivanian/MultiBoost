#ifndef __REGRESSOR_IMPL_HPP__
#define __REGRESSOR_IMPL_HPP__

template <typename DataType, typename RegressorType, typename... Args>
void ContinuousRegressorBase<DataType, RegressorType, Args...>::init_(
    const Mat<DataType>& dataset, Row<DataType>& labels, bool useWeights, Args&&... args) {
  if (useWeights) {
    // XXX
    // No weights
    setRegressor(dataset, labels, std::forward<Args>(args)...);
    args_ = std::tuple<Args...>(args...);
  } else {
    setRegressor(dataset, labels, std::forward<Args>(args)...);
    args_ = std::tuple<Args...>(args...);
  }
}

template <typename DataType, typename RegressorType, typename... Args>
void ContinuousRegressorBase<DataType, RegressorType, Args...>::Predict_(
    const Mat<DataType>& dataset, Row<DataType>& labels) {
  regressor_->Predict(dataset, labels);
}

template <typename DataType, typename RegressorType, typename... Args>
void ContinuousRegressorBase<DataType, RegressorType, Args...>::Predict_(
    Mat<DataType>&& dataset, Row<DataType>& labels) {
  regressor_->Predict(std::move(dataset), labels);
}

template <typename DataType, typename RegressorType, typename... Args>
void ContinuousRegressorBase<DataType, RegressorType, Args...>::setRegressor(
    const Mat<DataType>& dataset, Row<DataType>& labels, Args&&... args) {
  regressor_.reset(new RegressorType(dataset, labels, std::forward<Args>(args)...));
}

#endif
