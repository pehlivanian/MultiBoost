#ifndef __REGRESSOR_IMPL_HPP__
#define __REGRESSOR_IMPL_HPP__

template<typename DataType, typename RegressorType, typename... Args>
void
ContinuousRegressorBase<DataType, RegressorType, Args...>::Predict_(const mat& dataset, Row<DataType>& labels) {

  regressor_->Predict(dataset, labels);
}

template<typename DataType, typename RegressorType, typename... Args>
void
ContinuousRegressorBase<DataType, RegressorType, Args...>::Predict_(mat&& dataset, Row<DataType>& labels) {

  regressor_->Predict(std::move(dataset), labels);
}

template<typename DataType, typename RegressorType, typename... Args>
void
ContinuousRegressorBase<DataType, RegressorType, Args...>::setRegressor(const mat& dataset, Row<DataType>& labels, Args&&... args) {

  regressor_.reset(new RegressorType(dataset, labels, std::forward<Args>(args)...));
}

#endif
