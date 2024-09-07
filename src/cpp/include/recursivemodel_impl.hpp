#ifndef __RECURSIVEMODEL_IMPL_HPP__
#define __RECURSIVEMODEL_IMPL_HPP__


template<typename DataType, typename ModelType>
auto 
RecursiveModel<DataType, ModelType>::_constantLeaf() -> Row<DataType> const {

  Row<DataType> r;
  r.zeros(static_cast<ModelType*>(this)->dataset_.n_cols);
  return r;
}
template<typename DataType, typename ModelType>
auto 
RecursiveModel<DataType, ModelType>::_constantLeaf(double val) -> Row<DataType> const {
  
  Row<DataType> r;
  r.ones(static_cast<ModelType*>(this)->dataset_.n_cols);
  r *= val;
  return r;
}

template<typename DataType, typename ModelType>
auto
RecursiveModel<DataType, ModelType>::_randomLeaf() -> Row<DataType> const {

  Row<DataType> r(static_cast<ModelType*>(this)->dataset_.n_cols, arma::fill::none);
  std::mt19937 rng;
  std::uniform_real_distribution<DataType> dist{-1., 1.};
  r.imbue([&](){ return dist(rng);});
  return r;
}

template<typename DataType, typename ModelType>
void
RecursiveModel<DataType, ModelType>::updateModels(std::unique_ptr<Model<DataType>>&& model,
						  Row<DataType> prediction) {
  static_cast<ModelType*>(this)->latestPrediction_ += prediction;
  model->purge();
  models_.push_back(std::move(model));
  
}

#endif
