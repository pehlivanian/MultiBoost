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
RecursiveModel<DataType, ModelType>::_randomLeaf() -> Row<DataType> const {\
  
  Row<DataType> r(static_cast<ModelType*>(this)->dataset_.n_cols);
  std::mt19937 rng;
  std::uniform_real_distribution<DataType> dist{-1., 1.};
  r.imbue([&](){ return dist(rng); });
  return r;

}

/*
template<typename DataType, typename ModelType>
void
RecursiveModel<DataType, ModelType>::fit() {
  int numSteps = static_cast<ModelType*>(this)->steps_;

  for (int stepNum=1; stepNum<=numSteps; ++stepNum) {
    static_cast<ModelType*>(this)->fit_step(stepNum);

    if (serializeModel_) {
      static_cast<ModelType*>(this)->commit();
    }
    if (!quietRun_) {
      static_cast<ModelType*>(this)->printStats(stepNum);
    }
  }

  // Serialize residual
  if (serializeModel_)
    commit();

  // print final stats
  if (!quietRun_) {
    printStats(numSteps);
  }
}
*/

#endif
