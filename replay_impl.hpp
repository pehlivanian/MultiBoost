#ifndef __REPLAY_IMPL_HPP__
#define __REPLAY_IMPL_HPP__

template<typename DataType, typename ClassifierType>
void
Replay<DataType, ClassifierType>::read(GradientBoostClassifier<ClassifierType>& rhs,
				       std::string fileName) {
  using CerealT = GradientBoostClassifier<ClassifierType>;
  using CerealIArch = cereal::BinaryInputArchive;
  using CerealOArch = cereal::BinaryOutputArchive;
  
  loads<CerealT, CerealIArch, CerealOArch>(rhs, fileName);
}

template<typename DataType, typename ClassifierType>
void
Replay<DataType, ClassifierType>::Predict(std::string indexName,
					  const mat& dataset,
					  Row<DataType>& prediction) {
  
  std::vector<std::string> fileNames;
  readIndex(indexName, fileNames);

  using C = GradientBoostClassifier<ClassifierType>;
  std::unique_ptr<C> classifierNew = std::make_unique<C>();
  prediction = zeros<Row<DataType>>(dataset.n_cols);
  Row<DataType> predictionStep;

  bool ignoreSymmetrization = true;
  for (auto & fileName : fileNames) {
    read(*classifierNew, fileName);
    classifierNew->Predict(dataset, predictionStep, ignoreSymmetrization);
    prediction += predictionStep;
  }

}

template<typename DataType, typename ClassifierType>
void
Replay<DataType, ClassifierType>::Classify(std::string indexName,
					   const mat& dataset,
					   Row<DataType>& prediction) {
  Predict(indexName, dataset, prediction);
}

#endif
