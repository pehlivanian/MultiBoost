#ifndef __REPLAY_IMPL_HPP__
#define __REPLAY_IMPL_HPP__

template<typename DataType, typename ClassifierType>
template<typename T>
void
Replay<DataType, ClassifierType>::read(T& rhs,
				       std::string fileName) {
  using CerealT = T;
  using CerealIArch = cereal::BinaryInputArchive;
  using CerealOArch = cereal::BinaryOutputArchive;
  
  loads<CerealT, CerealIArch, CerealOArch>(rhs, fileName);
}

template<typename DataType, typename ClassifierType>
void
Replay<DataType, ClassifierType>::desymmetrize(Row<DataType>& prediction, double a, double b) {

  prediction = (sign(prediction) - b)/ a;
}

template<typename DataType, typename ClassifierType>
void
Replay<DataType, ClassifierType>::PredictStepwise(std::string indexName,
						  const mat& dataset,
						  Row<DataType>& prediction,
						  bool deSymmetrize) {
  std::vector<std::string> fileNames;
  readIndex(indexName, fileNames);

  using C = GradientBoostClassifier<ClassifierType>;
  std::unique_ptr<C> classifierNew = std::make_unique<C>();
  prediction = zeros<Row<DataType>>(dataset.n_cols);

  std::pair<double, double> ab;
  bool ignoreSymmetrization = true;
  for (auto & fileName : fileNames) {
    auto tokens = strSplit(fileName, '_');
    if (tokens[0] == "CLS") {
      fileName = strJoin(tokens, '_', 1);
      read(*classifierNew, fileName);
      classifierNew->Predict(dataset, prediction, ignoreSymmetrization);
      
      
    }
  }
  
}

template<typename DataType, typename ClassifierType>
void
Replay<DataType, ClassifierType>::Predict(std::string indexName,
					  const mat& dataset,
					  Row<DataType>& prediction,
					  bool deSymmetrize) {

  std::vector<std::string> fileNames;
  readIndex(indexName, fileNames);

  using C = GradientBoostClassifier<ClassifierType>;
  std::unique_ptr<C> classifierNew = std::make_unique<C>();
  prediction = zeros<Row<DataType>>(dataset.n_cols);
  Row<DataType> predictionStep;

  std::pair<double, double> ab;

  bool ignoreSymmetrization = true;
  for (auto & fileName : fileNames) {
    auto tokens = strSplit(fileName, '_');
    if (tokens[0] == "CLS") {
      fileName = strJoin(tokens, '_', 1);
      read(*classifierNew, fileName);
      classifierNew->Predict(dataset, predictionStep, ignoreSymmetrization);
      prediction += predictionStep;

      ab = classifierNew->getAB();
    }
  }

  if (deSymmetrize)
    desymmetrize(prediction, ab.first, ab.second);
}

template<typename DataType, typename ClassifierType>
void
Replay<DataType, ClassifierType>::Predict(std::string indexName, Row<DataType>& prediction) {

  Row<DataType> predictionNew;
  std::vector<std::string> fileNames;
  readIndex(indexName, fileNames);

  for  (auto &fileName : fileNames) {
    auto tokens = strSplit(fileName, '_');
    if (tokens[0] == "PRED") {
      fileName = strJoin(tokens, '_', 1);
      read(predictionNew, fileName);
      prediction = predictionNew;
    }
  }

}

template<typename DataType, typename ClassifierType>
void
Replay<DataType, ClassifierType>::Classify(std::string indexName,
					   const mat& dataset,
					   Row<DataType>& prediction,
					   bool deSymmetrize) {
  Predict(indexName, dataset, prediction, deSymmetrize);
}

template<typename DataType, typename ClassifierType>
void
Replay<DataType, ClassifierType>::Classify(std::string indexName,
					   Row<DataType>& prediction) {
  Predict(indexName, prediction);
}

template<typename DataType, typename ClassifierType>
void
Replay<DataType, ClassifierType>::readPrediction(std::string indexName, Row<DataType>& prediction) {

  Row<DataType> predictionNew;  
  std::vector<std::string> fileNames;
  readIndex(indexName, fileNames);

  for (auto &fileName : fileNames) {
    auto tokens = strSplit(fileName, '_');
    if (tokens[0] == "PRED") {
      fileName = strJoin(tokens, '_', 1);
      read(predictionNew, fileName);
      prediction = predictionNew;
    }
  }

}



#endif
