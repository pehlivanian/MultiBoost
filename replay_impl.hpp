#ifndef __REPLAY_IMPL_HPP__
#define __REPLAY_IMPL_HPP__

template<typename DataType, typename ClassifierType>
void
Replay<DataType, ClassifierType>::desymmetrize(Row<DataType>& prediction, double a, double b) {

  prediction = (sign(prediction) - b)/ a;
}

template<typename DataType, typename ClassifierType>
void
Replay<DataType, ClassifierType>::ClassifyStep(std::string classifierFileName,
					      std::string datasetFileName,
					      Row<double>& prediction,
					      bool deSymmetrize) {
  bool ignoreSymmetrization = true;
  std::pair<double, double> ab;

  using C = GradientBoostClassifier<ClassifierType>;
  std::unique_ptr<C> classifier = std::make_unique<C>();
  read(*classifier, classifierFileName);

  mat dataset;
  read(dataset, datasetFileName);

  classifier->Predict(dataset, prediction, ignoreSymmetrization);

  ab = classifier->getAB();

  if (deSymmetrize)
    desymmetrize(prediction, ab.first, ab.second);

}

template<typename DataType, typename RegressorType>
void
Replay<DataType, RegressorType>::PredictStep(std::string regressorFileName,
					     std::string datasetFileName,
					     Row<double>& prediction) {
  
  using R = GradientBoostRegressor<RegressorType>;
  std::unique_ptr<R> regressor = std::make_unique<R>();
  read(*regressor, regressorFileName);

  mat dataset;
  read(dataset, datasetFileName);

  regressor->Predict(dataset, prediction);
}

template<typename DataType, typename ClassifierType>
void
Replay<DataType, ClassifierType>::ClassifyStep(std::string classifierFileName,
					       std::string datasetFileName,
					       std::string outFileName,
					       bool deSymmetrize) {

  Row<DataType> prediction;
  ClassifyStep(classifierFileName, datasetFileName, prediction, deSymmetrize);

  writePrediction(prediction, outFileName);
}

template<typename DataType, typename RegressorType>
void
Replay<DataType, RegressorType>::PredictStep(std::string regressorFileName,
					     std::string datasetFileName,
					     std::string outFileName) {
  Row<DataType> prediction;
  PredictStep(regressorFileName, datasetFileName, prediction);

  writePrediction(prediction, outFileName);
}

template<typename DataType, typename ClassifierType>
void
Replay<DataType, ClassifierType>::ClassifyStepwise(std::string indexName,
						   Row<DataType>& prediction,
						   Row<DataType>& labels_oos,
						   bool deSymmetrize, 
						   bool distribute) {
  
  std::vector<std::string> fileNames, predictionFileNames;
  readIndex(indexName, fileNames);

  int n_rows, n_cols;
  bool ignoreSymmetrization = true;
  mat dataset, dataset_oos;
  Row<DataType> labels, predictionStep;
  int classifierNum = 0;
  std::string datasetFileName, datasetOOSFileName;
  std::string labelsFileName, labelsOOSFileName;
  std::string classifierFileName;
  std::string outFilePref = "classification_STEP_";

  // First pass - get dataset, labels
  for (const auto &fileName : fileNames) {
    auto tokens = strSplit(fileName, '_');
    auto fileName_short = strJoin(tokens, '_', 1);
    if (tokens[0] == "DIS") { // Dataset In Sample
      read(dataset, datasetFileName = fileName_short);
    }
    else if (tokens[0] == "DOOS") { // Dataset Out Of Sample
      read(dataset_oos, datasetOOSFileName = fileName_short);
      n_rows = dataset_oos.n_rows;
      n_cols = dataset_oos.n_cols;
    }
    else if (tokens[0] == "LIS") { // Labels In Sample
      read(labels, labelsFileName = fileName_short);
    }
    else if (tokens[0] == "LOOS") { // Labels Out Of Sample
      read(labels_oos, labelsOOSFileName = fileName_short);
    }
  }

  distribute = false; // before enabling, make sure we have the memory

  // Next pass - generate prediction
  if (distribute) {

    ThreadsafeQueue<Row<double>> results_queue;
    std::vector<ThreadPool::TaskFuture<int>> futures;
    
    for (auto &fileName : fileNames) {
      auto tokens = strSplit(fileName, '_');
      if (tokens[0] == "CLS") {
	classifierFileName = strJoin(tokens, '_', 1);
	
	auto task = [&results_queue, classifierFileName, datasetOOSFileName](Row<double>& prediction){
	  ClassifyStep(classifierFileName,
		       datasetOOSFileName,
		       prediction,
		       false);
	  results_queue.push(prediction);
	  return 0;
	};

	Row<double> prediction;
	futures.push_back(DefaultThreadPool::submitJob_n<3>(task, std::ref(prediction)));
	
	classifierNum++;
      }
    }

    prediction = zeros<Row<DataType>>(n_cols);
    
    for (auto &item : futures) {
      auto r = item.get();
    }
    
    while (!results_queue.empty()) {
      Row<double> predictionStep;
      bool valid = results_queue.waitPop(predictionStep);
      prediction += predictionStep;
    }
    
  } else {

    for (auto &fileName : fileNames) {
      auto tokens = strSplit(fileName, '_');
      if (tokens[0] == "CLS") {
	classifierFileName = strJoin(tokens, '_', 1);
	std::string outFile = outFilePref + std::to_string(classifierNum) + ".prd";
	// Call ClassifyStep at this point, but 
	// wrapped in another process we can launch as a 
	// true subprocess
	
	ipstream pipe_stream;
	// child c("gcc --version", std_out > pipe_stream);
	std::string cmd = "./build/replay_classify_stepwise";

	cmd += " --datasetFileName "	+ datasetOOSFileName;
	cmd += " --classifierFileName " + classifierFileName;
	cmd += " --outFileName "	+ outFile;
	
	child c(cmd, std_out > pipe_stream);
      
	std::string line;
	while (pipe_stream && std::getline(pipe_stream, line) && !line.empty())
	  std::cerr << line << std::endl;
	
	c.wait();

	predictionFileNames.push_back(outFile);
	classifierNum++;
      }
    }

    prediction = zeros<Row<DataType>>(n_cols);
  
    for (auto &fileName : predictionFileNames) {
      read(predictionStep, fileName);
      prediction+= predictionStep;
    }    

  }
  
  if (deSymmetrize) {
    // Assume a_ = 2, b_ = -1
    desymmetrize(prediction, 2., -1.);
  }
  
}


template<typename DataType, typename RegressorType>
void
Replay<DataType, RegressorType>::PredictStepwise(std::string indexName,
						 Row<DataType>& prediction,
						 Row<DataType>& labels_oos,
						 bool distribute) {
  
  std::vector<std::string> fileNames, predictionFileNames;
  readIndex(indexName, fileNames);

  int n_rows, n_cols;
  mat dataset, dataset_oos;
  Row<DataType> labels, predictionStep;
  int regressorNum = 0;
  std::string datasetFileName, datasetOOSFileName;
  std::string labelsFileName, labelsOOSFileName;
  std::string regressorFileName;
  std::string outFilePref = "prediction_STEP_";

  // First pass - get dataset, labels
  for (const auto &fileName : fileNames) {
    auto tokens = strSplit(fileName, '_');
    auto fileName_short = strJoin(tokens, '_', 1);
    if (tokens[0] == "DIS") { // Dataset In Sample
      read(dataset, datasetFileName = fileName_short);
    }
    else if (tokens[0] == "DOOS") { // Dataset Out Of Sample
      read(dataset_oos, datasetOOSFileName = fileName_short);
      n_rows = dataset_oos.n_rows;
      n_cols = dataset_oos.n_cols;
    }
    else if (tokens[0] == "LIS") { // Labels In Sample
      read(labels, labelsFileName = fileName_short);
    }
    else if (tokens[0] == "LOOS") { // Labels Out Of Sample
      read(labels_oos, labelsOOSFileName = fileName_short);
    }
  }

  distribute = false; // before enabling, make sure we have the memory

  // Next pass - generate prediction
  if (distribute) {

    ThreadsafeQueue<Row<double>> results_queue;
    std::vector<ThreadPool::TaskFuture<int>> futures;
    
    for (auto &fileName : fileNames) {
      auto tokens = strSplit(fileName, '_');
      if (tokens[0] == "REG") {
	regressorFileName = strJoin(tokens, '_', 1);
	
	auto task = [&results_queue, regressorFileName, datasetOOSFileName](Row<double>& prediction){
	  PredictStep(regressorFileName,
		      datasetOOSFileName,
		      prediction);
	  results_queue.push(prediction);
	  return 0;
	};

	Row<double> prediction;
	futures.push_back(DefaultThreadPool::submitJob_n<3>(task, std::ref(prediction)));
	
	regressorNum++;
      }
    }

    prediction = zeros<Row<DataType>>(n_cols);
    
    for (auto &item : futures) {
      auto r = item.get();
    }
    
    while (!results_queue.empty()) {
      Row<double> predictionStep;
      bool valid = results_queue.waitPop(predictionStep);
      prediction += predictionStep;
    }
    
  } else {

    for (auto &fileName : fileNames) {
      auto tokens = strSplit(fileName, '_');
      if (tokens[0] == "REG") {
	regressorFileName = strJoin(tokens, '_', 1);
	std::string outFile = outFilePref + std::to_string(regressorNum) + ".prd";
	// Call PredictStep at this point, but 
	// wrapped in another process we can launch as a 
	// true subprocess
	
	ipstream pipe_stream;
	// child c("gcc --version", std_out > pipe_stream);
	std::string cmd = "./build/incremental_predict";

	cmd += " --datasetFileName "	+ datasetOOSFileName;
	cmd += " --regressorFileName "  + regressorFileName;
	cmd += " --outFileName "	+ outFile;
	
	child c(cmd, std_out > pipe_stream);
      
	std::string line;
	while (pipe_stream && std::getline(pipe_stream, line) && !line.empty())
	  std::cerr << line << std::endl;
	
	c.wait();

	predictionFileNames.push_back(outFile);
	regressorNum++;
      }
    }

    prediction = zeros<Row<DataType>>(n_cols);
  
    for (auto &fileName : predictionFileNames) {
      read(predictionStep, fileName);
      prediction+= predictionStep;
    }    

  }
  
}

template<typename DataType, typename ClassifierType>
void
Replay<DataType, ClassifierType>::Classify(std::string indexName, Row<DataType>& prediction) {

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

template<typename DataType, typename RegressorType>
void
Replay<DataType, RegressorType>::Predict(std::string indexName,
					 const mat& dataset,
					 Row<DataType>& prediction) {
  std::vector<std::string> fileNames;
  readIndex(indexName, fileNames);

  using R = GradientBoostRegressor<RegressorType>;
  std::unique_ptr<R> regressorNew = std::make_unique<R>();

  prediction = zeros<Row<DataType>>(dataset.n_cols);
  Row<DataType> predictionStep;

  for (auto & fileName : fileNames) {
    auto tokens = strSplit(fileName, '_');
    if (tokens[0] == "REG") {
      fileName = strJoin(tokens, '_', 1);
      read(*regressorNew, fileName);
      regressorNew->Predict(dataset, predictionStep);
      prediction += predictionStep;
    }
  }
}

template<typename DataType, typename RegressorType>
void
Replay<DataType, RegressorType>::Predict(std::string indexName, Row<DataType>& prediction) {

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

#endif
