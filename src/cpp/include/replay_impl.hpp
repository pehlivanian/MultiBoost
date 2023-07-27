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
					       bool deSymmetrize,
					       boost::filesystem::path folderName) {
  bool ignoreSymmetrization = true;
  std::pair<double, double> ab;

  using C = GradientBoostClassifier<ClassifierType>;
  std::unique_ptr<C> classifier = std::make_unique<C>();
  read(*classifier, classifierFileName, folderName);

  mat dataset;
  read(dataset, datasetFileName, folderName);

  classifier->Predict(dataset, prediction, ignoreSymmetrization);

  ab = classifier->getAB();

  if (deSymmetrize)
    desymmetrize(prediction, ab.first, ab.second);

}

template<typename DataType, typename RegressorType>
void
Replay<DataType, RegressorType>::PredictStep(std::string regressorFileName,
					     std::string datasetFileName,
					     Row<double>& prediction,
					     boost::filesystem::path folderName) {
  
  using R = GradientBoostRegressor<RegressorType>;
  std::unique_ptr<R> regressor = std::make_unique<R>();
  read(*regressor, regressorFileName, folderName);

  mat dataset;
  read(dataset, datasetFileName, folderName);

  regressor->Predict(dataset, prediction);
}

template<typename DataType, typename ClassifierType>
void
Replay<DataType, ClassifierType>::ClassifyStep(std::string classifierFileName,
					       std::string datasetFileName,
					       std::string outFileName,
					       bool deSymmetrize,
					       boost::filesystem::path folderName) {

  Row<DataType> prediction;
  ClassifyStep(classifierFileName, datasetFileName, prediction, deSymmetrize, folderName);

  writePrediction(prediction, outFileName, folderName);
}

template<typename DataType, typename RegressorType>
void
Replay<DataType, RegressorType>::PredictStep(std::string regressorFileName,
					     std::string datasetFileName,
					     std::string outFileName,
					     boost::filesystem::path folderName) {
  Row<DataType> prediction;
  PredictStep(regressorFileName, datasetFileName, prediction, folderName);

  writePrediction(prediction, outFileName, folderName);
}

template<typename DataType, typename ClassifierType>
typename Replay<DataType, ClassifierType>::optCV
Replay<DataType, ClassifierType>::ClassifyStepwise(std::string indexName,
						   Row<DataType>& prediction,
						   Row<DataType>& labels_oos,
						   bool deSymmetrize, 
						   bool distribute,
						   bool include_loss,
						   boost::filesystem::path folderName) {
  
  std::vector<std::string> fileNames, predictionFileNames, predictionFileNamesIS;
  Row<DataType> prediction_is;
  readIndex(indexName, fileNames, folderName);

  int n_rows, n_cols;
  int n_rows_is, n_cols_is;
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
      read(dataset, datasetFileName = fileName_short, folderName);
      n_rows_is = dataset.n_rows;
      n_cols_is = dataset.n_cols;
    }
    else if (tokens[0] == "DOOS") { // Dataset Out Of Sample
      read(dataset_oos, datasetOOSFileName = fileName_short, folderName);
      n_rows = dataset_oos.n_rows;
      n_cols = dataset_oos.n_cols;
    }
    else if (tokens[0] == "LIS") { // Labels In Sample
      read(labels, labelsFileName = fileName_short, folderName);
    }
    else if (tokens[0] == "LOOS") { // Labels Out Of Sample
      read(labels_oos, labelsOOSFileName = fileName_short, folderName);
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
	
	auto task = [&results_queue, classifierFileName, datasetOOSFileName, folderName](Row<double>& prediction){
	  ClassifyStep(classifierFileName,
		       datasetOOSFileName,
		       prediction,
		       false,
		       folderName);
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
      item.get();
    }
    
    while (!results_queue.empty()) {
      Row<double> predictionStep;
      results_queue.waitPop(predictionStep);
      prediction += predictionStep;
    }
    
  } else {

    for (auto &fileName : fileNames) {
      auto tokens = strSplit(fileName, '_');
      if (tokens[0] == "CLS") {
	classifierFileName = strJoin(tokens, '_', 1);
	std::string outFile = outFilePref + std::to_string(classifierNum) + ".prd";
	std::string outFile_is = outFilePref + std::to_string(classifierNum) + ".is.prd";
	// Call ClassifyStep at this point, but 
	// wrapped in another process we can launch as a 
	// true subprocess
	
	ipstream pipe_stream;
	// child c("gcc --version", std_out > pipe_stream);
	std::string cmd = "/home/charles/src/C++/sandbox/Inductive-Boost/build/replay_classify_stepwise";

	cmd += " --datasetFileName "	+ datasetOOSFileName;
	cmd += " --classifierFileName " + classifierFileName;
	cmd += " --outFileName "	+ outFile;
	cmd += " --folderName "		+ folderName.string();
	
	child c(cmd, std_out > pipe_stream);
      
	std::string line;
	while (pipe_stream && std::getline(pipe_stream, line) && !line.empty())
	  std::cerr << line << std::endl;
	
	c.wait();

	predictionFileNames.push_back(outFile);

	ipstream pipe_stream_is;
	cmd = "/home/charles/src/C++/sandbox/Inductive-Boost/build/replay_classify_stepwise";

	cmd += " --datasetFileName "	+ datasetFileName;
	cmd += " --classifierFileName " + classifierFileName;
	cmd += " --outFileName "	+ outFile_is;
	cmd += " --folderName "		+ folderName.string();

	child c_is(cmd, std_out > pipe_stream_is);

	while (pipe_stream_is && std::getline(pipe_stream_is, line) && !line.empty())
	  std::cerr << line << std::endl;

	c.wait();

	predictionFileNamesIS.push_back(outFile_is);

	classifierNum++;
      }
    }

    prediction = zeros<Row<DataType>>(n_cols);
  
    for (auto &fileName : predictionFileNames) {
      read(predictionStep, fileName, folderName);
      prediction+= predictionStep;
    }    

    prediction_is = zeros<Row<DataType>>(n_cols_is);

    for (auto &fileName : predictionFileNamesIS) {
      Row<DataType> predictionStep_is;
      read(predictionStep_is, fileName, folderName);
      prediction_is += predictionStep_is;
    }

  }

  /*
    for (int i=0; i<15; ++i) {
    std::cout << "OOS: (raw y,y_hat): ("
    << prediction[i] << ", "
    << labels_oos[i] << ")" << std::endl;
    }
    
    for (int i=0; i<15; ++i) {
    std::cout << "IS: (raw labels, labels_hat): ("
    << prediction_is[i] << ", "
    << labels[i] << ")" << std::endl;
    }
  */
  
  if (deSymmetrize) {
    using C = GradientBoostClassifier<ClassifierType>;
    std::unique_ptr<C> c_archive = std::make_unique<C>();
    read(*c_archive, classifierFileName, folderName);

    auto ab = c_archive->getAB();

    desymmetrize(prediction, ab.first, ab.second);
    desymmetrize(prediction_is, ab.first, ab.second);
  }
   
  /*
    for (int i=0; i<10; ++i) {
    std::cout << "OOS: (sym y,y_hat): ("
    << prediction[i] << ", "
    << labels_oos[i] << ")" << std::endl;
    }
    std::cout << std::endl;
    
    for (int i=0; i<10; ++i) {
    std::cout << "IS: (labels, labels_hat): ("
    << prediction_is[i] << ", "
    << labels[i] << ")" << std::endl;
    }
  */

  if (include_loss) {

    // Pile on the metrics

    //
    // 1. error
    // 
    double error_OOS = err(prediction, labels_oos);
    double error_IS = err(prediction_is, labels);

    // Now symmetrize labels, labels_oos for remaining metrics
    
    using C = GradientBoostClassifier<ClassifierType>;
    std::unique_ptr<C> c_archive = std::make_unique<C>();
    read(*c_archive, classifierFileName, folderName);

    
    c_archive->symmetrizeLabels(labels);
    c_archive->symmetrizeLabels(labels_oos);
    c_archive->symmetrizeLabels(prediction_is);
    c_archive->symmetrizeLabels(prediction);
    
    Row<int> labels_is_i = conv_to<Row<int>>::from(labels);
    Row<int> labels_oos_i = conv_to<Row<int>>::from(labels_oos);
    Row<int> prediction_is_i = conv_to<Row<int>>::from(prediction_is);
    Row<int> prediction_oos_i = conv_to<Row<int>>::from(prediction);

    auto [precision_IS, recall_IS, F1_IS] = precision(labels_is_i, prediction_is_i);
    auto [precision_OOS, recall_OOS, F1_OOS] = precision(labels_oos_i, prediction_oos_i);    
    
    // 
    // 2. imbalance
    //
    double imbalance_OOS = imbalance(labels_oos_i);
    double imbalance_IS = imbalance(labels_is_i);

    return std::make_tuple(error_OOS, 
			   precision_OOS, 
			   recall_OOS, 
			   F1_OOS, 
			   imbalance_OOS,
			   error_IS, 
			   precision_IS, 
			   recall_IS, 
			   F1_IS, 
			   imbalance_IS);
  
  } else {
    return std::make_tuple(std::nullopt,
			   std::nullopt,
			   std::nullopt,
			   std::nullopt,
			   std::nullopt,
			   std::nullopt,
			   std::nullopt,
			   std::nullopt,
			   std::nullopt,
			   std::nullopt);
  }

}


template<typename DataType, typename RegressorType>
typename Replay<DataType, RegressorType>::optRV
Replay<DataType, RegressorType>::PredictStepwise(std::string indexName,
						 Row<DataType>& prediction,
						 Row<DataType>& labels_oos,
						 bool distribute,
						 bool include_loss,
						 boost::filesystem::path folderName) {
  
  std::vector<std::string> fileNames, predictionFileNames, predictionFileNamesIS;
  Row<DataType> prediction_is;
  readIndex(indexName, fileNames, folderName);



  int n_rows, n_cols;
  int n_rows_is, n_cols_is;
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
      read(dataset, datasetFileName = fileName_short, folderName);
      n_rows_is = dataset.n_rows;
      n_cols_is = dataset.n_cols;
    }
    else if (tokens[0] == "DOOS") { // Dataset Out Of Sample
      read(dataset_oos, datasetOOSFileName = fileName_short, folderName);
      n_rows = dataset_oos.n_rows;
      n_cols = dataset_oos.n_cols;
    }
    else if (tokens[0] == "LIS") { // Labels In Sample
      read(labels, labelsFileName = fileName_short, folderName);
    }
    else if (tokens[0] == "LOOS") { // Labels Out Of Sample
      read(labels_oos, labelsOOSFileName = fileName_short, folderName);
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
	
	auto task = [&results_queue, regressorFileName, datasetOOSFileName, folderName](Row<double>& prediction){
	  PredictStep(regressorFileName,
		      datasetOOSFileName,
		      prediction,
		      folderName);
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
      item.get();
    }
    
    while (!results_queue.empty()) {
      Row<double> predictionStep;
      results_queue.waitPop(predictionStep);
      prediction += predictionStep;
    }
    
  } else {

    for (auto &fileName : fileNames) {
      auto tokens = strSplit(fileName, '_');
      if (tokens[0] == "REG") {
	regressorFileName = strJoin(tokens, '_', 1);
	std::string outFile = outFilePref + std::to_string(regressorNum) + ".prd";
	std::string outFile_is = outFilePref + std::to_string(regressorNum) + ".is.prd";
	// Call PredictStep at this point, but 
	// wrapped in another process we can launch as a 
	// true subprocess
	
	ipstream pipe_stream;
	// child c("gcc --version", std_out > pipe_stream);
	std::string cmd = "/home/charles/src/C++/sandbox/Inductive-Boost/build/replay_predict_stepwise";

	cmd += " --datasetFileName "	+ datasetOOSFileName;
	cmd += " --regressorFileName "  + regressorFileName;
	cmd += " --outFileName "	+ outFile;
	cmd += " --folderName "		+ folderName.string();
	
	child c(cmd, std_out > pipe_stream);
      
	std::string line;
	while (pipe_stream && std::getline(pipe_stream, line) && !line.empty())
	  std::cerr << line << std::endl;
	
	c.wait();

	predictionFileNames.push_back(outFile);

	ipstream pipe_stream_is;
	cmd = "/home/charles/src/C++/sandbox/Inductive-Boost/build/replay_predict_stepwise";
	
	cmd += " --datasetFileName "	+ datasetFileName;
	cmd += " --regressorFileName "  + regressorFileName;
	cmd += " --outFileName "	+ outFile_is;
	cmd += " --folderName "		+ folderName.string();

	child c_is(cmd, std_out > pipe_stream_is);

	while (pipe_stream_is && std::getline(pipe_stream_is, line) && !line.empty())
	  std::cerr << line << std::endl;

	c.wait();

	predictionFileNamesIS.push_back(outFile_is);
	
	regressorNum++;
      }
    }

    prediction = zeros<Row<DataType>>(n_cols);
  
    for (auto &fileName : predictionFileNames) {
      read(predictionStep, fileName, folderName);
      prediction+= predictionStep;
    }    

    prediction_is = zeros<Row<DataType>>(n_cols_is);

    for (auto &fileName : predictionFileNamesIS) {
      Row<double> predictionStep_is;
      read(predictionStep_is, fileName, folderName);
      prediction_is += predictionStep_is;
    }

  }

  if (include_loss) {

    /*
      for (int i=0; i<10; ++i) {
      std::cout << "OOS: (y, y_hat): (" 
      << prediction[i] << ", "
      << labels_oos[i] << ")" << std::endl;
      }
      std::cout << std::endl;
      
      for (int i=0; i<10; ++i) {
      std::cout << "IS: (labels, labels_hat): (" 
      << prediction_is[i] << ", "
      << labels[i] << ")" << std::endl;
      }
    */

    // Pile on the metrics here
    //
    // 1. OOS r squared
    // 
    auto mn = mean(labels_oos);
    auto num = sum(pow((labels_oos - prediction), 2));
    auto den = sum(pow((labels_oos - mn), 2));
    double r_squared_OOS = 1. - (num/den);


    //
    // 2 IS r squared
    //
    mn = mean(labels);
    num = sum(pow((labels - prediction_is), 2));
    den = sum(pow((labels - mn), 2));
    double r_squared_IS = 1. - (num/den);

    // Rank-based statistics, see
    // @article{article,
    // author = {Rosset, Saharon and Perlich, Claudia and Zadrozny, Bianca},
    // year = {2007},
    // month = {08},
    // pages = {331-353},
    // title = {Ranking-based evaluation of regression models},
    // volume = {12},
    // journal = {Knowl. Inf. Syst.},
    // doi = {10.1109/ICDM.2005.126}
    // }
    Row<DataType> labels_oos_sorted = sort(labels_oos);
    Row<DataType> prediction_sorted = sort(prediction);
    int aa_ = 55;

    //
    // 2. OOS loss
    // 
    using R = GradientBoostRegressor<RegressorType>;
    std::unique_ptr<R> regressor = std::make_unique<R>();

    read(*regressor, regressorFileName, folderName);
    auto lossFn = lossMap<DataType>[regressor->getLoss()];
    
    double r_OOS = std::sqrt(lossFn->loss(prediction, labels_oos));

    // 
    // 3 IS loss
    // 
    double r_IS = std::sqrt(lossFn->loss(prediction_is, labels));
    
    return std::make_tuple(r_OOS, r_squared_OOS, r_IS, r_squared_IS);
    
    
  } else {
    return std::make_tuple(std::nullopt, 
			   std::nullopt, 
			   std::nullopt, 
			   std::nullopt);
  }
  
}

template<typename DataType, typename ClassifierType>
void
Replay<DataType, ClassifierType>::Classify(std::string indexName, 
					   Row<DataType>& prediction,
					   boost::filesystem::path fldr) {

  Row<DataType> predictionNew;
  std::vector<std::string> fileNames;
  readIndex(indexName, fileNames, fldr);

  for  (auto &fileName : fileNames) {
    auto tokens = strSplit(fileName, '_');
    if (tokens[0] == "PRED") {
      fileName = strJoin(tokens, '_', 1);
      read(predictionNew, fileName, fldr);
      prediction = predictionNew;
    }
  }

}


template<typename DataType, typename ClassifierType>
void
Replay<DataType, ClassifierType>::Classify(std::string indexName,
					   const mat& dataset,
					   Row<DataType>& prediction,
					   boost::filesystem::path fldr,
					   bool deSymmetrize) {

  std::vector<std::string> fileNames;
  readIndex(indexName, fileNames, fldr);

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
      read(*classifierNew, fileName, fldr);
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
					 Row<DataType>& prediction,
					 boost::filesystem::path fldr) {
  std::vector<std::string> fileNames;
  readIndex(indexName, fileNames, fldr);

  using R = GradientBoostRegressor<RegressorType>;
  std::unique_ptr<R> regressorNew = std::make_unique<R>();

  prediction = zeros<Row<DataType>>(dataset.n_cols);
  Row<DataType> predictionStep;

  for (auto & fileName : fileNames) {
    auto tokens = strSplit(fileName, '_');
    if (tokens[0] == "REG") {
      fileName = strJoin(tokens, '_', 1);
      read(*regressorNew, fileName, fldr);
      regressorNew->Predict(dataset, predictionStep);
      prediction += predictionStep;
    }
  }
}

template<typename DataType, typename RegressorType>
void
Replay<DataType, RegressorType>::Predict(std::string indexName, 
					 Row<DataType>& prediction,
					 boost::filesystem::path fldr) {

  Row<DataType> predictionNew;
  std::vector<std::string> fileNames;
  readIndex(indexName, fileNames, fldr);

  for  (auto &fileName : fileNames) {
    auto tokens = strSplit(fileName, '_');
    if (tokens[0] == "PRED") {
      fileName = strJoin(tokens, '_', 1);
      read(predictionNew, fileName, fldr);
      prediction = predictionNew;
    }
  }
  
}

#endif
