#include "pmlb_driver.hpp"

using namespace arma;
using namespace mlpack;
using namespace mlpack::tree;
using namespace mlpack::data;
using namespace mlpack::util;
using namespace std;

auto main(int argc, char **argv) -> int {

  /* 
     Old way
     
     std::string Xpath = "/home/charles/Data/test_X.csv";
     std::string ypath = "/home/charles/Data/test_y.csv";
     auto df = DataSet<float>(Xpath, ypath, false);
     std::size_t ind1 = 4, ind2 = 2;
     
     auto shape = df.shape();
     
     std::cout << "COMPLETE." << std::endl;
     std::cout << "SIZE: (" << shape.first << ", " 
     << shape.second << ")" << std::endl;
     std::cout << "df[" << ind1 << "][" << ind2
     << "]: " << df[ind1][ind2] << std::endl;
     
     auto splitter = SplitProcessor<float>(.8);
  
     df.accept(splitter);
  */

  /*
    The mlpack way

  */
  mat dataset;
  Row<size_t> labels, predictions;

  if (!data::Load("/home/charles/Data/test_X.csv", dataset))
    throw std::runtime_error("Could not load test_X.csv");
  if (!data::Load("/home/charles/Data/test_y.csv", labels))
    throw std::runtime_error("Could not load test_y.csv");

  DecisionTree<GiniGain, BestBinaryNumericSplit, AllCategoricalSplit, AllDimensionSelect, true> r(dataset,
												  labels,
												  7, // number of classes
												  10, // number of trees
												  3); // minimum leaf size

  cout << "dataset: " << dataset.n_rows << " : " << dataset.n_cols << endl;
  cout << "labels: " << labels.n_rows << " : " << labels.n_cols << endl;
  
  r.Classify(dataset, predictions);
  const double trainError = arma::accu(predictions != labels) * 100. / labels.n_elem;
  cout << "Training error: " << trainError << "%." << endl;
  
  auto gradientBoostClassifier = GradientBoostClassifier<double>(dataset, labels, 100);

  return 0;
}
