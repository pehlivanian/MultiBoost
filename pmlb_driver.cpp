#include "pmlb_driver.hpp"

using namespace arma;
using namespace mlpack;
using namespace mlpack::tree;
using namespace mlpack::data;
using namespace mlpack::util;
using namespace std;

using namespace LossMeasures;

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
  rowvec labels, predictions;

  if (!data::Load("/home/charles/Data/test_X.csv", dataset))
    throw std::runtime_error("Could not load test_X.csv");
  if (!data::Load("/home/charles/Data/test_y.csv", labels))
    throw std::runtime_error("Could not load test_y.csv");

  uvec rowMask = linspace<uvec>(0, -1+dataset.n_rows, dataset.n_rows);
  uvec colMask = linspace<uvec>(0, -1+10000, 10000);

  dataset = dataset.submat(rowMask, colMask);
  labels = labels.submat(zeros<uvec>(1), colMask);

  bool symmetrize = true;

  auto gradientBoostClassifier = GradientBoostClassifier<double>(dataset, 
								 labels, 
								 lossFunction::BinomialDeviance,
								 5,
								 .25,
								 500,
								 symmetrize);

  return 0;
}
