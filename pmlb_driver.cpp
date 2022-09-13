#include "pmlb_driver.hpp"

auto main(int argc, char **argv) -> int {

  std::string Xpath = "/home/charles/Data/test_X.csv";
  std::string ypath = "/home/charles/Data/test_y.csv";
  auto df = DataFrame<float>(Xpath, ypath, false);
  std::size_t ind1 = 4, ind2 = 2;
  
  auto shape = df.shape();

  std::cout << "COMPLETE." << std::endl;
  std::cout << "SIZE: (" << shape.first << ", " 
	    << shape.second << ")" << std::endl;
  std::cout << "df[" << ind1 << "][" << ind2
	    << "]: " << df[ind1][ind2] << std::endl;

  auto splitter = SplitProcessor<float>(.8);
  
  df.accept(splitter);

  std::cout << splitter.getX_train().size() << std::endl;
  std::cout << splitter.gety_train().size() << std::endl;
  std::cout << splitter.getX_test().size() << std::endl;
  std::cout << splitter.gety_test().size() << std::endl;

  // std::cout << df << std::endl;

  return 0;
}
