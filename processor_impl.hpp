#ifndef __PROCESSOR_IMPL_HPP__
#define __PROCESSOR_IMPL_HPP__

template<typename DataType>
void
SplitProcessor<DataType>::split(DataFrame<DataType>* d, float r) {
  int targetNumRows;
  std::random_device rnd_device;
  std::mt19937 mersenne_engine{rnd_device()};

  auto rows = d->shape().first;
  targetNumRows = int(r * rows);

  std::list<int> ind(rows);
  std::iota(ind.begin(), ind.end(), 0);

  std::vector<std::list<int>::iterator> v(static_cast<int>(ind.size()));
  std::iota(v.begin(), v.end(), ind.begin());

  std::shuffle(v.begin(), v.end(), std::mt19937{std::random_device{}()});
  
  
  /*
    std::cout << "ORIGINAL LIST:" << std::endl;
    for (auto const& i: ind) std::cout << i << " ";
    
    std::cout << "\nSHUFFLED LIST:" << std::endl;
    for (auto const& i : v) std::cout << *i << " ";
    std::cout << "\n";
  */
  
  int numRows = int(r*rows);
  std::vector<int> newInd, newIndComp;

  for (int i=0; i<numRows; ++i) {
    newInd.push_back(*(v[i]));
  }
  for (int i=numRows; i<v.size(); ++i) {
    newIndComp.push_back(*(v[i]));
  }


  // d->reduce_rows(newInd);
  auto trainPair = d->reduce_data(d->getData(),
				  d->gety(),
				  newInd,
				  true);
  auto testPair = d->reduce_data(d->getData(),
				 d->gety(),
				 newIndComp,
				 true);

  X_train = trainPair.first; y_train = trainPair.second;
  X_test = testPair.first; y_test = testPair.second;
  
}

template<typename DataType>
void
SplitProcessor<DataType>::generate(DataElement* d) {
  split(dynamic_cast<DataFrame<DataType>*>(d), r_);
  ;
}

template<typename DataType>
typename DataFrame<DataType>::DataVecVec
SplitProcessor<DataType>::getX_train() const {
  return X_train;
}

template<typename DataType>
typename DataFrame<DataType>::DataVec
SplitProcessor<DataType>::gety_train() const {
  return y_train;
}

template<typename DataType>
typename DataFrame<DataType>::DataVecVec
SplitProcessor<DataType>::getX_test() const {
  return X_test;
}

template<typename DataType>
typename DataFrame<DataType>::DataVec
SplitProcessor<DataType>::gety_test() const {
  return y_test;
}

#endif
