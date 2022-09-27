#ifndef __PROCESSOR_IMPL_HPP__
#define __PROCESSOR_IMPL_HPP__

template<typename DataType>
void
SplitProcessor<DataType>::split(DataSet<DataType>* d, float r) {
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
  std::pair<typename DataSet<DataType>::DataVecVec, 
    typename DataSet<DataType>::YVec> trainPair = d->reduce_data(d->getData(),
								 d->gety(),
								 newInd,
								 true);
  std::pair<typename DataSet<DataType>::DataVecVec, 
    typename DataSet<DataType>::YVec> testPair = d->reduce_data(d->getData(),
								d->gety(),
								newIndComp,
								true);
  
  train_ = DataSet<DataType>(trainPair.first, trainPair.second);
  test_ = DataSet<DataType>(testPair.first, testPair.second);
}

template<typename DataType>
void
SplitProcessor<DataType>::generate(DataElement* d) {
  split(dynamic_cast<DataSet<DataType>*>(d), r_);
}

template<typename DataType>
typename DataSet<DataType>::DataVecVec
SplitProcessor<DataType>::getX_train() const {
  return train_.getData();
}

template<typename DataType>
typename DataSet<DataType>::DataVec
SplitProcessor<DataType>::gety_train() const {
  return train_.gety();
}

template<typename DataType>
typename DataSet<DataType>::DataVecVec
SplitProcessor<DataType>::getX_test() const {
  return test_.getData();
}

template<typename DataType>
typename DataSet<DataType>::DataVec
SplitProcessor<DataType>::gety_test() const {
  return test_.gety();
}

template<typename DataType>
DataSet<DataType>
SplitProcessor<DataType>::get_train_data() const {
  return train_;
}

template<typename DataType>
DataSet<DataType>
SplitProcessor<DataType>::get_test_data() const {
  return test_;
}

#endif
