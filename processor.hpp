#ifndef __PROCESSOR_HPP__
#define __PROCESSOR_HPP__

#include <random>
#include <list>
#include <vector>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <functional>

#include "visitor.hpp"
#include "dataset.hpp"

template<typename DataType>
class SplitProcessor : public Visitor {
public:
  SplitProcessor() : r_{.8} {};
  SplitProcessor(float r) : r_{r} {}

  void split(DataSet<DataType>*, float);

  DataSet<DataType> get_train_data() const;
  DataSet<DataType> get_test_data() const;

  typename DataSet<DataType>::DataVecVec getX_train() const;
  typename DataSet<DataType>::DataVec gety_train() const;
  typename DataSet<DataType>::DataVecVec getX_test() const;
  typename DataSet<DataType>::DataVec gety_test() const;
private:
  void generate(DataElement*);
  
  float r_;
  DataSet<DataType> train_;
  DataSet<DataType> test_;
};

#include "processor_impl.hpp"

#endif
