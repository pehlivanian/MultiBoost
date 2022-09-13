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
#include "dataframe.hpp"

template<typename DataType>
class SplitProcessor : public Visitor {
public:
  SplitProcessor() : r_{.8} {};
  SplitProcessor(float r) : r_{r} {}

  void split(DataFrame<DataType>*, float);

  typename DataFrame<DataType>::DataVecVec getX_train() const;
  typename DataFrame<DataType>::DataVec gety_train() const;
  typename DataFrame<DataType>::DataVecVec getX_test() const;
  typename DataFrame<DataType>::DataVec gety_test() const;
private:
  void generate(DataElement*);
  
  float r_;
  typename DataFrame<DataType>::DataVecVec X_train;
  typename DataFrame<DataType>::DataVecVec X_test;
  typename DataFrame<DataType>::DataVec y_train;
  typename DataFrame<DataType>::DataVec y_test;
};

#include "processor_impl.hpp"

#endif
