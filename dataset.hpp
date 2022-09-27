#ifndef __DATAFRAME_HPP__
#define __DATAFRAME_HPP__


#include <functional>
#include <algorithm>
#include <future>
#include <ios>
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <typeindex>
#include <unordered_map>
#include <utility>
#include <vector>
#include <cassert>

#include "dataelement.hpp"

template<typename DataType>
class DataSet : public DataElement {

public:
  using DataVec = std::vector<DataType>;
  using DataVecVec = std::vector<DataVec>;
  using YVec = DataVec;

public:
  using size_type = typename std::vector<DataType>::size_type;
  using ColNameType = std::string;

  DataSet() = default;
  DataSet(std::string, std::string, bool);

  DataSet(const DataVecVec& data, const YVec& y) : 
    data_{data},
    y_{y},
    m_{data_.size()},
    n_{data[0].size()} {}

  DataSet(const DataSet&);
  DataSet(DataSet&&);
  DataSet &operator=(const DataSet&);
  DataSet &operator=(DataSet&&);

  DataVecVec getData() const;
  void setData(const DataVecVec& data);
  void setData(DataVecVec&&);

  DataVec gety() const;
  void sety(const DataVec&);
  void sety(DataVec&&);

  int getm() const { return m_; }
  int getn() const { return n_; }

  void reduce_rows(std::vector<int>);
  void reduce_columns(std::vector<int>);
  template<typename ContainerType>
  void reduce_rows(typename ContainerType::iterator, typename ContainerType::iterator);
  template<typename ContainerType>
  void reduce_columns(typename ContainerType::iterator, typename ContainerType::iterator);
  std::pair<DataVecVec, YVec> reduce_data(const DataVecVec&, 
					     const DataVec&,
					     std::vector<int>, 
					     bool);

  std::vector<DataType> operator[](std::size_t);

  std::pair<int, int> shape() const;

private:
  void read_csv(std::string, std::string, bool);

  using ColNameDict = 
    std::unordered_map<ColNameType,
		       size_type,
		       std::hash<ColNameType>>;
  using ColNameList = 
    std::vector<std::pair<ColNameType, size_type>>;

  DataVecVec data_ {};
  YVec y_ {};
  ColNameDict column_tb_ {};
  ColNameList column_list_ {};
  std::size_t m_, n_;

};

#include "dataset_impl.hpp"

#endif
