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
class DataFrame : public DataElement {

public:
  using DataVec = std::vector<DataType>;
  using YVec = DataVec;
  using DataVecVec = std::vector<DataVec>;

public:
  using size_type = typename std::vector<DataType>::size_type;
  using ColNameType = std::string;

  DataFrame() = default;
  DataFrame(std::string, std::string, bool);

  DataFrame(const DataVecVec&, const YVec&);
  DataFrame(const DataVecVec&, std::vector<int>, bool);

  DataFrame(const DataFrame&);
  DataFrame(DataFrame&&);
  DataFrame &operator=(const DataFrame&);
  DataFrame &operator=(DataFrame&&);

  DataVecVec getData() const;
  void setData(const DataVecVec& data);
  void setData(DataVecVec&&);

  DataVec gety() const;
  void sety(const DataVec&);
  void sety(DataVec&&);

  void reduce_rows(std::vector<int>);
  void reduce_columns(std::vector<int>);
  template<typename ContainerType>
  void reduce_rows(typename ContainerType::iterator, typename ContainerType::iterator);
  template<typename ContainerType>
  void reduce_columns(typename ContainerType::iterator, typename ContainerType::iterator);
  std::pair<DataVecVec, DataVec> reduce_data(const DataVecVec&, 
					     const DataVec&,
					     std::vector<int>, 
					     bool);

  std::vector<DataType> operator[](std::size_t);

  std::pair<int, int> shape() const;

private:
  using ColNameDict = 
    std::unordered_map<ColNameType,
		       size_type,
		       std::hash<ColNameType>>;
  using ColNameList = 
    std::vector<std::pair<ColNameType, size_type>>;

  void read_csv(std::string, std::string, bool);

  DataVecVec data_ {};
  DataVec y_ {};
  ColNameDict column_tb_ {};
  ColNameList column_list_ {};
  std::size_t m_, n_;
};

#include "dataframe_impl.hpp"

#endif
