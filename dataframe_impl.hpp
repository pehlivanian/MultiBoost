#ifndef __DATAFRAME_IMPL_HPP__
#define __DATAFRAME_IMPL_HPP__

template<typename DataType>
DataFrame<DataType>::DataFrame(const DataFrame& rhs) {
  *this = rhs;
}

template<typename DataType>
DataFrame<DataType>::DataFrame(DataFrame&& rhs) {
  *this = std::move(rhs);
}

template<typename DataType>
DataFrame<DataType>::DataFrame(std::string Xpath, std::string ypath, bool header) {
  read_csv(Xpath, ypath, header);
}

template<typename DataType>
std::pair<int, int>
DataFrame<DataType>::shape() const {
  return std::make_pair(m_, n_);
}

template<typename DataType>
DataFrame<DataType>::DataFrame(const DataVecVec& data, const YVec& y) {
  int m = data.size(), n = data[0].size();

  data_ = data;
  y_ = y;
  m_ = data_.size();
  n_ = data_[0].size();
}

template<typename DataType>
template<typename ContainerType>
void
DataFrame<DataType>::reduce_rows(typename ContainerType::iterator b, typename ContainerType::iterator e) {
  assert (std::distance(b,e) <= m_);
  
  DataVecVec newData;
  auto shape = this->shape();

  for (int i=0; i<shape.first; ++i) {
    newData[i].push_back(*(b+i));
  }
  
  data_.clear();
  data_ = newData;
}

template<typename DataType>
template<typename ContainerType>
void
DataFrame<DataType>::reduce_columns(typename ContainerType::iterator b, typename ContainerType::iterator e) {
  ;
}

template<typename DataType>
std::pair<typename DataFrame<DataType>::DataVecVec, typename DataFrame<DataType>::DataVec>
DataFrame<DataType>::reduce_data(const DataVecVec& d, 
				 const DataVec& y,
				 std::vector<int> filt,
				 bool reduceRows) {
  int n = d[0].size();
  DataVecVec newData;
  DataVec newy;

  if (reduceRows) {
    for (size_t i=0; i<filt.size(); ++i) {
      newData.push_back(d[filt[i]]);
      newy.push_back(y[filt[i]]);
    }
  }
  else {
    for (size_t i=0; i<n; ++i) {
      DataVec row;
      for (size_t j=0; j<filt.size(); ++j) {
	row.push_back(d[i][filt[j]]);
      }
      newData.push_back(row);
    }
    for(size_t i=0; i<n; ++i) {
      newy.push_back(y[filt[i]]);
    }
  }

    return std::make_pair(newData, newy);
}

template<typename DataType>
void
DataFrame<DataType>::reduce_rows(std::vector<int> rows) {

  assert (rows.size() <= m_);

  auto newData = reduce_data(data_, rows, true);
  this->setData(newData);

}

template<typename DataType>
void
DataFrame<DataType>::reduce_columns(std::vector<int> columns) {
  
  assert (columns.size() <= n_);

  auto newData = reduce_data(data_, columns, false);
  this->setData(newData);
}

template<typename DataType>
DataFrame<DataType>&
DataFrame<DataType>::operator=(const DataFrame& rhs) {
  if (this != &rhs) {
    data_ = rhs.data_;
    m_ = rhs.m_;
    n_ = rhs.n_;
  }
  return (*this);
}

template<typename DataType>
DataFrame<DataType>&
DataFrame<DataType>::operator=(DataFrame &&rhs) {
  if (this != &rhs) {
    data_ = std::exchange(rhs.data_, DataVecVec{});
    m_ = std::exchange(rhs.m_, 0);
    n_ = std::exchange(rhs.n_, 0);
  }
  return (*this);
}

template<typename DataType>
void
DataFrame<DataType>::read_csv(std::string Xpath, 
			      std::string ypath,
			      bool header) {
  
  std::ifstream file(Xpath);
  std::string line;
  
  while(std::getline(file, line)) {
    std::stringstream lineStream(line);
    std::string cell;
    std::vector<DataType> vin;
    while(std::getline(lineStream, cell, ',')) {
      DataType val;
      std::istringstream ss(cell);
      ss >> val;
      vin.push_back(val);
    }
    n_ = vin.size();
    data_.push_back(vin);
  } 
  m_ = data_.size();
  
  std::ifstream yfile(ypath);
  
  while (std::getline(yfile, line)) {
    std::stringstream ylineStream(line);
    std::string ycell;
    while (std::getline(ylineStream, ycell, ',')) {
      DataType yval;
      std::istringstream yss(ycell);
      yss >> yval;
      y_.push_back(yval);
    }
  }
 }
 
template<typename DataType>
typename DataFrame<DataType>::DataVec
DataFrame<DataType>::gety() const {
  return y_;
 }

template<typename DataType>
void
DataFrame<DataType>::sety(const DataVec& y) {
  y_ = y;
 }

template<typename DataType>
void
DataFrame<DataType>::sety(DataVec&& y) {
  y_ = std::exchange(y, DataVec{});
 }

template<typename DataType>
typename DataFrame<DataType>::DataVecVec
DataFrame<DataType>::getData() const {
  return data_;
}

template<typename DataType>
void
DataFrame<DataType>::setData(const DataVecVec& data) {
  data_ = data;
  m_ = data.size();
  n_ = data[0].size();
}

template<typename DataType>
void
DataFrame<DataType>::setData(DataVecVec&& data) {
  auto shape = data.shape();
  data_ = std::exchange(data, DataVecVec{});
  m_ = std::exchange(shape.first, 0);
  n_ = std::exchange(shape.second, 0);
}

template<typename DataType>
std::vector<DataType>
DataFrame<DataType>::operator[](std::size_t ind1) {
  return data_[ind1];
}

template<typename DataType>
std::ostream& operator<<(std::ostream& out, const DataFrame<DataType>& rhs) {
  typename DataFrame<DataType>::DataVecVec rhsData = rhs.getData();

  for (auto const& row : rhsData) {
    for (auto const& item : row) {
      out << item << " ";
    }
    out << "\n";
  }
  
  return out;
}

#endif
