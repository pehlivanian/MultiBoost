#ifndef __DATAFRAME_IMPL_HPP__
#define __DATAFRAME_IMPL_HPP__

template<typename DataType>
DataSet<DataType>::DataSet(const DataSet& rhs) {
  *this = rhs;
}

template<typename DataType>
DataSet<DataType>::DataSet(DataSet&& rhs) {
  *this = std::move(rhs);
}

template<typename DataType>
DataSet<DataType>::DataSet(std::string Xpath, std::string ypath, bool header) {
  read_csv(Xpath, ypath, header);
}

template<typename DataType>
std::pair<int, int>
DataSet<DataType>::shape() const {
  return std::make_pair(m_, n_);
}

template<typename DataType>
template<typename ContainerType>
void
DataSet<DataType>::reduce_rows(typename ContainerType::iterator b, typename ContainerType::iterator e) {
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
DataSet<DataType>::reduce_columns(typename ContainerType::iterator b, typename ContainerType::iterator e) {
  ;
}

template<typename DataType>
std::pair<typename DataSet<DataType>::DataVecVec, typename DataSet<DataType>::YVec>
DataSet<DataType>::reduce_data(const DataVecVec& d, 
			       const YVec& y,
			       std::vector<int> filt,
			       bool reduceRows) {
  int n = d[0].size();
  DataVecVec newData;
  YVec newy;

  if (reduceRows) {
    for (size_t i=0; i<filt.size(); ++i) {
      newData.push_back(d[filt[i]]);
      newy.push_back(y[filt[i]]);
    }
  }
  else {
    for (int i=0; i<n; ++i) {
      YVec row;
      for (size_t j=0; j<filt.size(); ++j) {
	row.push_back(d[i][filt[j]]);
      }
      newData.push_back(row);
    }
    for(size_t i=0; i<filt.size(); ++i) {
      newy.push_back(y[filt[i]]);
    }
  }

    return std::make_pair(newData, newy);
}

template<typename DataType>
void
DataSet<DataType>::reduce_rows(std::vector<int> rows) {

  assert (rows.size() <= m_);

  auto newPair = reduce_data(data_, rows, true);
  this->setData(newPair.first);
  this->sety(newPair.second);

}

template<typename DataType>
void
DataSet<DataType>::reduce_columns(std::vector<int> columns) {
  
  assert (columns.size() <= n_);

  auto newPair = reduce_data(data_, columns, false);
  this->setData(newPair.first);
  this->sety(newPair.second);
}

template<typename DataType>
DataSet<DataType>&
DataSet<DataType>::operator=(const DataSet& rhs) {
  if (this != &rhs) {
    data_ = rhs.data_;
    y_ = rhs.y_;
    m_ = rhs.m_;
    n_ = rhs.n_;
  }
  return (*this);
}

template<typename DataType>
DataSet<DataType>&
DataSet<DataType>::operator=(DataSet &&rhs) {
  if (this != &rhs) {
    data_ = std::exchange(rhs.data_, DataVecVec{});
    y_ = std::exchange(rhs.y_, YVec{});
    m_ = std::exchange(rhs.m_, 0);
    n_ = std::exchange(rhs.n_, 0);
  }
  return (*this);
}

template<typename DataType>
void
DataSet<DataType>::read_csv(std::string Xpath, 
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
typename DataSet<DataType>::DataVec
DataSet<DataType>::gety() const {
  return y_;
 }

template<typename DataType>
void
DataSet<DataType>::sety(const DataVec& y) {
  y_ = y;
 }

template<typename DataType>
void
DataSet<DataType>::sety(DataVec&& y) {
  y_ = std::exchange(y, DataVec{});
 }

template<typename DataType>
typename DataSet<DataType>::DataVecVec
DataSet<DataType>::getData() const {
  return data_;
}

template<typename DataType>
void
DataSet<DataType>::setData(const DataVecVec& data) {
  data_ = data;
  m_ = data.size();
  n_ = data[0].size();
}

template<typename DataType>
void
DataSet<DataType>::setData(DataVecVec&& data) {
  auto shape = data.shape();
  data_ = std::exchange(data, DataVecVec{});
  m_ = std::exchange(shape.first, 0);
  n_ = std::exchange(shape.second, 0);
}

template<typename DataType>
std::vector<DataType>
DataSet<DataType>::operator[](std::size_t ind1) {
  return data_[ind1];
}

template<typename DataType>
std::ostream& operator<<(std::ostream& out, const DataSet<DataType>& rhs) {
  typename DataSet<DataType>::DataVecVec rhsData = rhs.getData();

  for (auto const& row : rhsData) {
    for (auto const& item : row) {
      out << item << " ";
    }
    out << "\n";
  }
  
  return out;
}

#endif
