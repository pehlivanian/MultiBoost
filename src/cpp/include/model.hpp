#ifndef __MODEL_HPP__
#define __MODEL_HPP__

#include <memory>
#include <exception>

#include <mlpack/core.hpp>

using namespace arma;

struct predictionAfterClearedClassifiersException : public std::exception {
  const char* what() const throw () {
    return "Attempting to predict on a classifier that has been serialized and cleared";
  }
};

// Helpers for gdb
template<class Matrix>
void print_matrix(Matrix matrix) {
  matrix.print(std::cout);
}

template<class Row>
void print_vector(Row row) {
  row.print(std::cout);
}

template void print_matrix<arma::mat>(arma::mat matrix);
template void print_vector<arma::rowvec>(arma::rowvec row);

template<typename DataType, typename ClassifierType>
class Model {
public:
  Model() = default;
  Model(std::string id) : id_{id} {}

  void Project(const mat& data, Row<DataType>& projection) { 
    Project_(data, projection); 
  }

  std::string get_id() const { return id_; }

  template<class Archive>
  void serialize(Archive &ar) {
    ar(id_);
  }

private:
  std::string id_;

  virtual void Project_(const mat&, Row<DataType>&) = 0;
  virtual void Project_(mat&&, Row<DataType>&) = 0;
};

namespace Model_Traits {
  template<typename ModelType>
  struct is_classifier {
    bool operator()() { return true; }
  };

}// namespace Model_Traits

#endif