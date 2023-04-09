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
class ClassifierBase {
public:
  ClassifierBase() = default;
  ClassifierBase(std::string id) : id_{id} {}
  virtual void Classify_(const mat&, Row<DataType>&) = 0;
  virtual void purge() = 0;

  std::string get_id() const { return id_; }

  template<class Archive>
  void serialize(Archive &ar) {
    ar(id_);
  }
private:
  std::string id_;
};

template<typename DataType, typename ClassifierType, typename... Args>
class DiscreteClassifierBase : public ClassifierBase<DataType, ClassifierType> {
public:
  using LeavesMap = std::unordered_map<std::size_t, DataType>;

  DiscreteClassifierBase(const mat& dataset, Row<DataType>& labels, Args&&... args) : 
    ClassifierBase<DataType, ClassifierType>(typeid(*this).name())
  {

    labels_t_ = Row<std::size_t>(labels.n_cols);
    encode(labels, labels_t_);
    setClassifier(dataset, labels_t_, std::forward<Args>(args)...);
    args_ = std::tuple<Args...>(args...);
    
    // Check error
    /*
      Row<std::size_t> prediction;
      classifier_->Classify(dataset, prediction);
      const double trainError = err(prediction, labels_t_);
      for (size_t i=0; i<25; ++i)
      std::cout << labels_t_[i] << " ::(1) " << prediction[i] << std::endl;
      std::cout << "dataset size:    " << dataset.n_rows << " x " << dataset.n_cols << std::endl;
      std::cout << "prediction size: " << prediction.n_rows << " x " << prediction.n_cols << std::endl;
      std::cout << "Training error (1): " << trainError << "%." << std::endl;
    */
  
  }

  DiscreteClassifierBase(const LeavesMap& leavesMap, std::unique_ptr<ClassifierType> classifier) : 
    leavesMap_{leavesMap},
    classifier_{std::move(classifier)} {}

  DiscreteClassifierBase() = default;
  ~DiscreteClassifierBase() = default;

  void setClassifier(const mat&, Row<std::size_t>&, Args&&...);
  void Classify_(const mat&, Row<DataType>&) override;
  void Classify_(const mat&, Row<DataType>&, mat&);
  void purge() override;

  template<class Archive>
  void serialize(Archive &ar) {
    ar(cereal::base_class<ClassifierBase<DataType, ClassifierType>>(this), CEREAL_NVP(leavesMap_));
    ar(cereal::base_class<ClassifierBase<DataType, ClassifierType>>(this), CEREAL_NVP(classifier_));
  }

private:
  void encode(const Row<DataType>&, Row<std::size_t>&); 
  void decode(const Row<std::size_t>&, Row<DataType>&);

  Row<std::size_t> labels_t_;
  LeavesMap leavesMap_;
  std::unique_ptr<ClassifierType> classifier_;
  std::tuple<Args...> args_;

};

template<typename DataType, typename ClassifierType, typename... Args>
class ContinuousClassifierBase : public ClassifierBase<DataType, ClassifierType> {
public:  
  ContinuousClassifierBase(const mat& dataset, Row<DataType>& labels, Args&&... args) : 
    ClassifierBase<DataType, ClassifierType>(typeid(*this).name()) 
  {

    setClassifier(dataset, labels, std::forward<Args>(args)...);
    args_ = std::tuple<Args...>(args...);
  }

  ContinuousClassifierBase(std::unique_ptr<ClassifierType> classifier) : classifier_{std::move(classifier)} {}

  ContinuousClassifierBase() = default;
  ~ContinuousClassifierBase() = default;
  
  void setClassifier(const mat&, Row<DataType>&, Args&&...);
  void Classify_(const mat&, Row<DataType>&) override;
  void purge() override {};

  template<class Archive>
  void serialize(Archive &ar) {

    ar(cereal::base_class<ClassifierBase<DataType, ClassifierType>>(this), CEREAL_NVP(classifier_));
  }

private:
  std::unique_ptr<ClassifierType> classifier_;
  std::tuple<Args...> args_;
};

#include "model_impl.hpp"

#endif
