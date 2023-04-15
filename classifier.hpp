#ifndef __CLASSIFIER_HPP__
#define __CLASSIFIER_HPP__

#include <memory>

#include <mlpack/core.hpp>

#include "model.hpp"

using namespace arma;

template<typename DataType, typename ClassifierType>
class ClassifierBase : public Model<DataType, ClassifierType> {
public:
  ClassifierBase() = default;
  ClassifierBase(std::string id) : Model<DataType, ClassifierType>(id) {}

  virtual void Classify(const mat& data, Row<DataType>& pred) { Classify_(data, pred); }
  virtual void purge() { purge_(); }

private:
  virtual void purge_() = 0;
  virtual void Classify_(const mat&, Row<DataType>&) = 0;

  void Project_(const mat& data, Row<DataType>& pred) override { Classify_(data, pred); }
  
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
    
  }

  DiscreteClassifierBase(const LeavesMap& leavesMap, std::unique_ptr<ClassifierType> classifier) : 
    leavesMap_{leavesMap},
    classifier_{std::move(classifier)} {}

  DiscreteClassifierBase() = default;

  void setClassifier(const mat&, Row<std::size_t>&, Args&&...);

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

  void Classify_(const mat&, Row<DataType>&) override;
  void purge_() override;

};

#include "classifier_impl.hpp"

#endif
