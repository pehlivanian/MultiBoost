#ifndef __CLASSIFIER_HPP__
#define __CLASSIFIER_HPP__

#include <memory>

#include <mlpack/core.hpp>

#include "model.hpp"

using namespace arma;

template<typename DataType, typename ClassifierType>
class ClassifierBase : public Model<DataType> {
public:
  ClassifierBase() = default;
  ClassifierBase(const ClassifierBase&) = default;
  ClassifierBase(std::string id) : Model<DataType>(id) {}

  virtual ~ClassifierBase() = default;

  virtual void Classify(const Mat<DataType>& data, Row<DataType>& pred) { Classify_(data, pred); }
  virtual void Classify(Mat<DataType>&& data, Row<DataType>& pred) { Classify_(std::move(data), pred); }

  virtual void purge() { purge_(); }

private:
  virtual void purge_() = 0;
  virtual void Classify_(const Mat<DataType>&, Row<DataType>&) = 0;
  virtual void Classify_(Mat<DataType>&&, Row<DataType>&) = 0;

  void Project_(const Mat<DataType>& data, Row<DataType>& pred) override { Classify_(data, pred); }
  void Project_(Mat<DataType>&& data, Row<DataType>& pred) override { Classify_(std::move(data), pred); }
  
};

template<typename DataType, typename ClassifierType, typename... Args>
class DiscreteClassifierBase : public ClassifierBase<DataType, ClassifierType> {
public:
  using LeavesMap = std::unordered_map<std::size_t, DataType>;

  DiscreteClassifierBase(const Mat<DataType>& dataset, Row<DataType>& labels, Args&&... args) : 
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
  DiscreteClassifierBase(const DiscreteClassifierBase&) = default;
  virtual ~DiscreteClassifierBase() = default;

  void setClassifier(const Mat<DataType>&, Row<std::size_t>&, Args&&...);

  template<class Archive>
  void serialize(Archive &ar) {
    ar(cereal::base_class<ClassifierBase<DataType, ClassifierType>>(this), CEREAL_NVP(leavesMap_));
    ar(cereal::base_class<ClassifierBase<DataType, ClassifierType>>(this), CEREAL_NVP(classifier_));
    // Don't serialize args_
    // ar(cereal::base_class<ClassifierBase<DataType, ClassifierType>>(this), CEREAL_NVP(args_));
  }

private:
  void encode(const Row<DataType>&, Row<std::size_t>&); 
  void decode(const Row<std::size_t>&, Row<DataType>&);

  Row<std::size_t> labels_t_;
  LeavesMap leavesMap_;
  std::unique_ptr<ClassifierType> classifier_;
  std::tuple<Args...> args_;

  void Classify_(const Mat<DataType>&, Row<DataType>&) override;
  void Classify_(Mat<DataType>&&, Row<DataType>&) override;
  void purge_() override;

};

#include "classifier_impl.hpp"

#endif
