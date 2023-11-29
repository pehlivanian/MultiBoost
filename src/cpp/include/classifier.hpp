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

private:
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
  { init_(dataset, labels, false, std::forward<Args>(args)...); }

  DiscreteClassifierBase(const Mat<DataType>& dataset, Row<DataType>& labels, Row<DataType>& weights, Args&&... args) : 
    ClassifierBase<DataType, ClassifierType>(typeid(*this).name())
  { weights_ = weights;
    init_(dataset, labels, true, std::forward<Args>(args)...); 
  }

  DiscreteClassifierBase(const LeavesMap& leavesMap, std::unique_ptr<ClassifierType> classifier) : 
    leavesMap_{leavesMap},
    classifier_{std::move(classifier)} {}

  void init_(const Mat<DataType>&, Row<DataType>&, bool, Args&&...);

  DiscreteClassifierBase() = default;
  DiscreteClassifierBase(const DiscreteClassifierBase&) = default;
  virtual ~DiscreteClassifierBase() = default;

  void setClassifier(const Mat<DataType>&, Row<std::size_t>&, bool, Args&&...);
  Row<DataType> getWeights() const { return weights_; }
  void setWeights(const Row<DataType>& weights) { weights_ = weights; }

  template<class Archive>
  void serialize(Archive &ar) {
    ar(cereal::base_class<ClassifierBase<DataType, ClassifierType>>(this), CEREAL_NVP(leavesMap_));
    ar(cereal::base_class<ClassifierBase<DataType, ClassifierType>>(this), CEREAL_NVP(weights_));
    ar(cereal::base_class<ClassifierBase<DataType, ClassifierType>>(this), CEREAL_NVP(classifier_));
    // Don't serialize args_
    // ar(cereal::base_class<ClassifierBase<DataType, ClassifierType>>(this), CEREAL_NVP(args_));
  }

private:
  void encode(const Row<DataType>&, Row<std::size_t>&, bool); 
  void decode(const Row<std::size_t>&, Row<DataType>&);

  Row<std::size_t> labels_t_;
  Row<DataType> weights_;
  LeavesMap leavesMap_;
  std::unique_ptr<ClassifierType> classifier_;
  std::tuple<Args...> args_;

  void Classify_(const Mat<DataType>&, Row<DataType>&) override;
  void Classify_(Mat<DataType>&&, Row<DataType>&) override;
  void purge_() override;

};

#include "classifier_impl.hpp"

#endif
