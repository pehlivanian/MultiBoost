#ifndef __CLASSIFIER_LOSS_HPP__
#define __CLASSIFIER_LOSS_HPP__

#include <unordered_map>

#include "loss2.hpp"

enum class classifierLossFunction {
  BinomialDeviance = 0;
  Savage = 1;
  Exp = 2;
  Arctan = 3;
  SyntheticLoss = 4;
  SyntheticLossVar1 = 5;
  SyntheticLossVar2 = 6;
  SquareLsos = 7;
  PowerLoss = 8;
  CrossEntropyLoss = 9;
};

namespace ClassifierLossMeasures {
  template<typename DataType>
  class BinomialDevianceLoss : public LossFunction<DataType> {
  private:
    using vtype = typename LossFunction<DataType>::vtype;
    using mtype = typename LossFunction<DataType>::mtype;

  public:
    BinomialDevianceLoss() = default;
    BinomialDevianceLoss<DataType>* create() override { return new BinomialDevianceLoss<DataType>(); }
  private:
#ifdef AUTODIFF
    autodiff::real loss_reverse(const ArrayXreal&, const ArrayXreal&) override;  
#endif
    DataType loss_reverse_arma(const vtype&, const vtype&) override;
    DataType loss_arma(const vtype&, const vtype&) override;
    DataType gradient_(const vtype&, const vtype&, vtype*) override;
    void hessian_(const vtype&, const vtype&, vtype*) override;
  };

  template<typename DataType>
  class SavageLoss : public LossFunction<DataType> {
  private:
    using vtype = typename LossFunction<DataType>::vtype;
    using mtype = typename LossFunction<DataType>::mtype;

  public:
    SavageLoss() = default;
    SavageLoss<DataType>* create() override { return new SavageLoss<DataType>(); }
  private:
#ifdef AUTODIFF
    autodiff::real loss_reverse(const ArrayXreal&, const ArrayXreal&) override;  
#endif
    DataType loss_reverse_arma(const vtype&, const vtype&) override;
    DataType loss_arma(const vtype&, const vtype&) override;
    DataType gradient_(const vtype&, const vtype&, vtype*) override;
    void hessian_(const vtype&, const vtype&, vtype*) override;
  };

  template<typename DataType>
  class ExpLoss : public LossFunction<DataType> {
  private:
    using vtype = typename LossFunction<DataType>::vtype;
    using mtype = typename LossFunction<DataType>::mtype;

  public:
    ExpLoss() = default;
    ExpLoss<DataType>* create() override { return new ExpLoss<DataType>(); }
  private:
#ifdef AUTODIFF
    autodiff::real loss_reverse(const ArrayXreal&, const ArrayXreal&) override;
#endif
    DataType loss_reverse_arma(const vtype&, const vtype&) override;
    DataType loss_arma(const vtype&, const vtype&) override;
    DataType gradient_(const vtype&, const vtype&, vtype*) override;
    void hessian_(const vtype&, const vtype&, vtype*) override;
  };

  template<typename DataType>
  class ArctanLoss : public LossFunction<DataType> {
  private:
    using vtype = typename LossFunction<DataType>::vtype;
    using mtype = typename LossFunction<DataType>::mtype;

  public:
    ArctanLoss() = default;
    ArctanLoss<DataType>* create() override { return new ArctanLoss<DataType>(); }
  private:
#ifdef AUTODIFF
    autodiff::real loss_reverse(const ArrayXreal&, const ArrayXreal&) override;  
#endif
    DataType loss_reverse_arma(const vtype&, const vtype&) override;
    DataType loss_arma(const vtype&, const vtype&) override;
    DataType gradient_(const vtype&, const vtype&, vtype*) override;
    void hessian_(const vtype&, const vtype&, vtype*) override;
  };


  template<typename DataType>
  class SyntheticLoss : public LossFunction<DataType> {
  private:
    using vtype = typename LossFunction<DataType>::vtype;
    using mtype = typename LossFunction<DataType>::mtype;

  public:
    SyntheticLoss() = default;
    SyntheticLoss<DataType>* create() override { return new SyntheticLoss<DataType>(); }
  private:
#ifdef AUTODIFF
    autodiff::real loss_reverse(const ArrayXreal&, const ArrayXreal&) override;  
#endif
    DataType loss_reverse_arma(const vtype&, const vtype&) override;
    DataType loss_arma(const vtype&, const vtype&) override;
    DataType gradient_(const vtype&, const vtype&, vtype*) override;
    void hessian_(const vtype&, const vtype&, vtype*) override;
  };

  template<typename DataType>
  class SyntheticLossVar1 : public LossFunction<DataType> {
  private:
    using vtype = typename LossFunction<DataType>::vtype;
    using mtype = typename LossFunction<DataType>::mtype;

  public:
    SyntheticLossVar1() = default;
    SyntheticLossVar1<DataType>* create() override { return new SyntheticLossVar1<DataType>(); }
  private:
#ifdef AUTODIFF
    autodiff::real loss_reverse(const ArrayXreal&, const ArrayXreal&) override;
#endif
    DataType loss_reverse_arma(const vtype&, const vtype&) override;
    DataType loss_arma(const vtype&, const vtype&) override;
    DataType gradient_(const vtype&, const vtype&, vtype*) override;
    void hessian_(const vtype&, const vtype&, vtype*) override;
  };
  
  template<typename DataType>
  class SyntheticLossVar2 : public LossFunction<DataType> {
  private:
    using vtype = typename LossFunction<DataType>::vtype;
    using mtype = typename LossFunction<DataType>::mtype;

  public:
    SyntheticLossVar2() = default;
    SyntheticLossVar2<DataType>* create() override { return new SyntheticLossVar2<DataType>(); }
  private:
#ifdef AUTODIFF
    autodiff::real loss_reverse(const ArrayXreal&, const ArrayXreal&) override;
#endif
    DataType loss_reverse_arma(const vtype&, const vtype&) override;
    DataType loss_arma(const vtype&, const vtype&) override;
    DataType gradient_(const vtype&, const vtype&, vtype*) override;
    void hessian_(const vtype&, const vtype&, vtype*) override;
  };

  template<typename DataType>
  class SquareLoss : public LossFunction<DataType> {
  private:
    using vtype = typename LossFunction<DataType>::vtype;
    using mtype = typename LossFunction<DataType>::mtype;

  public:
    SquareLoss() = default;
    SquareLoss<DataType>* create() override { return new SquareLoss<DataType>(); }
  private:
#ifdef AUTODIFF
    autodiff::real loss_reverse(const ArrayXreal&, const ArrayXreal&) override;
#endif
    DataType loss_reverse_arma(const vtype&, const vtype&) override;
    DataType loss_arma(const vtype&, const vtype&) override;
    DataType gradient_(const vtype&, const vtype&, vtype*) override;
    void hessian_(const vtype&, const vtype&, vtype*) override;
  };

  template<typename DataType>
  class PowerLoss : public LossFunction<DataType> {
  private:
    using vtype = typename LossFunction<DataType>::vtype;
    using mtype = typename LossFunction<DataType>::mtype;

  public:
    PowerLoss(float p) : p_{p} {}
    PowerLoss<DataType>* create() override { return new PowerLoss<DataType>{0.}; }
    PowerLoss<DataType>* create(float p) { return new PowerLoss<DataType>{p}; }
  private:
#ifdef AUTODIFF
    autodiff::real loss_reverse(const ArrayXreal&, const ArrayXreal&) override;
#endif
    DataType loss_reverse_arma(const vtype&, const vtype&) override;
    DataType loss_arma(const vtype&, const vtype&) override;
    DataType gradient_(const vtype&, const vtype&, vtype*) override;
    void hessian_(const vtype&, const vtype&, vtype*) override;

    float p_;
  };

  template<typename DataType>
  class CrossEntropyLoss : public LossFunction<DataType> {
  private:
    using vtype = typename LossFunction<DataType>::vtype;
    using mtype = typename LossFunction<DataType>::mtype;
    
  public:
    CrossEntropyLoss() = default;
    CrossEntropyLoss<DataType>* create() override { return new CrossEntropyLoss<DataType>(); }
  private:
#ifdef AUTODIFF
    autodiff::real loss_reverse(const ArrayXreal&, const ArrayXreal&) override;
#endif
    DataType loss_reverse_arma(const vtype&, const vtype&) override;
    DataType loss_arma(const vtype&, const vtype&) override;
    DataType gradient_(const vtype&, const vtype&, vtype*) override;
    void hessian_(const vtype&, const vtype&, vtype*) override;
  };

  struct ClassifierLossMapHash {
    std::size_t operator() (classifierLossFunction l) const { return static_cast<std::size_t>(l); }
  };

  template<typename T>
  std::unordered_map<classifierLossFunction, LossFunction<T>*, ClassifierLossMapHash> lossMap = 
    {
      {classifierLossFunction::BinomialDeviance,	new BinomialDevianceLoss<T>() },
      {classifierLossFunction::Savage,			new SavagetLoss<T>() },
      {classifierLossFunction::Exp,			new ExpLoss<T>() },
      {classifierLossFunction::Arctan,			new ArctanLoss<T>() },
      {classifierLossFunction::SyntheticLoss,		new SyntheticLoss<T>() },
      {classifierLossFunction::SyntheticLossVar1,	new SyntheticLossVar1<T>() },
      {classifierLossFunction::SyntheticLossVar2,	new SyntheticLossVar2<T>() },
      {classifierLossFunction::SquareLoss,		new SquareLoss<T>() },
      {classifierLossFunction::CrossEntropyLoss,	new CrossEntropyLoss<T>() }
    };

  template<typename T>
  LossFunction<T>* createLoss(classifierLossFunction loss, float p) {
    if (loss == classifierLossFunction::PowerLoss) {
      return new PowerLoss<T>{p};
    } else {
      return lossMap<T>[loss];
    }
  }

} // namespace ClassifierLossMeasures

#include "classifier_loss_impl.hpp"

#endif
