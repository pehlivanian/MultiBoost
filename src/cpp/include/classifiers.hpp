#ifndef __CLASSIFIERS_HPP__
#define __CLASSIFIERS_HPP__

#define UNUSED(expr) do { (void)(expr); } while (0)

#include <utility>
#include <tuple>

#include <cereal/types/polymorphic.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/archives/portable_binary.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/archives/xml.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/map.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/access.hpp>

#include "model_traits.hpp"
#include "classifier.hpp"

template<typename DecoratedType, typename... Args>
class DecoratorClassifierBase :
  public DiscreteClassifierBase<typename Model_Traits::model_traits<DecoratedType>::datatype,
				typename Model_Traits::model_traits<DecoratedType>::model,
				Args...> {

public:
  using DataType = typename Model_Traits::model_traits<DecisionTreeClassifier>::datatype;
  using ClassifierType = typename Model_Traits::model_traits<DecoratedType>::model;

  DecoratorClassifierBase() = default;

  DecoratorClassifierBase(const Mat<DataType>& dataset,
			  Row<DataType>& labels,
			  Args&&... args) :
    DiscreteClassifierBase<DataType,
			   ClassifierType,
			   Args...>(dataset, labels, std::forward<Args>(args)...) 
  {}

  DecoratorClassifierBase(const Mat<DataType>& dataset,
			  Row<DataType>& labels,
			  Args const&... args) :
    DiscreteClassifierBase<DataType,
			   ClassifierType,
			   Args...>(dataset, labels, args...)
  {}

  DecoratorClassifierBase(const Mat<DataType>& dataset,
			  Row<DataType>& labels,
			  Row<DataType>& weights,
			  Args&&... args) :
    DiscreteClassifierBase<DataType,
			   ClassifierType,
			   Args...>(dataset, labels, weights, std::forward<Args>(args)...) {}

  DecoratorClassifierBase(const Mat<DataType>& dataset,
			  Row<DataType>& labels,
			  Row<DataType>& weights,
			  Args const&... args) :
    DiscreteClassifierBase<DataType,
			   ClassifierType,
			   Args...>(dataset, labels, weights, const_cast<Args&>(args)...)
  {}

};

template<typename DecoratedType, typename... Args>
class DecoratorClassifier : public DecoratorClassifierBase<DecoratedType, Args...> {

public:
  using tupArgs = typename Model_Traits::model_traits<DecisionTreeClassifier>::modelArgs;
  using DataType = typename Model_Traits::model_traits<DecisionTreeClassifier>::datatype;

  DecoratorClassifier() = default;

  DecoratorClassifier(const Mat<DataType>& dataset,
		      Row<DataType>& labels,
		      Args &&... args) :
    DecoratorClassifierBase<DecoratedType, Args...>(dataset,
						    labels,
						    std::forward<Args>(args)...)
  {}

  DecoratorClassifier(const Mat<DataType>& dataset,
		      Row<DataType>& labels,
		      Args const&... args) :
    DecoratorClassifierBase<DecoratedType, Args...>(dataset,
						    labels,
						    args...)
  {}

  DecoratorClassifier(const Mat<DataType>& dataset,
		      Row<DataType>& labels,
		      Row<DataType>& weights,
		      Args &&... args) :
    DecoratorClassifierBase<DecoratedType, Args...>(dataset,
						    labels,
						    weights,
						    std::forward<Args>(args)...)
  {}

  DecoratorClassifier(const Mat<DataType>& dataset,
		      Row<DataType>& labels,
		      Row<DataType>& weights,
		      Args const&... args) :
    DecoratorClassifierBase<DecoratedType, Args...>(dataset,
						    labels,
						    weights,
						    args...)
  {}
	
  static tupArgs _args(const Model_Traits::AllClassifierArgs& p) {
    return DecoratedType::_args(p);
  }
  
};

template<typename DecoratedType, typename... DecoratedArgs>
class NegativeFeedback : public DecoratorClassifier<DecoratedType, DecoratedArgs...> {

public:
  using Args = typename Model_Traits::model_traits<DecoratedType>::modelArgs;
  using DataType = typename Model_Traits::model_traits<DecoratedType>::datatype;
  
  NegativeFeedback() : beta_{.001}, iterations_{3} {}

  NegativeFeedback(float beta, std::size_t iterations) :
    beta_{beta},
    iterations_{iterations}
  {}

  NegativeFeedback(const Mat<DataType>& dataset,
		   Row<DataType>& labels,
		   DecoratedArgs &&... args) : 
    DecoratorClassifier<DecoratedType, DecoratedArgs...>(dataset,
							 labels,
							 std::forward<DecoratedArgs>(args)...)
  {}

  NegativeFeedback(const Mat<DataType>& dataset,
		   Row<DataType>& labels,
		   DecoratedArgs const&... args) :
    DecoratorClassifier<DecoratedType, DecoratedArgs...>(dataset,
							 labels,
							 args...)
  {}

  NegativeFeedback(const Mat<DataType>& dataset,
		   Row<DataType>& labels,
		   Row<DataType>& weights,
		   DecoratedArgs &&... args) :
    DecoratorClassifier<DecoratedType, DecoratedArgs...>(dataset,
							 labels,
							 weights,
							 std::forward<DecoratedArgs>(args)...)
  {}

  NegativeFeedback(const Mat<DataType>& dataset,
		   Row<DataType>& labels,
		   Row<DataType>& weights,
		   DecoratedArgs const&... args) :
    DecoratorClassifier<DecoratedType, DecoratedArgs...>(dataset,
							 labels,
							 weights,
							 args...)
  {}

  template<typename... Ts>
  void setRootClassifier(std::unique_ptr<DecoratedType>&,
			 const Mat<DataType>&,
			 Row<DataType>&,
			 std::tuple<Ts...> const&);

  template<typename... Ts>
  void setRootClassifier(std::unique_ptr<DecoratedType>&,
			 const Mat<DataType>&,
			 Row<DataType>&,
			 Row<DataType>&,
			 std::tuple<Ts...> const&);

  static Args _args(const Model_Traits::AllClassifierArgs& p) {
    return DecoratedType::_args(p);
  }

  template<class Archive>
  void serialize(Archive &ar) {
    ar(beta_);
    ar(iterations_);
  }

private:
  float beta_;
  std::size_t iterations_;

};

template<typename... Args>
class RandomForestClassifierBase : 
  public DiscreteClassifierBase<Model_Traits::model_traits<RandomForestClassifier>::datatype,
				Model_Traits::ClassifierTypes::RandomForestClassifierType,
				Args...> {
public:
  using DataType = Model_Traits::model_traits<RandomForestClassifier>::datatype;
  using ClassifierType = Model_Traits::ClassifierTypes::RandomForestClassifierType;

  RandomForestClassifierBase() = default;
  
  RandomForestClassifierBase(const Mat<DataType>& dataset,
			     Row<DataType>& labels,
			     Args&&... args) :
    DiscreteClassifierBase<DataType, 
			   ClassifierType, 
			   Args...>(dataset, labels, std::forward<Args>(args)...) {}

  RandomForestClassifierBase(const Mat<DataType>& dataset,
			     Row<DataType>& labels,
			     Row<DataType>& weights,
			     Args&&... args) :
    DiscreteClassifierBase<DataType, 
			   ClassifierType, 
			   Args...>(dataset, labels, weights, std::forward<Args>(args)...) {}
  
};

class RandomForestClassifier : 
  public RandomForestClassifierBase<std::size_t, std::size_t, std::size_t> {

public:

  using Args = std::tuple<std::size_t, std::size_t, std::size_t>;
  using DataType = Model_Traits::model_traits<RandomForestClassifier>::datatype;

  RandomForestClassifier() = default;

  RandomForestClassifier(const Mat<DataType>& dataset,
			 Row<DataType>& labels,
			 std::size_t numClasses=1,
			 std::size_t numTrees=10,
			 std::size_t minLeafSize=2) :
    RandomForestClassifierBase<std::size_t, std::size_t, std::size_t>(dataset, 
								      labels, 
								      std::move(numClasses),
								      std::move(numTrees),
								      std::move(minLeafSize)) {}
  
  RandomForestClassifier(const Mat<DataType>& dataset,
			 Row<DataType>& labels,
			 Row<DataType>& weights,
			 std::size_t numClasses=1,
			 std::size_t numTrees=10,
			 std::size_t minLeafSize=2) :
    RandomForestClassifierBase<std::size_t, std::size_t, std::size_t>(dataset,
								      labels,
								      weights,
								      std::move(numClasses),
								      std::move(numTrees),
								      std::move(minLeafSize)) {}

  static Args _args(const Model_Traits::AllClassifierArgs& p) {
    return std::make_tuple(std::get<0>(p),	// numClasses
			   std::get<3>(p),	// numTrees
			   std::get<1>(p));	// minLeafSize
  }

};

template<typename... Args>
class DecisionTreeClassifierBase : 
  public DiscreteClassifierBase<Model_Traits::model_traits<DecisionTreeClassifier>::datatype,
				Model_Traits::ClassifierTypes::DecisionTreeClassifierType,
				Args...> {
public:

  using DataType = Model_Traits::model_traits<DecisionTreeClassifier>::datatype;
  using ClassifierType = Model_Traits::ClassifierTypes::DecisionTreeClassifierType;

  DecisionTreeClassifierBase() = default;
  
  DecisionTreeClassifierBase(const Mat<DataType>& dataset,
			     Row<DataType>& labels,
			     Args&&... args) :
    DiscreteClassifierBase<DataType, 
			   ClassifierType, 
			   Args...>(dataset, labels, std::forward<Args>(args)...) {}

  DecisionTreeClassifierBase(const Mat<DataType>& dataset,
			     Row<DataType>& labels,
			     Row<DataType>& weights,
			     Args&&... args) :
    DiscreteClassifierBase<DataType, 
			   ClassifierType, 
			   Args...>(dataset, labels, weights, std::forward<Args>(args)...) {}
};

class DecisionTreeClassifier : 
  public DecisionTreeClassifierBase<std::size_t, std::size_t, double, std::size_t> {

public:
  using Args = std::tuple<std::size_t, std::size_t, double, std::size_t>;
  using DataType = Model_Traits::model_traits<DecisionTreeClassifier>::datatype;

  DecisionTreeClassifier() = default;
  
  DecisionTreeClassifier(const Mat<DataType>& dataset,
			 Row<DataType>& labels,
			 std::size_t numClasses,
			 std::size_t minLeafSize,
			 double minGainSplit,
			 std::size_t maxDepth) :
    DecisionTreeClassifierBase<std::size_t, std::size_t, double, std::size_t>(dataset,
									      labels,
									      std::move(numClasses),
									      std::move(minLeafSize),
									      std::move(minGainSplit),
									      std::move(maxDepth))
  {}

  DecisionTreeClassifier(const Mat<DataType>& dataset,
			 Row<DataType>& labels,
			 Row<DataType>& weights,
			 std::size_t numClasses,
			 std::size_t minLeafSize,
			 double minGainSplit,
			 std::size_t maxDepth) :
    DecisionTreeClassifierBase<std::size_t, std::size_t, double, std::size_t>(dataset,
									      labels,
									      weights,
									      std::move(numClasses),
									      std::move(minLeafSize),
									      std::move(minGainSplit),
									      std::move(maxDepth))
  {}
  
  static Args _args(const Model_Traits::AllClassifierArgs& p) {
    return std::make_tuple(std::get<0>(p),	// numClasses
			   std::get<1>(p),	// minLeafSize
			   std::get<2>(p),	// minGainSplit
			   std::get<4>(p));	// maxDepth
  }
};

template<typename... Args>
class ConstantTreeClassifierBase :
  public DiscreteClassifierBase<Model_Traits::model_traits<ConstantTreeClassifier>::datatype,
				Model_Traits::ClassifierTypes::ConstantTreeClassifierType,
				Args...> {
public:
  using DataType = Model_Traits::model_traits<ConstantTreeClassifier>::datatype;
  using ClassifierType = Model_Traits::ClassifierTypes::ConstantTreeClassifierType;
  
  ConstantTreeClassifierBase() = default;

  ConstantTreeClassifierBase(const Mat<DataType>& dataset,
			     Row<DataType>& labels,
			     Args&&... args) :
    DiscreteClassifierBase<DataType,
			   ClassifierType,
			   Args...>(dataset, labels, std::forward<Args>(args)...) {}
  
  ConstantTreeClassifierBase(const Mat<DataType>& dataset,
			     Row<DataType>& labels,
			     Row<DataType>& weights,
			     Args&&... args) :
    DiscreteClassifierBase<DataType,
			   ClassifierType,
			   Args...>(dataset, labels, weights, std::forward<Args>(args)...) {}
};
  
class ConstantTreeClassifier :
  public ConstantTreeClassifierBase<> {
public:
  using DataType = Model_Traits::model_traits<ConstantTreeClassifier>::datatype;
    
  ConstantTreeClassifier() = default;

  ConstantTreeClassifier(const Mat<DataType>& dataset,
			 Row<DataType>& labels) :
    ConstantTreeClassifierBase<>(dataset,
				 labels)
  {}
  
  ConstantTreeClassifier(const Mat<DataType>& dataset,
			 Row<DataType>& labels,
			 Row<DataType>& weights) :
    ConstantTreeClassifierBase<>(dataset,
				 labels,
				 weights)
  {}

};

////////////////////////////////////////////////////////
// CEREAL DEFINITIONS, REGISTRATIONS, OVERLOADS, ETC. //
////////////////////////////////////////////////////////

// LEVEL 3 using directives
using DTCB  = DecisionTreeClassifierBase<std::size_t, std::size_t, double, std::size_t>;
using RFCB  = RandomForestClassifierBase<std::size_t, std::size_t, std::size_t>;
using CTCB  = ConstantTreeClassifierBase<>;

// LEVEL 2 using directives
using DiscreteClassifierBaseDTCD = DiscreteClassifierBase<double, 
							  Model_Traits::ClassifierTypes::DecisionTreeClassifierType,
							  std::size_t,
							  std::size_t,
							  double,
							  std::size_t>;

using DiscreteClassifierBaseDTCF = DiscreteClassifierBase<float, 
							  Model_Traits::ClassifierTypes::DecisionTreeClassifierType,
							  std::size_t,
							  std::size_t,
							  double,
							  std::size_t>;

using DiscreteClassifierBaseRFCD = DiscreteClassifierBase<double,
							  Model_Traits::ClassifierTypes::RandomForestClassifierType,
							  std::size_t,
							  std::size_t,
							  std::size_t>;

using DiscreteClassifierBaseRFCF = DiscreteClassifierBase<float,
							  Model_Traits::ClassifierTypes::RandomForestClassifierType,
							  std::size_t,
							  std::size_t,
							  std::size_t>;

using DiscreteClassifierBaseCTCD = DiscreteClassifierBase<double,
							  Model_Traits::ClassifierTypes::ConstantTreeClassifierType>;

using DiscreteClassifierBaseCTCF = DiscreteClassifierBase<float,
							  Model_Traits::ClassifierTypes::ConstantTreeClassifierType>;

// LEVEL 1 using directives
using ClassifierBaseDTCD  = ClassifierBase<double, Model_Traits::ClassifierTypes::DecisionTreeClassifierType>;
using ClassifierBaseDTCF  = ClassifierBase<float,  Model_Traits::ClassifierTypes::DecisionTreeClassifierType>;
using ClassifierBaseRFCD  = ClassifierBase<double, Model_Traits::ClassifierTypes::RandomForestClassifierType>;
using ClassifierBaseRFCF  = ClassifierBase<float,  Model_Traits::ClassifierTypes::RandomForestClassifierType>;
using ClassifierBaseCTCD  = ClassifierBase<double, Model_Traits::ClassifierTypes::ConstantTreeClassifierType>;
using ClassifierBaseCTCF  = ClassifierBase<float,  Model_Traits::ClassifierTypes::ConstantTreeClassifierType>;

// LEVEL 0 using directives
using ModelD = Model<double>;
using ModelF = Model<float>;

/*
// Decorator class using directives
using DC5  = NegativeFeedback<DecisionTreeClassifier, std::size_t, std::size_t, double, std::size_t>;
using DC4  = DecoratorClassifier<DecisionTreeClassifier, std::size_t, std::size_t, double, std::size_t>;
using DC3  = DecoratorClassifierBase<DecisionTreeClassifier, std::size_t, std::size_t, double, std::size_t>;
using DC2D = DiscreteClassifierBase<double,
Model_Traits::ClassifierTypes::NegativeFeedbackDecisionTreeClassifierType,
std::size_t,
std::size_t,
double,
std::size_t>;
using DC2F = DiscreteClassifierBase<float,
Model_Traits::ClassifierTypes::NegativeFeedbackDecisionTreeClassifierType,
std::size_t,
std::size_t,
double,
std::size_t>;
using DC1D = ClassifierBase<double, Model_Traits::ClassifierTypes::NegativeFeedbackDecisionTreeClassifierType>;
using DC1F = ClassifierBase<float, Model_Traits::ClassifierTypes::NegativeFeedbackDecisionTreeClassifierType>;
*/


// LEVEL 0 ~ Model<DataType>
// No registration needed

// LEVEL 1 registrations
CEREAL_REGISTER_TYPE(ClassifierBaseDTCD);
CEREAL_REGISTER_TYPE(ClassifierBaseDTCF);
CEREAL_REGISTER_TYPE(ClassifierBaseRFCD);
CEREAL_REGISTER_TYPE(ClassifierBaseRFCF);
CEREAL_REGISTER_TYPE(ClassifierBaseCTCD);
CEREAL_REGISTER_TYPE(ClassifierBaseCTCF);

/*
// LEVEL 1 Decorator class registrations
CEREAL_REGISTER_TYPE(DC1D);
CEREAL_REGISTER_TYPE(DC1F);
*/

// LEVEL 2 registrations
CEREAL_REGISTER_TYPE(DiscreteClassifierBaseDTCD);
CEREAL_REGISTER_TYPE(DiscreteClassifierBaseDTCF);
CEREAL_REGISTER_TYPE(DiscreteClassifierBaseRFCD);
CEREAL_REGISTER_TYPE(DiscreteClassifierBaseRFCF);
CEREAL_REGISTER_TYPE(DiscreteClassifierBaseCTCD);
CEREAL_REGISTER_TYPE(DiscreteClassifierBaseCTCF);

/*
// LEVEL 2 Decorator registrations
CEREAL_REGISTER_TYPE(DC2D);
CEREAL_REGISTER_TYPE(DC2F);
*/

// LEVEL 3 registrations
CEREAL_REGISTER_TYPE(DTCB);
CEREAL_REGISTER_TYPE(RFCB);
CEREAL_REGISTER_TYPE(CTCB);

/*
// LEVEL 3 Decorator registrations
CEREAL_REGISTER_TYPE(DC3);
*/

// LEVEL 4 registrations
CEREAL_REGISTER_TYPE(DecisionTreeClassifier);
CEREAL_REGISTER_TYPE(RandomForestClassifier);
CEREAL_REGISTER_TYPE(ConstantTreeClassifier);

/*
// LEVEL 4 Decorator registrations
CEREAL_REGISTER_TYPE(DC4);
*/

/*
// LEVEL 5 Decorator registrations
CEREAL_REGISTER_TYPE(DC5);
*/

// LEVEL 0 registrations
CEREAL_REGISTER_TYPE(ModelD);
CEREAL_REGISTER_TYPE(ModelF);

// LEVEL 4 polymorphic relations
CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelD, DecisionTreeClassifier);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelF, DecisionTreeClassifier);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelD, RandomForestClassifier);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelF, RandomForestClassifier);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelD, ConstantTreeClassifier);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelF, ConstantTreeClassifier);

/*
// LEVEL 5 Decorator polymorphic relations
CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelD, DC5);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelF, DC5);
*/

/*
// LEVEL 4 Decorator polymorphic relations
CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelD, DC4);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelF, DC4);
*/

// LEVEL 3 polymorphic relations
CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelD, DTCB);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelF, DTCB);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelD, RFCB);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelF, RFCB);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelD, CTCB);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelF, CTCB);

/*
// LEVEL 3 Decorator polymorphic relations
CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelD, DC3);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelF, DC3);
*/

// LEVEL 2 polymorphic relations
CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelD, DiscreteClassifierBaseDTCD);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelF, DiscreteClassifierBaseDTCF);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelD, DiscreteClassifierBaseRFCD);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelF, DiscreteClassifierBaseRFCF);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelD, DiscreteClassifierBaseCTCD);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelF, DiscreteClassifierBaseCTCF);

/*
// LEVEL 2 Decorator polymorphic relations
CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelD, DC2D);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelF, DC2F);
*/

// LEVEL 1 polymorphic relations
CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelD, ClassifierBaseDTCD);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelF, ClassifierBaseDTCF);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelD, ClassifierBaseRFCD);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelF, ClassifierBaseRFCF);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelD, ClassifierBaseCTCD);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelF, ClassifierBaseCTCF);

/*
// LEVEL 1 Decorator polymorphic relations
CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelD, DC1D);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelF, DC1F);
*/

#endif
