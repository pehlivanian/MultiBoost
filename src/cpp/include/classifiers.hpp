#ifndef __CLASSIFIERS_HPP__
#define __CLASSIFIERS_HPP__

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

};


////////////////////////////////////////////////////////
// CEREAL DEFINITIONS, REGISTRATIONS, OVERLOADS, ETC. //
////////////////////////////////////////////////////////

using DTCB = DecisionTreeClassifierBase<std::size_t, std::size_t, double, std::size_t>;
using RFCB = RandomForestClassifierBase<std::size_t, std::size_t, std::size_t>;
using CTCB = ConstantTreeClassifierBase<>;

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

using ClassifierBaseDTCD = ClassifierBase<double, Model_Traits::ClassifierTypes::DecisionTreeClassifierType>;
using ClassifierBaseDTCF = ClassifierBase<float,  Model_Traits::ClassifierTypes::DecisionTreeClassifierType>;
using ClassifierBaseRFCD = ClassifierBase<double, Model_Traits::ClassifierTypes::RandomForestClassifierType>;
using ClassifierBaseRFCF = ClassifierBase<float,  Model_Traits::ClassifierTypes::RandomForestClassifierType>;
using ClassifierBaseCTCD = ClassifierBase<double, Model_Traits::ClassifierTypes::ConstantTreeClassifierType>;
using ClassifierBaseCTCF = ClassifierBase<float,  Model_Traits::ClassifierTypes::ConstantTreeClassifierType>;

using ModelD = Model<double>;
using ModelF = Model<float>;

CEREAL_REGISTER_TYPE(ClassifierBaseDTCD);
CEREAL_REGISTER_TYPE(ClassifierBaseDTCF);
CEREAL_REGISTER_TYPE(ClassifierBaseRFCD);
CEREAL_REGISTER_TYPE(ClassifierBaseRFCF);
CEREAL_REGISTER_TYPE(ClassifierBaseCTCD);
CEREAL_REGISTER_TYPE(ClassifierBaseCTCF);

CEREAL_REGISTER_TYPE(DiscreteClassifierBaseDTCD);
CEREAL_REGISTER_TYPE(DiscreteClassifierBaseDTCF);
CEREAL_REGISTER_TYPE(DiscreteClassifierBaseRFCD);
CEREAL_REGISTER_TYPE(DiscreteClassifierBaseRFCF);
CEREAL_REGISTER_TYPE(DiscreteClassifierBaseCTCD);
CEREAL_REGISTER_TYPE(DiscreteClassifierBaseCTCF);

CEREAL_REGISTER_TYPE(DTCB);
CEREAL_REGISTER_TYPE(RFCB);
CEREAL_REGISTER_TYPE(CTCB);

CEREAL_REGISTER_TYPE(DecisionTreeClassifier);
CEREAL_REGISTER_TYPE(RandomForestClassifier);
CEREAL_REGISTER_TYPE(ConstantTreeClassifier);

CEREAL_REGISTER_TYPE(ModelD);
CEREAL_REGISTER_TYPE(ModelF);

CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelD, DecisionTreeClassifier);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelF, DecisionTreeClassifier);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelD, RandomForestClassifier);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelF, RandomForestClassifier);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelD, ConstantTreeClassifier);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelF, ConstantTreeClassifier);

CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelD, DTCB);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelF, DTCB);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelD, RFCB);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelF, RFCB);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelD, CTCB);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelF, CTCB);

CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelD, DiscreteClassifierBaseDTCD);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelF, DiscreteClassifierBaseDTCF);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelD, DiscreteClassifierBaseRFCD);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelF, DiscreteClassifierBaseRFCF);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelD, DiscreteClassifierBaseCTCD);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelF, DiscreteClassifierBaseCTCF);

CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelD, ClassifierBaseDTCD);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelF, ClassifierBaseDTCF);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelD, ClassifierBaseRFCD);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelF, ClassifierBaseRFCF);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelD, ClassifierBaseCTCD);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelF, ClassifierBaseCTCF);

CEREAL_REGISTER_POLYMORPHIC_RELATION(ClassifierBaseDTCD, DecisionTreeClassifier);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ClassifierBaseDTCF, DecisionTreeClassifier);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ClassifierBaseRFCD, RandomForestClassifier);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ClassifierBaseRFCF, RandomForestClassifier);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ClassifierBaseRFCD, ConstantTreeClassifier);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ClassifierBaseRFCF, ConstantTreeClassifier);

#endif
