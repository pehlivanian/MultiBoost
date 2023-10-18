#ifndef __CLASSIFIERS_HPP__
#define __CLASSIFIERS_HPP__

#include <utility>
#include <tuple>

#include <mlpack/core.hpp>
#include <mlpack/methods/decision_tree/decision_tree.hpp>
#include <mlpack/methods/decision_tree/decision_tree_regressor.hpp>
#include <mlpack/methods/decision_tree/information_gain.hpp>
#include <mlpack/methods/decision_tree/gini_gain.hpp>
#include <mlpack/methods/decision_tree/random_dimension_select.hpp>
#include <mlpack/methods/decision_tree/multiple_random_dimension_select.hpp>
#include <mlpack/methods/random_forest/random_forest.hpp>

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

#include "classifier.hpp"

using namespace mlpack;
using namespace mlpack::tree;
using namespace mlpack::data;
using namespace mlpack::util;

class DecisionTreeClassifier;
class RandomForestClassifier;

namespace Model_Traits {

  using AllClassifierArgs = std::tuple<std::size_t,	// (0) numClasses
				       std::size_t,	// (1) minLeafSize
				       double,		// (2) minGainSplit
				       std::size_t,	// (3) numTrees
				       std::size_t>;	// (4) maxDepth

  namespace ClassifierTypes {
    using RandomForestClassifierType = RandomForest<>;
    using DecisionTreeClassifierType = DecisionTree<>;

    // [==========--===========]
    // [============--=========]
    // [==============--=======]
    // Possible options
    // [==========--===========]
    // [========--=============]
    // [======--===============]
    // using DecisionTreeClassifierType = DecisionTree<GiniGain, BestBinaryNumericSplit>;
    // using DecisionTreeClassifierType = DecisionTree<GiniGain, BestBinaryNumericSplit, AllCategoricalSplit, AllDimensionSelect, true>;
    // using DecisionTreeClassifierType = DecisionTreeRegressor<MADGain>;
    // using DecisionTreeClassifierType = DecisionTreeRegressor<>;
    // using DecisionTreeClassifierType = DecisionTreeRegressor<MSEGain, BestBinaryNumericSplit, AllCategoricalSplit, AllDimensionSelect, true>;
    // using DecisionTreeClassifierType = DecisionTreeRegressor<InformationGain, BestBinaryNumericSplit, AllCategoricalSplit, AllDimensionSelect, true>;
  
  };

  template<typename T>
  struct classifier_traits {
    using datatype = double;
    using integrallabeltype = std::size_t;
    using model = ClassifierTypes::DecisionTreeClassifierType;
    using modelArgs = std::tuple<std::size_t, std::size_t, double, std::size_t>;
  };

  template<>
  struct classifier_traits<DecisionTreeClassifier> {
    using datatype = double;
    using integrallabeltype = std::size_t;
    using model = ClassifierTypes::DecisionTreeClassifierType;
    using modelArgs = std::tuple<std::size_t, std::size_t, double, std::size_t>;
  };

  template<>
  struct classifier_traits<RandomForestClassifier> {
    using datatype = double;
    using integrallabeltype = std::size_t;
    using model = ClassifierTypes::RandomForestClassifierType;
    using modelArgs = std::tuple<std::size_t, std::size_t, std::size_t>;
  };
} // namespace Model_Traits

template<typename... Args>
class RandomForestClassifierBase : 
  public DiscreteClassifierBase<Model_Traits::classifier_traits<RandomForestClassifier>::datatype,
				Model_Traits::ClassifierTypes::RandomForestClassifierType,
				Args...> {
public:
  using DataType = Model_Traits::classifier_traits<RandomForestClassifier>::datatype;
  using ClassifierType = Model_Traits::ClassifierTypes::RandomForestClassifierType;

  RandomForestClassifierBase() = default;
  RandomForestClassifierBase(const RandomForestClassifierBase&) = default;
  
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
  using DataType = Model_Traits::classifier_traits<RandomForestClassifier>::datatype;

  RandomForestClassifier() = default;
  RandomForestClassifier(const RandomForestClassifier&) = default;

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
  public DiscreteClassifierBase<Model_Traits::classifier_traits<DecisionTreeClassifier>::datatype,
				Model_Traits::ClassifierTypes::DecisionTreeClassifierType,
				Args...> {
public:

  using DataType = Model_Traits::classifier_traits<DecisionTreeClassifier>::datatype;
  using ClassifierType = Model_Traits::ClassifierTypes::DecisionTreeClassifierType;

  DecisionTreeClassifierBase() = default;
  DecisionTreeClassifierBase(const DecisionTreeClassifierBase&) = default;
  
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
  using DataType = Model_Traits::classifier_traits<DecisionTreeClassifier>::datatype;

  DecisionTreeClassifier() = default;
  DecisionTreeClassifier(const DecisionTreeClassifier&) = default;
  
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

////////////////////////////////////////////////////////
// CEREAL DEFINITIONS, REGISTRATIONS, OVERLOADS, ETC. //
////////////////////////////////////////////////////////

using DTCB = DecisionTreeClassifierBase<std::size_t, std::size_t, double, std::size_t>;
using RFCB = RandomForestClassifierBase<std::size_t, std::size_t, std::size_t>;

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

using ClassifierBaseDTCD = ClassifierBase<double, Model_Traits::ClassifierTypes::DecisionTreeClassifierType>;
using ClassifierBaseDTCF = ClassifierBase<float,  Model_Traits::ClassifierTypes::DecisionTreeClassifierType>;
using ClassifierBaseRFCD = ClassifierBase<double, Model_Traits::ClassifierTypes::RandomForestClassifierType>;
using ClassifierBaseRFCF = ClassifierBase<float,  Model_Traits::ClassifierTypes::RandomForestClassifierType>;


using ModelDTCD = Model<double, Model_Traits::ClassifierTypes::DecisionTreeClassifierType>;
using ModelDTCF = Model<float,  Model_Traits::ClassifierTypes::DecisionTreeClassifierType>;
using ModelRFCD = Model<double, Model_Traits::ClassifierTypes::RandomForestClassifierType>;
using ModelRFCF = Model<float,  Model_Traits::ClassifierTypes::RandomForestClassifierType>;

CEREAL_REGISTER_TYPE(ClassifierBaseDTCD);
CEREAL_REGISTER_TYPE(ClassifierBaseDTCF);
CEREAL_REGISTER_TYPE(ClassifierBaseRFCD);
CEREAL_REGISTER_TYPE(ClassifierBaseRFCF);

CEREAL_REGISTER_TYPE(DiscreteClassifierBaseDTCD);
CEREAL_REGISTER_TYPE(DiscreteClassifierBaseDTCF);
CEREAL_REGISTER_TYPE(DiscreteClassifierBaseRFCD);
CEREAL_REGISTER_TYPE(DiscreteClassifierBaseRFCF);

CEREAL_REGISTER_TYPE(DTCB);
CEREAL_REGISTER_TYPE(RFCB);

CEREAL_REGISTER_TYPE(DecisionTreeClassifier);
CEREAL_REGISTER_TYPE(RandomForestClassifier);

CEREAL_REGISTER_TYPE(ModelDTCD);
CEREAL_REGISTER_TYPE(ModelDTCF);
CEREAL_REGISTER_TYPE(ModelRFCD);
CEREAL_REGISTER_TYPE(ModelRFCF);

CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelDTCD, DecisionTreeClassifier);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelDTCF, DecisionTreeClassifier);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelRFCD, RandomForestClassifier);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelRFCF, RandomForestClassifier);

CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelDTCD, DTCB);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelDTCF, DTCB);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelRFCD, RFCB);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelRFCF, RFCB);

CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelDTCD, DiscreteClassifierBaseDTCD);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelDTCF, DiscreteClassifierBaseDTCF);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelRFCD, DiscreteClassifierBaseRFCD);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelRFCF, DiscreteClassifierBaseRFCF);

CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelRFCD, ClassifierBaseDTCD);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelRFCF, ClassifierBaseDTCF);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelRFCD, ClassifierBaseRFCD);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelRFCF, ClassifierBaseRFCF);

CEREAL_REGISTER_POLYMORPHIC_RELATION(ClassifierBaseDTCD, DecisionTreeClassifier);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ClassifierBaseDTCF, DecisionTreeClassifier);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ClassifierBaseRFCD, RandomForestClassifier);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ClassifierBaseRFCF, RandomForestClassifier);

#endif
