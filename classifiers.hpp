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

namespace Model_Traits {

  using AllClassifierArgs = std::tuple<std::size_t,	// (0) numClasses
				       std::size_t,	// (1) minLeafSize
				       double,		// (2) minGainSplit
				       std::size_t,	// (3) numTrees
				       std::size_t>;	// (4) maxDepth

  namespace ClassifierTypes {
    using RandomForestClassifierType = RandomForest<>;
    using DecisionTreeClassifierType = DecisionTree<>;

    // using DecisionTreeClassifierType = DecisionTree<GiniGain, BestBinaryNumericSplit>;
    // using DecisionTreeClassifierType = DecisionTree<GiniGain, BestBinaryNumericSplit, AllCategoricalSplit, AllDimensionSelect, true>;
    // using DecisionTreeClassifierType = DecisionTreeRegressor<MADGain>;
    // using DecisionTreeClassifierType = DecisionTreeRegressor<>;
    // using DecisionTreeClassifierType = DecisionTreeRegressor<MSEGain, BestBinaryNumericSplit, AllCategoricalSplit, AllDimensionSelect, true>;
    // using DecisionTreeClassifierType = DecisionTreeRegressor<InformationGain, BestBinaryNumericSplit, AllCategoricalSplit, AllDimensionSelect, true>;
  
  };
} // namespace Model_Traits

template<typename... Args>
class RandomForestClassifierBase : 
  public DiscreteClassifierBase<double,
				Model_Traits::ClassifierTypes::RandomForestClassifierType,
				Args...> {
public:
  RandomForestClassifierBase() = default;
  
  RandomForestClassifierBase(const mat& dataset,
			     rowvec& labels,
			     Args&&... args) :
    DiscreteClassifierBase<double, Model_Traits::ClassifierTypes::RandomForestClassifierType, Args...>(dataset, labels, std::forward<Args>(args)...) {}
  
};

class RandomForestClassifier : 
    public RandomForestClassifierBase<std::size_t, std::size_t, std::size_t> {

public:

  using Args = std::tuple<std::size_t, std::size_t, std::size_t>;

  RandomForestClassifier() = default;

  RandomForestClassifier(const mat& dataset,
			 rowvec& labels,
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
  public DiscreteClassifierBase<double,
				Model_Traits::ClassifierTypes::DecisionTreeClassifierType,
				Args...> {
public:
  DecisionTreeClassifierBase() = default;
  
  DecisionTreeClassifierBase(const mat& dataset,
			     rowvec& labels,
			     Args&&... args) :
    DiscreteClassifierBase<double, Model_Traits::ClassifierTypes::DecisionTreeClassifierType, Args...>(dataset, labels, std::forward<Args>(args)...) {}
};

class DecisionTreeClassifier : 
  public DecisionTreeClassifierBase<std::size_t, std::size_t, double, std::size_t> {

public:
  using Args = std::tuple<std::size_t, std::size_t, double, std::size_t>;

  DecisionTreeClassifier() = default;
  
  DecisionTreeClassifier(const mat& dataset,
			 rowvec& labels,
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

namespace Model_Traits {

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


////////////////////////////////////////////////////////
// CEREAL DEFINITIONS, REGISTRATIONS, OVERLOADS, ETC. //
////////////////////////////////////////////////////////

using DTCB = DecisionTreeClassifierBase<std::size_t, std::size_t, double, std::size_t>;
using RFCB = RandomForestClassifierBase<std::size_t, std::size_t, std::size_t>;

using DiscreteClassifierBaseDTC = DiscreteClassifierBase<double, 
							 Model_Traits::ClassifierTypes::DecisionTreeClassifierType,
							 std::size_t,
							 std::size_t,
							 double,
							 std::size_t>;
using DiscreteClassifierBaseRFC = DiscreteClassifierBase<double,
							 Model_Traits::ClassifierTypes::RandomForestClassifierType,
							 std::size_t,
							 std::size_t,
							 std::size_t>;

using ClassifierBaseDTC = ClassifierBase<double, Model_Traits::ClassifierTypes::DecisionTreeClassifierType>;
using ClassifierBaseRFC = ClassifierBase<double, Model_Traits::ClassifierTypes::RandomForestClassifierType>;


using ModelDTC = Model<double, Model_Traits::ClassifierTypes::DecisionTreeClassifierType>;
using ModelRFC = Model<double, Model_Traits::ClassifierTypes::RandomForestClassifierType>;

CEREAL_REGISTER_TYPE(ClassifierBaseDTC);
CEREAL_REGISTER_TYPE(ClassifierBaseRFC);

CEREAL_REGISTER_TYPE(DiscreteClassifierBaseDTC);
CEREAL_REGISTER_TYPE(DiscreteClassifierBaseRFC);

CEREAL_REGISTER_TYPE(DTCB);
CEREAL_REGISTER_TYPE(RFCB);

CEREAL_REGISTER_TYPE(DecisionTreeClassifier);
CEREAL_REGISTER_TYPE(RandomForestClassifier);

CEREAL_REGISTER_TYPE(ModelDTC);
CEREAL_REGISTER_TYPE(ModelRFC);			      	

CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelDTC, DecisionTreeClassifier);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelRFC, RandomForestClassifier);

CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelDTC, DTCB);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelRFC, RFCB);

CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelDTC, DiscreteClassifierBaseDTC);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelRFC, DiscreteClassifierBaseRFC);

CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelRFC, ClassifierBaseDTC);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ModelRFC, ClassifierBaseRFC);

CEREAL_REGISTER_POLYMORPHIC_RELATION(ClassifierBaseDTC, DecisionTreeClassifier);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ClassifierBaseRFC, RandomForestClassifier);

#endif
