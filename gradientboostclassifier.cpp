#include "gradientboostclassifier.hpp"

/* 
   --- <.> --- <.> --- <.>
   OLD SERIALIZATION MODEL
   --- <.> --- <.> --- <.>   
*/

/*
  using DTC = ClassifierTypes::DecisionTreeClassifierType;
  using CTC = ClassifierTypes::DecisionTreeRegressorType;
  using RFC = ClassifierTypes::RandomForestClassifierType;

  using DiscreteClassifierBaseDTC = DiscreteClassifierBase<double, 
  DTC, 
  std::size_t,
  std::size_t,
  double,
  std::size_t>;
  using DiscreteClassifierBaseRFC = DiscreteClassifierBase<double,
  RFC,
  std::size_t,
  std::size_t,
  std::size_t>;
  using ContinuousClassifierBaseD = ContinuousClassifierBase<double, 
  CTC,
  unsigned long,
  double,
  unsigned long>;
  using ClassifierBaseDD = ClassifierBase<double, DTC>;
  using ClassifierBaseRD = ClassifierBase<double, RFC>;
  using ClassifierBaseCD = ClassifierBase<double, CTC>;

  using DecisionTreeRegressorClassifierBaseLDL = DecisionTreeRegressorClassifier<unsigned long, double, unsigned long>;
  using DecisionTreeClassifierBaseLLDL = DecisionTreeClassifier<unsigned long, unsigned long, double, unsigned long>;
  using RandomForestClassifierBaseLLL = RandomForestClassifier<unsigned long, unsigned long, unsigned long>;

  using GradientBoostClassifierDTC = GradientBoostClassifier<DecisionTreeClassifier>);
  using GradientBoostClassifierRFC = GradientBoostClassifier<RandomForestClassifier>);
  using GradientBoostClassifierCTC = GradientBoostClassifier<DecisionTreeRegressorClassifier>);

  // Register class with cereal
  CEREAL_REGISTER_TYPE(DiscreteClassifierBaseDTC);
  CEREAL_REGISTER_TYPE(DiscreteClassifierBaseRFC);
  CEREAL_REGISTER_TYPE(ContinuousClassifierBaseD);

  CEREAL_REGISTER_TYPE(DecisionTreeClassifierBaseLLDL);
  CEREAL_REGISTER_TYPE(RandomForestClassifierBaseLLL);
  CEREAL_REGISTER_TYPE(DecisionTreeRegressorClassifierBaseLDL);

  CEREAL_REGISTER_TYPE(GradientBoostClassifierDTC);
  CEREAL_REGISTER_TYPE(GradientBoostClassifierRFC);
  CEREAL_REGISTER_TYPE(GradientBoostClassifierCTC);


  // Register class with cereal
  CEREAL_REGISTER_TYPE(DiscreteClassifierBaseDTC);
  CEREAL_REGISTER_TYPE(DiscreteClassifierBaseRFC);
  CEREAL_REGISTER_TYPE(ContinuousClassifierBaseD);

  CEREAL_REGISTER_TYPE(DecisionTreeClassifierLLDL);
  CEREAL_REGISTER_TYPE(RandomForestClassifierLLL);
  CEREAL_REGISTER_TYPE(DecisionTreeRegressorClassifierLDL);

  CEREAL_REGISTER_TYPE(GradientBoostClassifier<DecisionTreeClassifierLLDL>);
  CEREAL_REGISTER_TYPE(GradientBoostClassifier<DecisionTreeRegressorClassifierLDL>);

  // Register class hierarchy with cereal
  CEREAL_REGISTER_POLYMORPHIC_RELATION(ClassifierBaseDD, DecisionTreeClassifierLLDL);
  CEREAL_REGISTER_POLYMORPHIC_RELATION(ClassifierBaseRD, RandomForestClassifierLLL);
  CEREAL_REGISTER_POLYMORPHIC_RELATION(ClassifierBaseCD, DecisionTreeRegressorClassifierLDL);

  CEREAL_REGISTER_POLYMORPHIC_RELATION(ClassifierBaseDD, DiscreteClassifierBaseDTC);
  CEREAL_REGISTER_POLYMORPHIC_RELATION(ClassifierBaseRD, DiscreteClassifierBaseRFC);
  CEREAL_REGISTER_POLYMORPHIC_RELATION(ClassifierBaseCD, ContinuousClassifierBaseD);

  template<typename DataType, typename ClassifierType, typename... Args>
  class ContinuousClassifierBase;

  template<typename DataType, typename ClassifierType, typename... Args>
  class DiscreteClassifierBase;

  template<typename DataType>
  using LeavesMap = std::unordered_map<std::size_t, DataType>;

  namespace cereal {
  
  template<typename DataType>
  using LeavesMap = std::unordered_map<std::size_t, DataType>;

  template<typename DataType, typename ClassifierType, typename... Args> 
  struct LoadAndConstruct<ContinuousClassifierBase<DataType, ClassifierType, Args...>> {
  template<class Archive>
  static void load_anod_construct(Archive &ar, cereal::construct<ContinuousClassifierBase<DataType, ClassifierType, Args...>> &construct) {
  std::unique_ptr<ClassifierType> classifier;
  ar(CEREAL_NVP(classifier));
  construct(std::move(classifier));
  }
  };


  template<typename DataType, typename ClassifierType, typename... Args>
  struct LoadAndConstruct<DiscreteClassifierBase<DataType, ClassifierType, Args...>> {
  template<class Archive>
  static void load_and_construct(Archive &ar, cereal::construct<DiscreteClassifierBase<DataType, ClassifierType, Args...>> &construct) {
  LeavesMap<DataType> leavesMap;
  std::unique_ptr<ClassifierType> classifier;
  ar(CEREAL_NVP(leavesMap));
  ar(CEREAL_NVP(classifier));
  construct(leavesMap, std::move(classifier));
  }
  };

  } // namespace cereal
*/  


