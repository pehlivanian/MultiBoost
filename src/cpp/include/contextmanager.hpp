#ifndef __CONTEXT_MANAGER_HPP__
#define __CONTEXT_MANAGER_HPP__

#include <tuple>

#include "utils.hpp"

using namespace IB_utils;
using namespace ModelContext;

template<typename ClassifierType>
class CompositeClassifier;

class ContextManager {
public:
  using childModelInfo		= std::tuple<std::size_t, std::size_t, double>;
  using childPartitionInfo	= std::tuple<std::size_t, std::size_t, double, double>;

  ContextManager() = delete;

  ContextManager(const Context& context) = delete;
  ContextManager(Context&& context) = delete;
  
  template<typename ClassifierType>
  static void contextInit(CompositeClassifier<ClassifierType>&, const Context&);

  template<typename ClassifierType>
  static void childContext(Context&, const CompositeClassifier<ClassifierType>&);

private:
  template<typename ClassifierType>
  static auto computeChildPartitionInfo(const CompositeClassifier<ClassifierType>&) -> childPartitionInfo;
  
  template<typename ClassifierType>
  static auto computeChildModelInfo(const CompositeClassifier<ClassifierType>&) -> childModelInfo;

};

#include "contextmanager_impl.hpp"

#endif
