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

  ContextManager() = default;
  ContextManager(const Context& context) : context_{context} {}
  ContextManager(Context&& context) : context_{std::move(context)} {}
  
  template<typename ClassifierType>
  void contextInit_(CompositeClassifier<ClassifierType>&);

  template<typename ClassifierType>
  void childContext(Context&, const CompositeClassifier<ClassifierType>&);

private:
  template<typename ClassifierType>
  auto computeChildPartitionInfo(const CompositeClassifier<ClassifierType>&) -> childPartitionInfo;
  
  template<typename ClassifierType>
  auto computeChildModelInfo(const CompositeClassifier<ClassifierType>&) -> childModelInfo;

  Context context_;
};

#include "contextmanager_impl.hpp"

#endif
