#ifndef __DATAELEMENT_HPP__
#define __DATAELEMENT_HPP__

#include "visitor.hpp"

class DataElement {
public:
  inline void accept(Visitor& v) { v.visit(this); }
  virtual ~DataElement() = default;
protected:
  DataElement() = default;
};
#endif
