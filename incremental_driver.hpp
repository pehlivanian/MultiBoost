#ifndef __INCREMENTAL_DRIVER_HPP__
#define __INCREMENTAL_DRIVER_HPP__

#include <iostream>
#include <numeric>
#include <string>
#include <cstring>
#include <memory>

#include <boost/program_options.hpp>

#include <mlpack/core.hpp>
#include <mlpack/methods/decision_tree/decision_tree.hpp>
#include <mlpack/methods/decision_tree/information_gain.hpp>
#include <mlpack/methods/decision_tree/gini_gain.hpp>
#include <mlpack/methods/decision_tree/random_dimension_select.hpp>
#include <mlpack/methods/decision_tree/multiple_random_dimension_select.hpp>


#include "loss.hpp"
#include "gradientboostclassifier.hpp"
#include "replay.hpp"
#include "utils.hpp"


#endif
