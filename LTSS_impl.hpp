#ifndef __LTSS_IMPL_HPP__
#define __LTSS_IMPL_HPP__

template<typename DataType>
DataType
LTSSSolver<DataType>::compute_score(int i, int j) {
  return context_->get_score(i, j);
}

template<typename DataType>
void
LTSSSolver<DataType>::sort_by_priority(std::vector<DataType>& a, std::vector<DataType>& b) {
  std::vector<int> ind(a.size());
  std::iota(ind.begin(), ind.end(), 0);

  std::stable_sort(ind.begin(), ind.end(),
		   [&a, &b](int i, int j) {
		     return (a[i]/b[i]) < (a[j]/b[j]);
		   });

  priority_sortind_ = ind;

  // Inefficient reordering
  std::vector<DataType> a_s, b_s;
  for (auto i : ind) {
    a_s.push_back(a[i]);
    b_s.push_back(b[i]);
  }

  std::copy(a_s.begin(), a_s.end(), a.begin());
  std::copy(b_s.begin(), b_s.end(), b.begin());
  
}

template<typename DataType>
void
LTSSSolver<DataType>::createContext() {
  // create reference to score function
  // always use multiple clustering objective
  if (parametric_dist_ == objective_fn::Gaussian) {
    context_ = std::make_unique<GaussianContext<DataType>>(a_, 
						 b_, 
						 n_, 
						 false,
						 false);
  }
  else if (parametric_dist_ == objective_fn::Poisson) {
    context_ = std::make_unique<PoissonContext<DataType>>(a_, 
						b_, 
						n_,
						false,
						false);
  }
  else if (parametric_dist_ == objective_fn::RationalScore) {
    context_ = std::make_unique<RationalScoreContext<DataType>>(a_,
						      b_,
						      n_,
						      false,
						      false);
  }
  else {
    throw distributionException();
  }

  context_->init();
}

template<typename DataType>
void 
LTSSSolver<DataType>::create() {
  // sort by priority
  sort_by_priority(a_, b_);

  subset_ = std::vector<int>();

  // create context
  createContext();
}

template<typename DataType>
void
LTSSSolver<DataType>::optimize() {
  optimal_score_ = 0.;

  DataType maxScore = -std::numeric_limits<DataType>::max();
  std::pair<int, int> p;
  // Test ascending partitions
  for (int i=1; i<=n_; ++i) {
    auto score = compute_score(0, i);
    if (score > maxScore) {
      maxScore = score;
      p = std::make_pair(0, i);
    }
  }
  // Test descending partitions
  for (int i=n_-1; i>=0; --i) {
    auto score = compute_score(i, n_);
    if (score > maxScore) {
      maxScore = score;
      p = std::make_pair(i, n_);
    }
  }
  
  for (int i=p.first; i<p.second; ++i) {
    subset_.push_back(priority_sortind_[i]);
  }
  optimal_score_ = maxScore;
}

template<typename DataType>
std::vector<int>
LTSSSolver<DataType>::get_optimal_subset_extern() const {
  return subset_;
}

template<typename DataType>
DataType
LTSSSolver<DataType>::get_optimal_score_extern() const {
  return optimal_score_;
}

#endif
