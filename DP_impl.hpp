#ifndef __DP_IMPL_HPP__
#define __DP_IMPL_HPP__

template<typename T>
class TD;

template<typename DataType>
void
DPSolver<DataType>::sort_by_priority(std::vector<DataType>& a, std::vector<DataType>& b) {
  std::vector<int> ind(a.size());
  std::iota(ind.begin(), ind.end(), 0);

  std::sort(ind.begin(), ind.end(),
	    [&a, &b](int i, int j) {
	      return (a[i]/b[i]) < (a[j]/b[j]);
	    });
  
  priority_sortind_ = ind;
  
  std::vector<DataType> a_s(n_), b_s(n_);
  for (int i=0; i<n_; ++i) {
    a_s[i] = a[ind[i]];
    b_s[i] = b[ind[i]];
  }

  std::copy(a_s.cbegin(), a_s.cend(), a.begin());
  std::copy(b_s.cbegin(), b_s.cend(), b.begin());
}

template<typename DataType>
void 
DPSolver<DataType>::createContext() {
  // create reference to score function
  if (parametric_dist_ == objective_fn::Gaussian) {
    context_ = std::make_unique<GaussianContext<DataType>>(a_, 
						 b_, 
						 n_, 
						 risk_partitioning_objective_,
						 use_rational_optimization_);
  }
  else if (parametric_dist_ == objective_fn::Poisson) {
    context_ = std::make_unique<PoissonContext<DataType>>(a_, 
						b_, 
						n_,
						risk_partitioning_objective_,
						use_rational_optimization_);
  }
  else if (parametric_dist_ == objective_fn::RationalScore) {
    context_ = std::make_unique<RationalScoreContext<DataType>>(a_, 
						      b_, 
						      n_,
						      risk_partitioning_objective_,
						      use_rational_optimization_);
  }
  else {
    throw distributionException();
  }

  context_->init();
}

template<typename DataType>
void 
DPSolver<DataType>::create() {
  // reset optimal_score_
  optimal_score_ = 0.;

  // sort vectors by priority function G(x,y) = x/y
  sort_by_priority(a_, b_);

  // create context
  createContext();
    
  // Initialize matrix
  maxScore_ = std::vector<std::vector<DataType> >(n_, std::vector<DataType>(T_+1, std::numeric_limits<DataType>::lowest()));
  nextStart_ = std::vector<std::vector<int> >(n_, std::vector<int>(T_+1, -1));
  subsets_ = std::vector<std::vector<int> >(T_, std::vector<int>());
  score_by_subset_ = std::vector<DataType>(T_, 0.);

  // Fill in first,second columns corresponding to T = 0,1
  for(int j=0; j<2; ++j) {
    for (int i=0; i<n_; ++i) {
      maxScore_[i][j] = (j==0)?0.:compute_score(i,n_);
      nextStart_[i][j] = (j==0)?-1:n_;
    }
  }

  // Fill in column-by-column from the left
  DataType score;
  DataType maxScore;
  int maxNextStart = -1;
  for(int j=2; j<=T_; ++j) {
    for (int i=0; i<n_; ++i) {
      maxScore = std::numeric_limits<DataType>::lowest();
      for (int k=i+1; k<=(n_-(j-1)); ++k) {
	score = compute_score(i,k) + maxScore_[k][j-1];
	if (score > maxScore) {
	  maxScore = score;
	  maxNextStart = k;
	}
      }
      maxScore_[i][j] = maxScore;
      nextStart_[i][j] = maxNextStart;
      // Only need the initial entry in last column
      if (j == T_)
	break;
    }
  }
}

template<typename DataType>
typename DPSolver<DataType>::all_scores
DPSolver<DataType>::optimize_for_fixed_S(int S) {
  // Pick out associated maxScores element
  int currentInd = 0, nextInd = 0;
  DataType optimal_score = 0.;
  auto subsets = std::vector<std::vector<int> >(S, std::vector<int>());
  auto score_by_subset = std::vector<DataType>(S, 0.);

  for (int t=S; t>0; --t) {
    nextInd = nextStart_[currentInd][t];
    for (int i=currentInd; i<nextInd; ++i) {
      subsets[S-t].push_back(priority_sortind_[i]);
    }
    score_by_subset[S-t] = compute_score(currentInd, nextInd);
    optimal_score += score_by_subset[S-t];
    currentInd = nextInd;
  }

  if (!risk_partitioning_objective_) {
    reorder_subsets(subsets, score_by_subset);
  }

  // subtract regularization term
  optimal_score -= gamma_ * std::pow(S, reg_power_);

  // Retain score_by_subsets if S is maximal
  if (S == T_)
    score_by_subset_ = score_by_subset;

  return all_scores{subsets, optimal_score};
}

#if EIGEN
template<typename DataType>
void
DPSolver<DataType>::find_optimal_t() {

  // Should not reach this
  throw distributionException();

  std::vector<DataType> X; X.resize(subsets_and_scores_.size()); std::iota(X.begin(), X.end(), 1.);
  std::vector<DataType> scores, score_diffs;
  scores.resize(subsets_and_scores_.size()); score_diffs.resize(subsets_and_scores_.size());
  std::transform(subsets_and_scores_.begin(), 
		 subsets_and_scores_.end(), scores.begin(), [](const auto &a){return a.second;});
  std::adjacent_difference(scores.begin(), scores.end(), score_diffs.begin());
  for (auto& el : X)
    el = log(el);
  for (auto& el : score_diffs)
    el = log(el);
  
  Eigen::MatrixXf X_mat = Eigen::MatrixXf::Zero(score_diffs.size()-2, 2);
  Eigen::VectorXf y_vec = Eigen::VectorXf::Zero(score_diffs.size()-2);
  
  for (size_t i=0; i<score_diffs.size()-2; ++i) {
    X_mat(i,0) = X[i+2];
    X_mat(i,1) = 1.;
    y_vec(i) = score_diffs[i+2];
  }
  
  // OLS solution is a matrix of shape (2,1)
  Eigen::MatrixXf beta_hat = X_mat.colPivHouseholderQr().solve(y_vec);
  
  // Compute residuals
  Eigen::VectorXf resids = Eigen::VectorXf::Zero(score_diffs.size()-2);
  int bestInd = 0;
  DataType minResid = std::numeric_limits<DataType>::max();
  for (int i=0; i<static_cast<int>(score_diffs.size())-2; ++i) {
    resids(i) = y_vec(i) - beta_hat(0) * X_mat(i,0) - beta_hat(1) * X_mat(i,1);
    if (resids(i) < minResid) {
      bestInd = i;
      minResid = resids(i);
    }
  }
  optimal_num_clusters_OLS_ = bestInd + 1;
  subsets_ = subsets_and_scores_[optimal_num_clusters_OLS_].first;
  optimal_score_ = subsets_and_scores_[optimal_num_clusters_OLS_].second;
}
#endif

template<typename DataType>
void
DPSolver<DataType>::optimize() {
  if (sweep_down_ || find_optimal_t_) {
    int S;
    subsets_and_scores_ = all_part_scores{static_cast<size_t>(T_+1)};
    auto beg = subsets_and_scores_.rend();
    for (auto it=subsets_and_scores_.rbegin(); it!=subsets_and_scores_.rend(); ++it) {
      S = static_cast<int>(std::distance(it, beg)) - 1;
      if (S >= 1) {
	auto optimal = optimize_for_fixed_S(S);
	it->first = optimal.first;
	it->second = optimal.second;
      }
    }
    if (find_optimal_t_) {
#if EIGEN
#if (!IS_CXX_11) && !(__cplusplus == 201103L) 
      find_optimal_t();
#else
      optimal_num_clusters_OLS_ = -1;
#endif
#endif
    }
  }
  else {
    auto optimal = optimize_for_fixed_S(T_);
    subsets_ = optimal.first;
    optimal_score_ = optimal.second;
  }
}

template<typename DataType>
void
DPSolver<DataType>::reorder_subsets(std::vector<std::vector<int> >& subsets, 
			  std::vector<DataType>& score_by_subsets) {
  std::vector<int> ind(subsets.size(), 0);
  std::iota(ind.begin(), ind.end(), 0.);

  std::stable_sort(ind.begin(), ind.end(),
		   [score_by_subsets](int i, int j) {
		     return (score_by_subsets[i] < score_by_subsets[j]);
		   });

  // Inefficient reordering
  std::vector<std::vector<int> > subsets_s;
  std::vector<DataType> score_by_subsets_s;
  subsets_s = std::vector<std::vector<int> >(subsets.size(), std::vector<int>());
  score_by_subsets_s = std::vector<DataType>(subsets.size(), 0.);

  for (size_t i=0; i<subsets.size(); ++i) {
    subsets_s[i] = subsets[ind[i]];
    score_by_subsets_s[i] = score_by_subsets[ind[i]];
  }

  std::copy(subsets_s.cbegin(), subsets_s.cend(), subsets.begin());
  std::copy(score_by_subsets_s.cbegin(), score_by_subsets_s.cend(), score_by_subsets.begin());
		   
  
}

template<typename DataType>
std::vector<std::vector<int> >
DPSolver<DataType>::get_optimal_subsets_extern() const {
  return subsets_;
}

template<typename DataType>
DataType
DPSolver<DataType>::get_optimal_score_extern() const {
  if (risk_partitioning_objective_) {
    return optimal_score_;
  }
  else {
    return std::accumulate(score_by_subset_.cbegin()+1, score_by_subset_.cend(), 0.) - gamma_ * std::pow(T_, reg_power_);
  }
}

template<typename DataType>
std::vector<DataType>
DPSolver<DataType>::get_score_by_subset_extern() const {
  return score_by_subset_;
}

template<typename DataType>
typename DPSolver<DataType>::all_part_scores 
DPSolver<DataType>::get_all_subsets_and_scores_extern() const {
  return subsets_and_scores_;
}

template<typename DataType>
int
DPSolver<DataType>::get_optimal_num_clusters_OLS_extern() const {
  return optimal_num_clusters_OLS_;
}

template<typename DataType>
void
DPSolver<DataType>::print_maxScore_() {

  for (int i=0; i<n_; ++i) {
    std::copy(maxScore_[i].cbegin(), maxScore_[i].cend(), std::ostream_iterator<DataType>(std::cout, " "));
    std::cout << std::endl;
  }
}

template<typename DataType>
void
DPSolver<DataType>::print_nextStart_() {
  for (int i=0; i<n_; ++i) {
    std::copy(nextStart_[i].cbegin(), nextStart_[i].cend(), std::ostream_iterator<int>(std::cout, " "));
    std::cout << std::endl;
  }
}

template<typename DataType>
DataType
DPSolver<DataType>::compute_score(int i, int j) {
  return context_->get_score(i, j);
}

template<typename DataType>
DataType
DPSolver<DataType>::compute_ambient_score(DataType a, DataType b) {
  return context_->compute_ambient_score(a, b);
}

#endif
