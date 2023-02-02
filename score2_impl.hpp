#ifndef __SCORE2_IMPL_HPP__
#define __SCORE2_IMPL_HPP__

namespace {
  const int NUMTHREADS = 8;
}

namespace Objectives {
  template<typename DataType>
  DataType
  ParametricContext<DataType>::get_score(int i, int j) const {
    return partialSums_[i][j]; 
  }

  template<typename DataType>
  DataType
  ParametricContext<DataType>::get_ambient_score(DataType a, DataType b) const {
    return compute_ambient_score(a, b);
  }

  template<typename DataType>
  std::vector<std::vector<DataType>> 
  ParametricContext<DataType>::get_scores() const {
    return partialSums_;
  }

  template<typename DataType>
  std::string 
  ParametricContext<DataType>::getName() const { 
    return name_;
  }

  template<typename DataType>
  bool 
  ParametricContext<DataType>::getRiskPartitioningObjective() const {
    return risk_partitioning_objective_;
  }

  template<typename DataType>
  bool 
  ParametricContext<DataType>::getUseRationalOptimization() const {
    return use_rational_optimization_;
  }

  template<typename DataType>
  std::vector<std::vector<DataType>>
  ParametricContext<DataType>::get_partial_sums_a() const {
    return a_sums_;
  }

  template<typename DataType>
  std::vector<std::vector<DataType>>
  ParametricContext<DataType>::get_partial_sums_b() const {
    return b_sums_;
  }

  template<typename DataType>
  void 
  ParametricContext<DataType>::compute_partial_sums() {
    a_sums_ = std::vector<std::vector<DataType> >(n_, std::vector<DataType>(n_+1, std::numeric_limits<DataType>::lowest()));
    b_sums_ = std::vector<std::vector<DataType> >(n_, std::vector<DataType>(n_+1, std::numeric_limits<DataType>::lowest()));
    partialSums_ = std::vector<std::vector<DataType>>(n_, std::vector<DataType>(n_+1, 0.));
  
    for (int i=0; i<n_; ++i) {
      a_sums_[i][i] = 0.;
      b_sums_[i][i] = 0.;
    }
  
    for (int i=0; i<n_; ++i) {
      for (int j=i+1; j<=n_; ++j) {
	a_sums_[i][j] = a_sums_[i][j-1] + a_[j-1];
	b_sums_[i][j] = b_sums_[i][j-1] + b_[j-1];
      }
    }  
  }

  template<typename DataType>
  void
  ParametricContext<DataType>::compute_partial_sums_AVX256() {

    a_sums_ = std::vector<std::vector<DataType>>(n_+1, std::vector<DataType>(n_, std::numeric_limits<DataType>::lowest()));
    b_sums_ = std::vector<std::vector<DataType>>(n_+1, std::vector<DataType>(n_, std::numeric_limits<DataType>::lowest()));

    for (int j=0; j<n_; ++j) {
      a_sums_[j][j] = 0.;
      b_sums_[j][j] = 0.;
    }

    DataType r_[8];
    for (int j=1; j<n_+1; ++j) {
      int unroll = (j/4)*4, i=0;
      for (; i<unroll; i+=4) {
	__m256 v1 = _mm256_set_ps(a_sums_[j-1][i], a_sums_[j-1][i+1], a_sums_[j-1][i+2], a_sums_[j-1][i+3],
				  b_sums_[j-1][i], b_sums_[j-1][i+1], b_sums_[j-1][i+2], b_sums_[j-1][i+3]); 
	__m256 v2 = _mm256_set_ps(a_[j-1], a_[j-1], a_[j-1], a_[j-1],
				  b_[j-1], b_[j-1], b_[j-1], b_[j-1]);
	__m256 r = _mm256_add_ps(v1, v2);
	memcpy(r_, &r, sizeof(r_));
	a_sums_[j][i]   = r_[7];
	a_sums_[j][i+1] = r_[6];
	a_sums_[j][i+2] = r_[5];
	a_sums_[j][i+3] = r_[4];
	b_sums_[j][i]   = r_[3];
	b_sums_[j][i+1] = r_[2];
	b_sums_[j][i+2] = r_[1];
	b_sums_[j][i+3] = r_[0];
      }
    
      for(;i<j;++i) {
	a_sums_[j][i] = a_sums_[j-1][i] + a_[j-1];
	b_sums_[j][i] = b_sums_[j-1][i] + b_[j-1];      
      }
    }
  }

  template<typename DataType>
  void
  ParametricContext<DataType>::compute_partial_sums_parallel() {

    const int numThreads = std::min(n_-2, NUMTHREADS);

    a_sums_ = std::vector<std::vector<DataType> >(n_, std::vector<DataType>(n_+1, std::numeric_limits<DataType>::lowest()));
    b_sums_ = std::vector<std::vector<DataType> >(n_, std::vector<DataType>(n_+1, std::numeric_limits<DataType>::lowest()));

    for (int i=0; i<n_; ++i) {
      a_sums_[i][i] = 0.;
      b_sums_[i][i] = 0.;
    }

    auto task_ab_block = [this](int ind1, int ind2) {
      for (int i=ind1; i<ind2; ++i) {
	for (int j=i+1; j<=this->n_; ++j) {
	  this->a_sums_[i][j] = this->a_sums_[i][j-1] + this->a_[j-1];
	  this->b_sums_[i][j] = this->b_sums_[i][j-1] + this->b_[j-1];
	}
      }
    };

    int blockSize = static_cast<DataType>(n_)/static_cast<DataType>(numThreads);
    int startOfBlock = 0, endOfBlock = startOfBlock + blockSize;
  
    std::vector<std::thread> threads;

    while (endOfBlock < n_) {
      threads.emplace_back(task_ab_block, startOfBlock, endOfBlock);
      startOfBlock = endOfBlock; endOfBlock+= blockSize;
    }
    threads.emplace_back(task_ab_block, startOfBlock, n_);
  
    for (auto it=threads.begin(); it!=threads.end(); ++it)
      it->join();

  }

  template<typename DataType>
  void 
  ParametricContext<DataType>::compute_scores() {
    a_sums_ = std::vector<std::vector<DataType> >(n_, std::vector<DataType>(n_+1, std::numeric_limits<DataType>::lowest()));
    b_sums_ = std::vector<std::vector<DataType> >(n_, std::vector<DataType>(n_+1, std::numeric_limits<DataType>::lowest()));
    partialSums_ = std::vector<std::vector<DataType>>(n_, std::vector<DataType>(n_+1, 0.));
  
    for (int i=0; i<n_; ++i) {
      a_sums_[i][i] = 0.;
      b_sums_[i][i] = 0.;
    }
  
    for (int i=0; i<n_; ++i) {
      for (int j=i+1; j<=n_; ++j) {
	a_sums_[i][j] = a_sums_[i][j-1] + a_[j-1];
	b_sums_[i][j] = b_sums_[i][j-1] + b_[j-1];
	partialSums_[i][j] = this->compute_score(i, j);
      }
    }  
  }

  template<typename DataType>
  void
  ParametricContext<DataType>::compute_scores_AVX256() {

    a_sums_ = std::vector<std::vector<DataType>>(n_+1, std::vector<DataType>(n_, std::numeric_limits<DataType>::lowest()));
    b_sums_ = std::vector<std::vector<DataType>>(n_+1, std::vector<DataType>(n_, std::numeric_limits<DataType>::lowest()));
    partialSums_ = std::vector<std::vector<DataType>>(n_, std::vector<DataType>(n_+1, 0.));

    for (int j=0; j<n_; ++j) {
      a_sums_[j][j] = 0.;
      b_sums_[j][j] = 0.;
    }

    DataType r_[8];
    for (int j=1; j<n_+1; ++j) {
      int unroll = (j/4)*4, i=0;
      for (; i<unroll; i+=4) {
	__m256 v1 = _mm256_set_ps(a_sums_[j-1][i], a_sums_[j-1][i+1], a_sums_[j-1][i+2], a_sums_[j-1][i+3],
				  b_sums_[j-1][i], b_sums_[j-1][i+1], b_sums_[j-1][i+2], b_sums_[j-1][i+3]); 
	__m256 v2 = _mm256_set_ps(a_[j-1], a_[j-1], a_[j-1], a_[j-1],
				  b_[j-1], b_[j-1], b_[j-1], b_[j-1]);
	__m256 r = _mm256_add_ps(v1, v2);
	memcpy(r_, &r, sizeof(r_));
	a_sums_[j][i]   = r_[7];
	a_sums_[j][i+1] = r_[6];
	a_sums_[j][i+2] = r_[5];
	a_sums_[j][i+3] = r_[4];
	b_sums_[j][i]   = r_[3];
	b_sums_[j][i+1] = r_[2];
	b_sums_[j][i+2] = r_[1];
	b_sums_[j][i+3] = r_[0];

	partialSums_[i][j]   = this->compute_score(j, i);
	partialSums_[i+1][j] = this->compute_score(j, i+1);
	partialSums_[i+2][j] = this->compute_score(j, i+2);
	partialSums_[i+3][j] = this->compute_score(j, i+3);
      }
    
      for(;i<j;++i) {
	a_sums_[j][i] = a_sums_[j-1][i] + a_[j-1];
	b_sums_[j][i] = b_sums_[j-1][i] + b_[j-1];      
	partialSums_[i][j] = this->compute_score(j, i);
      }
    }
  }


  template<typename DataType>
  void
  ParametricContext<DataType>::compute_scores_parallel() {

    const int numThreads = std::min(n_-2, NUMTHREADS);

    a_sums_ = std::vector<std::vector<DataType> >(n_, std::vector<DataType>(n_+1, std::numeric_limits<DataType>::lowest()));
    b_sums_ = std::vector<std::vector<DataType> >(n_, std::vector<DataType>(n_+1, std::numeric_limits<DataType>::lowest()));
    partialSums_ = std::vector<std::vector<DataType>>(n_, std::vector<DataType>(n_+1, 0.));
  
    for (int i=0; i<n_; ++i) {
      a_sums_[i][i] = 0.;
      b_sums_[i][i] = 0.;
    }

    auto task_ab_block = [this](int ind1, int ind2) {
      for (int i=ind1; i<ind2; ++i) {
	for (int j=i+1; j<=this->n_; ++j) {
	  this->a_sums_[i][j] = this->a_sums_[i][j-1] + this->a_[j-1];
	  this->b_sums_[i][j] = this->b_sums_[i][j-1] + this->b_[j-1];
	  this->partialSums_[i][j] = this->compute_score(i, j);
	}
      }
    };

    int blockSize = static_cast<DataType>(n_)/static_cast<DataType>(numThreads);
    int startOfBlock = 0, endOfBlock = startOfBlock + blockSize;
  
    std::vector<std::thread> threads;

    while (endOfBlock < n_) {
      threads.emplace_back(task_ab_block, startOfBlock, endOfBlock);
      startOfBlock = endOfBlock; endOfBlock+= blockSize;
    }
    threads.emplace_back(task_ab_block, startOfBlock, n_);
  
    for (auto it=threads.begin(); it!=threads.end(); ++it)
      it->join();

  }

  template<typename DataType>
  DataType
  ParametricContext<DataType>::compute_score(int i, int j) {
    if (risk_partitioning_objective_) {
      return compute_score_riskpart_optimized(i, j);
    }
    else {
      return compute_score_multclust_optimized(i, j);
    }
  }

  template<typename DataType>
  DataType 
  ParametricContext<DataType>::compute_ambient_score(DataType a, DataType b) {
    if (risk_partitioning_objective_) {
      return compute_ambient_score_riskpart(a, b);
    }
    else {
      return compute_ambient_score_multclust(a, b);
    }
  }

  template<typename DataType>
  void
  ParametricContext<DataType>::init() {
    // XXX
    // compute_scores_parallel();
    compute_scores();
    // compute_scores_AVX256();
  }


  ////////////////////////////////////////////


  template<typename DataType>
  DataType 
  PoissonContext<DataType>::compute_score_multclust(int i, int j) {    
    auto& a = ParametricContext<DataType>::a_;
    auto& b = ParametricContext<DataType>::b_;
    DataType C = std::accumulate(a.cbegin()+i, a.cbegin()+j, 0.);
    DataType B = std::accumulate(b.cbegin()+i, b.cbegin()+j, 0.);
    return (C>B)? C*std::log(C/B) + B - C : 0.;
  }

  template<typename DataType>
  DataType 
  PoissonContext<DataType>::compute_score_riskpart(int i, int j) {
    auto& a = ParametricContext<DataType>::a_;
    auto& b = ParametricContext<DataType>::b_;
    DataType C = std::accumulate(a.cbegin()+i, a.cbegin()+j, 0.);
    DataType B = std::accumulate(b.cbegin()+i, b.cbegin()+j, 0.);
    return C*std::log(C/B);
  }
 
  template<typename DataType>   
  DataType 
  PoissonContext<DataType>::compute_ambient_score_multclust(DataType a, DataType b) {
    return (a>b)? a*std::log(a/b) + b - a : 0.;
  }

  template<typename DataType>
  DataType 
  PoissonContext<DataType>::compute_ambient_score_riskpart(DataType a, DataType b) {
    return a*std::log(a/b);
  }  

  template<typename DataType>
  DataType 
  PoissonContext<DataType>::compute_score_riskpart_optimized(int i, int j) {
    auto& a_sums = ParametricContext<DataType>::a_sums_;
    auto& b_sums = ParametricContext<DataType>::b_sums_;
    DataType score = a_sums[i][j]*std::log(a_sums[i][j]/b_sums[i][j]);
    return score;
  }

  template<typename DataType>
  DataType
  PoissonContext<DataType>::compute_score_multclust_optimized(int i, int j) {
    auto& a_sums = ParametricContext<DataType>::a_sums_;
    auto& b_sums = ParametricContext<DataType>::b_sums_;
    DataType score = (a_sums[i][j] > b_sums[i][j]) ? a_sums[i][j]*std::log(a_sums[i][j]/b_sums[i][j]) + b_sums[i][j] - a_sums[i][j]: 0.;
    return score;
  }

  template<typename DataType>
  DataType 
  GaussianContext<DataType>::compute_score_multclust(int i, int j) {
    auto& a = ParametricContext<DataType>::a_;
    auto& b = ParametricContext<DataType>::b_;
    DataType C = std::accumulate(a.cbegin()+i, a.cbegin()+j, 0.);
    DataType B = std::accumulate(b.cbegin()+i, b.cbegin()+j, 0.);
    return (C>B)? .5*(std::pow(C,2)/B + B) - C : 0.;
  }

  template<typename DataType>
  DataType 
  GaussianContext<DataType>::compute_score_riskpart(int i, int j) {
    auto& a = ParametricContext<DataType>::a_;
    auto& b = ParametricContext<DataType>::b_;
    DataType C = std::accumulate(a.cbegin()+i, a.cbegin()+j, 0.);
    DataType B = std::accumulate(b.cbegin()+i, b.cbegin()+j, 0.);
    return C*C/2./B;
  }
  
  template<typename DataType>
  DataType 
  GaussianContext<DataType>::compute_ambient_score_multclust(DataType a, DataType b) {
    return (a>b)? .5*(std::pow(a,2)/b + b) - a : 0.;
  }
  
  template<typename DataType>
  DataType 
  GaussianContext<DataType>::compute_ambient_score_riskpart(DataType a, DataType b) {
    return a*a/2./b;
  }
  
  template<typename DataType>
  DataType 
  GaussianContext<DataType>::compute_score_multclust_optimized(int i, int j) {
    auto& a_sums = ParametricContext<DataType>::a_sums_;
    auto& b_sums = ParametricContext<DataType>::b_sums_;
    DataType score = (a_sums[i][j] > b_sums[i][j]) ? .5*(std::pow(a_sums[i][j], 2)/b_sums[i][j] + b_sums[i][j]) - a_sums[i][j] : 0.;
    return score;
  }
   
  template<typename DataType>
  DataType 
  GaussianContext<DataType>::compute_score_riskpart_optimized(int i, int j) {
    auto& a_sums = ParametricContext<DataType>::a_sums_;
    auto& b_sums = ParametricContext<DataType>::b_sums_;
    DataType score = a_sums[i][j]*a_sums[i][j]/2./b_sums[i][j];
    return score;
  }

  template<typename DataType>
  DataType 
  RationalScoreContext<DataType>::compute_score_multclust(int i, int j) {
    auto& a = ParametricContext<DataType>::a_;
    auto& b = ParametricContext<DataType>::b_;
    DataType score = std::pow(std::accumulate(a.cbegin()+i, a.cbegin()+j, 0.), 2) /
      std::accumulate(b.cbegin()+i, b.cbegin()+j, 0.);
    return score;
  }

  template<typename DataType>
  DataType 
  RationalScoreContext<DataType>::compute_score_riskpart(int i, int j) {
    return compute_score_multclust(i, j);
  }
  
  template<typename DataType>
  DataType 
  RationalScoreContext<DataType>::compute_score_riskpart_optimized(int i, int j) {
    return compute_score_multclust_optimized(i, j);
  }

  template<typename DataType>
  DataType 
  RationalScoreContext<DataType>::compute_score_multclust_optimized(int i, int j) {
    auto& a_sums = ParametricContext<DataType>::a_sums_;
    auto& b_sums = ParametricContext<DataType>::b_sums_;
    DataType score = a_sums[i][j] * a_sums[i][j] / b_sums[i][j];
    return score;
  }

  template<typename DataType>
  DataType 
  RationalScoreContext<DataType>::compute_ambient_score_multclust(DataType a, DataType b) {
    return a*a/b;
  }
  
  template<typename DataType>
  DataType 
  RationalScoreContext<DataType>::compute_ambient_score_riskpart(DataType a, DataType b) {
    return a*a/b;
  }

}
#endif
