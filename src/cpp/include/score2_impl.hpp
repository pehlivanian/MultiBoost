#ifndef __SCORE2_IMPL_HPP__
#define __SCORE2_IMPL_HPP__

#undef SERIAL_CALCULATION
#undef AVX256_CALCULATION
#if !defined(PARALLEL_CALCULATION)
  #define PARALLEL_CALCULATION
#endif

namespace {
  // Only used #ifdef PARALLEL_CALCULATION
  // Hard-coded rather than based on hardware_concurrency
  const std::size_t NUMTHREADS = 8;
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
  ParametricContext<DataType>::get_partial_sums() const {
    return partialSums_;
  }

  template<typename DataType>
  std::vector<std::vector<DataType>>
  ParametricContext<DataType>::get_partial_sums_b() const {
    return b_sums_;
  }

  template<typename DataType>
  void 
  ParametricContext<DataType>::compute_partial_sums() {

    using vvec = std::vector<std::vector<DataType>>;
    a_sums_ = vvec{n_+1, std::vector<DataType>(n_+1,0.)};
    b_sums_ = vvec{n_+1, std::vector<DataType>(n_+1,0.)};

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

    using vvec = std::vector<std::vector<DataType>>;
    a_sums_ = vvec{n_+1, std::vector<DataType>(n_+1,0.)};
    b_sums_ = vvec{n_+1, std::vector<DataType>(n_+1,0.)};
    
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

    using vvec = std::vector<std::vector<DataType>>;
    a_sums_ = vvec{n_+1, std::vector<DataType>(n_+1,0.)};
    b_sums_ = vvec{n_+1, std::vector<DataType>(n_+1,0.)};

    const int numThreads = std::min(n_-2, NUMTHREADS);

    auto task_ab_block = [this](int ind1, int ind2) {
      for (int i=ind1; i<ind2; ++i) {
	for (int j=i+1; j<=n_; ++j) {
	  a_sums_[i][j] = a_sums_[i][j-1] + a_[j-1];
	  b_sums_[i][j] = b_sums_[i][j-1] + b_[j-1];
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

    using vvec = std::vector<std::vector<DataType>>;
    a_sums_ = vvec{n_+1, std::vector<DataType>(n_+1,0.)};
    b_sums_ = vvec{n_+1, std::vector<DataType>(n_+1,0.)};
  
    for (int i=0; i<n_; ++i) {
      for (int j=i+1; j<=n_; ++j) {
	a_sums_[i][j] = a_sums_[i][j-1] + a_[j-1];
	b_sums_[i][j] = b_sums_[i][j-1] + b_[j-1];
	partialSums_[i][j] = compute_score(i, j);
      }
    }  
  }

  template<typename DataType>
  void
  ParametricContext<DataType>::compute_scores_AVX256() {

    using vvec = std::vector<std::vector<DataType>>;
    a_sums_ = vvec{n_+1, std::vector<DataType>(n_+1,0.)};
    b_sums_ = vvec{n_+1, std::vector<DataType>(n_+1,0.)};

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

	partialSums_[i][j]   = compute_score(j, i);
	partialSums_[i+1][j] = compute_score(j, i+1);
	partialSums_[i+2][j] = compute_score(j, i+2);
	partialSums_[i+3][j] = compute_score(j, i+3);
      }
    
      for(;i<j;++i) {
	a_sums_[j][i] = a_sums_[j-1][i] + a_[j-1];
	b_sums_[j][i] = b_sums_[j-1][i] + b_[j-1];      
	partialSums_[i][j] = compute_score(j, i);
      }
    }
  }


  template<typename DataType>
  void
  ParametricContext<DataType>::compute_scores_parallel() {

    using vvec = std::vector<std::vector<DataType>>;
    a_sums_ = vvec{n_+1, std::vector<DataType>(n_+1,0.)};
    b_sums_ = vvec{n_+1, std::vector<DataType>(n_+1,0.)};

    const int numThreads = std::min(n_-2, NUMTHREADS);

    auto task_ab_block = [this](int ind1, int ind2) {
      for (int i=ind1; i<ind2; ++i) {
	for (int j=i+1; j<=n_; ++j) {
	  a_sums_[i][j] = a_sums_[i][j-1] + a_[j-1];
	  b_sums_[i][j] = b_sums_[i][j-1] + b_[j-1];
	  partialSums_[i][j] = compute_score(i, j);
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

#if defined(SERIAL_CALCULATION)
    // naive version
    compute_scores();
#elif defined(AVX256_CALCULATION)
    // AVX256 SIMD directives
    compute_scores_AVX256();
#elif defined(PARALLEL_CALCULATION)
    // parallel version
    compute_scores_parallel();
#endif
  }


  ////////////////////////////////////////////


  template<typename DataType>
  DataType 
  PoissonContext<DataType>::compute_score_multclust(int i, int j) {    
    throw std::runtime_error("Deprecated; please use optimized versions");
    DataType C = std::accumulate(ParametricContext<DataType>::a_.cbegin()+i, 
				 ParametricContext<DataType>::a_.cbegin()+j, 0.);
    DataType B = std::accumulate(ParametricContext<DataType>::b_.cbegin()+i, 
				 ParametricContext<DataType>::b_.cbegin()+j, 0.);
    return (C>B)? C*std::log(C/B) + B - C : 0.;
  }

  template<typename DataType>
  DataType 
  PoissonContext<DataType>::compute_score_riskpart(int i, int j) {
    throw std::runtime_error("Deprecated; please use optimized versions");
    DataType C = std::accumulate(ParametricContext<DataType>::a_.cbegin()+i, 
				 ParametricContext<DataType>::a_.cbegin()+j, 0.);
    DataType B = std::accumulate(ParametricContext<DataType>::b_.cbegin()+i, 
				 ParametricContext<DataType>::b_.cbegin()+j, 0.);
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
    DataType score = ParametricContext<DataType>::a_sums_[i][j]*
      std::log(ParametricContext<DataType>::a_sums_[i][j]/ParametricContext<DataType>::b_sums_[i][j]);
    return score;
  }

  template<typename DataType>
  DataType
  PoissonContext<DataType>::compute_score_multclust_optimized(int i, int j) {
    DataType score = (ParametricContext<DataType>::a_sums_[i][j] > ParametricContext<DataType>::b_sums_[i][j]) ? 
      ParametricContext<DataType>::a_sums_[i][j]*
      std::log(ParametricContext<DataType>::a_sums_[i][j]/ParametricContext<DataType>::b_sums_[i][j]) + 
      ParametricContext<DataType>::b_sums_[i][j] - ParametricContext<DataType>::a_sums_[i][j]: 0.;
    return score;
  }

  template<typename DataType>
  DataType 
  GaussianContext<DataType>::compute_score_multclust(int i, int j) {
    throw std::runtime_error("Deprecated; please use optimized versions");
    DataType C = std::accumulate(ParametricContext<DataType>::a_.cbegin()+i, 
				 ParametricContext<DataType>::a_.cbegin()+j, 0.);
    DataType B = std::accumulate(ParametricContext<DataType>::b_.cbegin()+i, 
				 ParametricContext<DataType>::b_.cbegin()+j, 0.);
    return (C>B)? .5*(std::pow(C,2)/B + B) - C : 0.;
  }

  template<typename DataType>
  DataType 
  GaussianContext<DataType>::compute_score_riskpart(int i, int j) {
    throw std::runtime_error("Deprecated; please use optimized versions");
    DataType C = std::accumulate(ParametricContext<DataType>::a_.cbegin()+i, 
				 ParametricContext<DataType>::a_.cbegin()+j, 0.);
    DataType B = std::accumulate(ParametricContext<DataType>::b_.cbegin()+i, 
				 ParametricContext<DataType>::b_.cbegin()+j, 0.);
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
    DataType score = (ParametricContext<DataType>::a_sums_[i][j] > ParametricContext<DataType>::b_sums_[i][j]) ? 
      .5*(std::pow(ParametricContext<DataType>::a_sums_[i][j], 2)/ParametricContext<DataType>::b_sums_[i][j] + 
	  ParametricContext<DataType>::b_sums_[i][j]) - ParametricContext<DataType>::a_sums_[i][j] : 0.;
    return score;
  }
   
  template<typename DataType>
  DataType 
  GaussianContext<DataType>::compute_score_riskpart_optimized(int i, int j) {
    DataType score = ParametricContext<DataType>::a_sums_[i][j]*
      ParametricContext<DataType>::a_sums_[i][j]/2./ParametricContext<DataType>::b_sums_[i][j];
    return score;
  }

  template<typename DataType>
  DataType 
  RationalScoreContext<DataType>::compute_score_multclust(int i, int j) {
    throw std::runtime_error("Deprecated; please use optimized versions");
    DataType score = std::pow(std::accumulate(ParametricContext<DataType>::a_.cbegin()+i, 
					      ParametricContext<DataType>::a_.cbegin()+j, 0.), 2) /
      std::accumulate(ParametricContext<DataType>::b_.cbegin()+i, 
		      ParametricContext<DataType>::b_.cbegin()+j, 0.);
    return score;
  }

  template<typename DataType>
  DataType 
  RationalScoreContext<DataType>::compute_score_riskpart(int i, int j) {
    throw std::runtime_error("Deprecated; please use optimized versions");
    return compute_score_multclust(i, j);
  }
  
  template<typename DataType>
  DataType 
  RationalScoreContext<DataType>::compute_score_riskpart_optimized(int i, int j) {
    return ParametricContext<DataType>::partialSums_[i][j];
  }

  template<typename DataType>
  DataType 
  RationalScoreContext<DataType>::compute_score_multclust_optimized(int i, int j) {
    return ParametricContext<DataType>::partialSums_[i][j];
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

  template<typename DataType>
  void
  RationalScoreContext<DataType>::compute_scores() {

    std::size_t n_ = ParametricContext<DataType>::n_;

    for (int i=0; i<n_; ++i) {
      DataType a_sum=0., b_sum=0.;
      for (int j=i+1; j<=n_; ++j) {
	a_sum += ParametricContext<DataType>::a_[j-1];
	b_sum += ParametricContext<DataType>::b_[j-1];
	ParametricContext<DataType>::partialSums_[i][j] = a_sum * a_sum / b_sum;
      }
    }  
  }

  template<typename DataType>
  void
  RationalScoreContext<DataType>::compute_scores_AVX256() {

    std::size_t n_ = ParametricContext<DataType>::n_;
    using vvec = std::vector<std::vector<DataType>>;

    vvec a_sums_ = vvec{n_+1, std::vector<DataType>(n_+1,0.)};
    vvec b_sums_ = vvec{n_+1, std::vector<DataType>(n_+1,0.)};
    
    std::vector<DataType> a_ = std::move(ParametricContext<DataType>::a_);
    std::vector<DataType> b_ = std::move(ParametricContext<DataType>::b_);
    vvec partialSums_ = std::move(ParametricContext<DataType>::partialSums_);

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

	partialSums_[i][j]   = a_sums_[j][i] * a_sums_[j][i] / b_sums_[j][i];
	partialSums_[i+1][j] = a_sums_[j][i+1] * a_sums_[j][i+1] / b_sums_[j][i+1];
	partialSums_[i+2][j] = a_sums_[j][i+2] * a_sums_[j][i+2] / b_sums_[j][i+2];
	partialSums_[i+3][j] = a_sums_[j][i+3] * a_sums_[j][i+3] / b_sums_[j][i+3];
      }
    
      for(;i<j;++i) {
	a_sums_[j][i] = a_sums_[j-1][i] + a_[j-1];
	b_sums_[j][i] = b_sums_[j-1][i] + b_[j-1];      
	partialSums_[i][j] = a_sums_[j][i] * a_sums_[j][i] / b_sums_[j][i];
      }
    }    

    ParametricContext<DataType>::a_ = std::move(a_);
    ParametricContext<DataType>::b_ = std::move(b_);
    ParametricContext<DataType>::partialSums_ = std::move(partialSums_);
  }

  template<typename DataType>
  void
  RationalScoreContext<DataType>::compute_scores_parallel() {

    std::size_t n_ = ParametricContext<DataType>::n_;
    const std::size_t numThreads = std::min(n_-2, NUMTHREADS);

    auto task_ab_block = [this, n_](int ind1, int ind2) {
      for (int i=ind1; i<ind2; ++i) {
	DataType a_sum=0., b_sum=0.;
	for (int j=i+1; j<=n_; ++j) {
	  a_sum += ParametricContext<DataType>::a_[j-1];	 
	  b_sum += ParametricContext<DataType>::b_[j-1];
	  ParametricContext<DataType>::partialSums_[i][j] = a_sum * a_sum / b_sum;
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


}
#endif
