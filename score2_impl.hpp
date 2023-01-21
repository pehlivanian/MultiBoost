#ifndef __SCORE2_IMPL_HPP__
#define __SCORE2_IMPL_HPP__

const int NUMTHREADS = 6;

namespace Objectives {

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
  ParametricContext<DataType>::init_() {
    // XXX
    // compute_scores_parallel();
    compute_scores();
  }

  template<typename DataType>
  DataType 
  PoissonContext<DataType>::compute_score_multclust(int i, int j) {    
    DataType C = std::accumulate(a_.cbegin()+i, a_.cbegin()+j, 0.);
    DataType B = std::accumulate(b_.cbegin()+i, b_.cbegin()+j, 0.);
    return (C>B)? C*std::log(C/B) + B - C : 0.;
  }

  template<typename DataType>
  DataType 
  PoissonContext<DataType>::compute_score_riskpart(int i, int j) {
    DataType C = std::accumulate(a_.cbegin()+i, a_.cbegin()+j, 0.);
    DataType B = std::accumulate(b_.cbegin()+i, b_.cbegin()+j, 0.);
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
    DataType score = a_sums_[i][j]*std::log(a_sums_[i][j]/b_sums_[i][j]);
    return score;
  }

  template<typename DataType>
  DataType 
  PoissonContext<DataType>::compute_score_multclust_optimized(int i, int j) {
    DataType score = (a_sums_[i][j] > b_sums_[i][j]) ? a_sums_[i][j]*std::log(a_sums_[i][j]/b_sums_[i][j]) + b_sums_[i][j] - a_sums_[i][j]: 0.;
    return score;
  }

  template<typename DataType>
  DataType 
  GaussianContext<DataType>::compute_score_multclust(int i, int j) {
    DataType C = std::accumulate(a_.cbegin()+i, a_.cbegin()+j, 0.);
    DataType B = std::accumulate(b_.cbegin()+i, b_.cbegin()+j, 0.);
    return (C>B)? .5*(std::pow(C,2)/B + B) - C : 0.;
  }

  template<typename DataType>
  DataType 
  GaussianContext<DataType>::compute_score_riskpart(int i, int j) {
    DataType C = std::accumulate(a_.cbegin()+i, a_.cbegin()+j, 0.);
    DataType B = std::accumulate(b_.cbegin()+i, b_.cbegin()+j, 0.);
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
    DataType score = (a_sums_[i][j] > b_sums_[i][j]) ? .5*(std::pow(a_sums_[i][j], 2)/b_sums_[i][j] + b_sums_[i][j]) - a_sums_[i][j] : 0.;
    return score;
  }
   
  template<typename DataType>
  DataType 
  GaussianContext<DataType>::compute_score_riskpart_optimized(int i, int j) {
    DataType score = a_sums_[i][j]*a_sums_[i][j]/2./b_sums_[i][j];
    return score;
  }

  template<typename DataType>
  DataType 
  RationalScoreContext<DataType>::compute_score_multclust(int i, int j) {
    DataType score = std::pow(std::accumulate(a_.cbegin()+i, a_.cbegin()+j, 0.), 2) /
      std::accumulate(b_.cbegin()+i, b_.cbegin()+j, 0.);
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
    DataType score = a_sums_[i][j] * a_sums_[i][j] / b_sums_[i][j];
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
