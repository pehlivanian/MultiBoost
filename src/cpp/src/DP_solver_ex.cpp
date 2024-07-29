#include "DP_solver_ex.hpp"

void print_subsets(std::vector<std::vector<int> >& subsets, const std::vector<double>& a, const std::vector<double>& b) {
  std::cout << "SUBSETS\n";
  std::cout << "[\n";
  std::for_each(subsets.begin(), subsets.end(), [&a, &b](std::vector<int>& subset){
		  std::cout << "  [size: " << subset.size() << "] ";
		  std::cout << "[";
		  std::sort(subset.begin(), subset.end());
		  std::copy(subset.begin(), subset.end(),
			    std::ostream_iterator<int>(std::cout, " "));
		  std::cout << "] ";

		  double sum_a=0., sum_b=0.;
		  for (const int& ind : subset) {
		    sum_a+=a[ind];
		    sum_b+=b[ind];
		  }
		  std::cout << "[avg. priority: " << sum_a/sum_b 
			    << " wtd. priority: " << sum_a * subset.size() / sum_b
			    << " cum ratl score: " << sum_a*sum_a/sum_b << "]\n";
		});
  std::cout << "]";
}

void print_scores(std::vector<double>& scores) {
  std::cout << "SCORES\n";
  std::cout << "[\n";
  std::for_each(scores.begin(), scores.end(), [](double score) {
		  std::cout << "[ " << score << "]\n";
		});
  std::cout << "]";
}

double mixture_of_uniforms(int n) {
  int bin = 1;
  std::random_device rnd_device;
  std::mt19937 mersenne_engine{rnd_device()};
  std::uniform_real_distribution<double> distmixer(0., 1.);
  std::uniform_real_distribution<double> dista(0., 1./static_cast<double>(n));
  
  double mixer = distmixer(mersenne_engine);

  while (bin < n) {
    if (mixer < static_cast<double>(bin)/static_cast<double>(n))
      break;
    ++bin;
  }
  return dista(mersenne_engine) + static_cast<double>(bin)-1.;
}

void sort_by_priority(std::vector<double>& a, std::vector<double>& b) {
  const int n = a.size();

  std::vector<int> ind(n);
  std::iota(ind.begin(), ind.end(), 0);
  
  std::stable_sort(ind.begin(), ind.end(),
		   [&a, &b](int i, int j) {
		     return (a[i]/b[i]) < (a[j]/b[j]);
		   });
  
  std::vector<int> priority_sortind = ind;
  
  std::vector<double> a_s(n), b_s(n);
  for (int i=0; i<n; ++i) {
    a_s[i] = a[ind[i]];
    b_s[i] = b[ind[i]];
  }

  std::copy(a_s.cbegin(), a_s.cend(), a.begin());
  std::copy(b_s.cbegin(), b_s.cend(), b.begin());
		   
}

auto main() -> int {

  /*
  constexpr int n = 5000;
  constexpr int T = 20;
  constexpr int NUM_TRIALS = 5;
  constexpr int NUM_BEST_SWEEP_OLS_TRIALS = 10;
  constexpr int NUM_DISTRIBUTIONS_IN_MIX = 4;
    
  int cluster_sum = 0;

  std::random_device rnd_device;
  std::mt19937 mersenne_engine{rnd_device()};
  std::uniform_real_distribution<double> dista(-10., 10.);
  std::uniform_real_distribution<double> distb(0., 10.);

  // auto gena = [&dista, &mersenne_engine]() { return dista(mersenne_engine); };
  auto gena = []() { return mixture_of_uniforms(NUM_DISTRIBUTIONS_IN_MIX); };
  auto genb = [&distb, &mersenne_engine]() { return distb(mersenne_engine); };

  std::vector<double> a(n), b(n);

  for (int i=0; i<NUM_BEST_SWEEP_OLS_TRIALS; ++i) {
    std::generate(a.begin(), a.end(), gena);
    std::generate(b.begin(), b.end(), genb);

    auto dp = DPSolver<double>(n, T, a, b,
			       objective_fn::Gaussian,
			       true,
			       true,
			       0.0,
			       1.0,
			       false,
			       true);

    auto dp_opt = dp.get_optimal_subsets_extern();
    auto dp_scores = dp.get_score_by_subset_extern();
    int num_clusters = dp.get_optimal_num_clusters_OLS_extern();

    std::cout << "TRIAL: " << i << " optimal number of clusters: " 
	      << num_clusters << " vs theoretical: " << NUM_DISTRIBUTIONS_IN_MIX
	      << std::endl;
    cluster_sum += num_clusters;
    
  }
  std::cout << "CLUSTER COUNT: " 
	    << static_cast<double>(cluster_sum)/static_cast<double>(NUM_BEST_SWEEP_OLS_TRIALS) 
	    << std::endl;

  for (int i=0; i<NUM_TRIALS; ++i) {
    constexpr int m = 25;
    constexpr int S = 2;

    std::vector<double> c(m), d(m);
    
    std::uniform_real_distribution<double> distc(-10., 10.);
    std::uniform_real_distribution<double> distd(0., 1.);
    
    auto genc = [&distc, &mersenne_engine]() { return distc(mersenne_engine); };
    auto gend = [&distd, &mersenne_engine]() { return distd(mersenne_engine); };

    std::generate(c.begin(), c.end(), genc);
    std::generate(d.begin(), d.end(), gend);
    
    auto dp_rp = DPSolver<double>(m, S, c, d,
				  objective_fn::Gaussian,
				  true,
				  true);
    auto dp_rp_opt = dp_rp.get_optimal_subsets_extern();
    auto dp_rp_scores = dp_rp.get_score_by_subset_extern();
    auto dp_rp_score = dp_rp.get_optimal_score_extern();

    auto dp_mc = DPSolver<double>(m, S, c, d, 
				  objective_fn::Gaussian, 
				  false,
				  true);
    auto dp_mc_opt = dp_mc.get_optimal_subsets_extern();
    auto dp_mc_scores = dp_mc.get_score_by_subset_extern();
    auto dp_mc_score = dp_mc.get_optimal_score_extern();
    
    std::cout << "\n========\nTRIAL: " << i << "\n========\n";
    std::cout << "\na: { ";
    std::copy(c.begin(), c.end(), std::ostream_iterator<double>(std::cout, " "));
    std::cout << "}" << std::endl;
    std::cout << "b: { ";
    std::copy(d.begin(), d.end(), std::ostream_iterator<double>(std::cout, " "));
    std::cout << "}" << std::endl;
    std::cout << "\nDPSolver risk partitioning subsets:\n";
    std::cout << "================\n";
    print_subsets(dp_rp_opt);
    std::cout << "\nSubset scores: ";
    std::copy(dp_rp_scores.begin(), dp_rp_scores.end(), std::ostream_iterator<double>(std::cout, " "));
    std::cout << std::endl;
    std::cout << "\nPartition score: ";
    std::cout << dp_rp_score << std::endl;
    std::cout << "\n\nDPSolver multiple clustering subsets:\n";
    std::cout << "================\n";
    print_subsets(dp_mc_opt);
    std::cout << "\nSubset scores: ";
    std::copy(dp_mc_scores.begin(), dp_mc_scores.end(), std::ostream_iterator<double>(std::cout, " "));
    std::cout << std::endl;
    std::cout << "\nPartition score: ";
    std::cout << dp_mc_score << std::endl;
    std::cout << "=================\n\n";
  
  }
  */

  /*
  // Large-scale example
  const std::size_t n = 4000;
  const std::size_t T = 400;

  std::random_device rnd_device;
  std::mt19937 mersenne_engine{rnd_device()};
  std::uniform_real_distribution<double> dista(-10., 10.);
  std::uniform_real_distribution<double> distb(0., 10.);
  
  auto gena = [&dista, &mersenne_engine](){ return dista(mersenne_engine); };
  auto genb = [&distb, &mersenne_engine](){ return distb(mersenne_engine); };

  std::vector<double> a(n), b(n);

  // First run
  std::generate(a.begin(), a.end(), gena);
  std::generate(b.begin(), b.end(), genb);

  auto dp = DPSolver<double>(n, T, a, b,
			     objective_fn::RationalScore,
			     true,
			     true);
  auto dp_opt = dp.get_optimal_subsets_extern();
  auto dp_scores = dp.get_score_by_subset_extern();

  std::cout << "FIRST CASE RISK PARTITIONING OBJECTIVE\n";
  std::cout << "======================================\n";
  print_subsets(dp_opt);
  print_scores(dp_scores);
  */

  const int n = 50;
  const int T = 5;

  std::random_device rnd_device;
  std::mt19937 mersenne_engine{rnd_device()};
  std::uniform_real_distribution<double> dista(-10., 10.);
  std::uniform_real_distribution<double> distb(0., 10.);

  auto gena = [&mersenne_engine, &dista](){ return dista(mersenne_engine); };
  auto genb = [&mersenne_engine, &distb](){ return distb(mersenne_engine); };

  std::vector<double> a0(n), b0(n), a1(n), b1(n), a2(n), b2(n);

  std::generate(a0.begin(), a0.end(), gena);
  std::generate(b0.begin(), b0.end(), genb);

  sort_by_priority(a0, b0);

  for (std::size_t i=0; i<a0.size(); ++i) {
    a1[i] = a0[i]/b0[i];
    b1[i] = 1.;
    a2[i] = 1.;
    b2[i] = b0[i]/a0[i];
  }

  bool risk_partitioning_objective = false;
  bool use_rational_optimization = true;

  auto dp00 = DPSolver<double>(n, T, a0, b0,
			       objective_fn::RationalScore,
			       risk_partitioning_objective,
			       use_rational_optimization);

  auto dp00_opt = dp00.get_optimal_subsets_extern();
  auto dp00_scores = dp00.get_score_by_subset_extern();

  auto dp01 = DPSolver<double>(n, T, a0, b0,
			       objective_fn::RationalScore,
			       !risk_partitioning_objective,
			       use_rational_optimization);

  auto dp01_opt = dp01.get_optimal_subsets_extern();
  auto dp01_scores = dp01.get_score_by_subset_extern();

  auto dp1 = DPSolver<double>(n, T, a1, b1,
			      objective_fn::RationalScore,
			      risk_partitioning_objective,
			      use_rational_optimization);


  auto dp1_opt = dp1.get_optimal_subsets_extern();
  auto dp1_scores = dp1.get_score_by_subset_extern();

  auto dp2 = DPSolver<double>(n, T, a2, b2,
			      objective_fn::RationalScore,
			      risk_partitioning_objective,
			      use_rational_optimization);


  auto dp2_opt = dp2.get_optimal_subsets_extern();
  auto dp2_scores = dp2.get_score_by_subset_extern();

  print_subsets(dp00_opt, a0, b0);
  std::cout << "\n dp00 Subset scores: ";
  std::copy(dp00_scores.begin(), dp00_scores.end(), std::ostream_iterator<double>(std::cout, " "));
  std::cout << std::endl;

  print_subsets(dp01_opt, a0, b0);
  std::cout << "\n dp01 Subset scores: ";
  std::copy(dp01_scores.begin(), dp01_scores.end(), std::ostream_iterator<double>(std::cout, " "));
  std::cout << std::endl;

  print_subsets(dp1_opt, a1, b1);
  std::cout << "\n dp1 Subset scores: ";
  std::copy(dp1_scores.begin(), dp1_scores.end(), std::ostream_iterator<double>(std::cout, " "));
  std::cout << std::endl;

  print_subsets(dp2_opt, a2, b2);
  std::cout << "\n dp2 Subset scores: ";
  std::copy(dp2_scores.begin(), dp2_scores.end(), std::ostream_iterator<double>(std::cout, " "));
  std::cout << std::endl;
  
  /*  
  const int n = 50;
  const int T = 4;
  
  std::random_device rnd_device;
  std::mt19937 mersenne_engine{rnd_device()};
  std::uniform_real_distribution<double> dista(-10. ,10.);
  std::uniform_real_distribution<double> distb(0., 10.);

  auto gena = [&dista, &mersenne_engine](){ return dista(mersenne_engine); };
  auto genb = [&distb, &mersenne_engine](){ return distb(mersenne_engine); };

  std::vector<double> a(n), b(n);

  // First run
  std::generate(a.begin(), a.end(), gena);
  std::generate(b.begin(), b.end(), genb);

  sort_by_priority(a, b);

  auto dp11 = DPSolver<double>(n, T, a, b,
			       objective_fn::RationalScore,
			       true,
			       true);

  auto dp11_opt = dp11.get_optimal_subsets_extern();
  auto dp11_scores = dp11.get_score_by_subset_extern();

  std::cout << "FIRST CASE RISK PARTITIONING OBJECTIVE\n";
  std::cout << "======================================\n";
  print_subsets(dp11_opt);
  print_scores(dp11_scores);

  auto dp12 = DPSolver<double>(n, T, a, b,
			       objective_fn::RationalScore,
			       false,
			       true);

  auto dp12_opt = dp12.get_optimal_subsets_extern();
  auto dp12_scores = dp12.get_score_by_subset_extern();

  std::cout << "FIRST CASE MULTIPLE CLUSTERING OBJECTIVE\n";
  std::cout << "========================================\n";
  print_subsets(dp12_opt);
  print_scores(dp12_scores);

  // Second run
  std::transform(a.begin(), a.end(), a.begin(), [](double a){ return -1. * a; });

  auto dp21 = DPSolver<double>(n, T, a, b,
			       objective_fn::RationalScore,
			       true,
			       true);

  auto dp21_opt = dp21.get_optimal_subsets_extern();
  auto dp21_scores = dp21.get_score_by_subset_extern();

  std::cout << "SECOND CASE RISK PARTITIONING OBJECTIVE\n";
  std::cout << "=======================================\n";
  print_subsets(dp21_opt);
  print_scores(dp21_scores);

  auto dp22 = DPSolver<double>(n, T, a, b,
			       objective_fn::RationalScore,
			       false,
			       true);

  auto dp22_opt = dp22.get_optimal_subsets_extern();
  auto dp22_scores = dp22.get_score_by_subset_extern();

  std::cout << "SECOND CASE MULTIPLE CLUSTERING OBJECTIVE\n";
  std::cout << "=========================================\n";
  print_subsets(dp22_opt);
  print_scores(dp22_scores);
  */

  return 0;
}
