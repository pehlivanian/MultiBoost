#include "DP_solver_ex.hpp"

void print_subsets(std::vector<std::vector<int> >& subsets) {
  std::cout << "SUBSETS\n";
  std::cout << "[\n";
  std::for_each(subsets.begin(), subsets.end(), [](std::vector<int>& subset){
		  std::cout << "[";
		  std::copy(subset.begin(), subset.end(),
			    std::ostream_iterator<int>(std::cout, " "));
		  std::cout << "]\n";
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


auto main() -> int {
 
  constexpr int n = 5000;
  constexpr int T = 20;
  constexpr int NUM_TRIALS = 5;
  constexpr int NUM_BEST_SWEEP_OLS_TRIALS = 10;
  constexpr int NUM_DISTRIBUTIONS_IN_MIX = 4;
    
  int cluster_sum = 0;

  std::random_device rnd_device;
  std::mt19937 mersenne_engine{rnd_device()};
  std::uniform_real_distribution<double> dista(-10., 10.);
  std::uniform_real_distribution<double> distb(1., 1.);

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
    
    auto ltss = LTSSSolver(m, c, d);
    auto ltss_opt = ltss.get_optimal_subset_extern();
    auto ltss_score = ltss.get_optimal_score_extern();
    
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
    std::cout << "\nLTSSSolver subset:\n";
    std::cout << "=================\n";
    std::copy(ltss_opt.begin(), ltss_opt.end(), std::ostream_iterator<int>(std::cout, " "));
    std::cout << "\n";
    std::cout << "\nScore: ";
    std::cout << ltss_score << "\n";
  }

  return 0;
}
