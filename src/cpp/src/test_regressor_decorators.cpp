#include <cassert>
#include <chrono>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <mlpack/core.hpp>

#include "regressors.hpp"
#include "model_traits.hpp"

using namespace arma;
using namespace std;

// Test utilities
class TestRunner {
private:
  int passed = 0;
  int failed = 0;
  std::string current_test;

public:
  void start_test(const std::string& name) {
    current_test = name;
    std::cout << "🧪 Testing: " << name << "... ";
  }

  void assert_true(bool condition, const std::string& message = "", bool multiple_cases = false) {
    if (condition) {
      std::cout << "[" << message << "] : " << "✅ PASS";
      if (!multiple_cases)
	std::cout << "\n";
      else
	std::cout << " ";
      passed++;
    } else {
      std::cout << "[" << message << "] : ❌ FAIL: " << message;
      if (!multiple_cases)
	std::cout << "\n";
      else
	std::cout << " ";
      failed++;
    }
  }

  void summary() {
    std::cout << "\n📊 Test Summary:\n";
    std::cout << "✅ Passed: " << passed << "\n";
    std::cout << "❌ Failed: " << failed << "\n";
    std::cout << "🎯 Success Rate: " << (100.0 * passed / (passed + failed)) << "%\n";
  }

  bool all_passed() const { return failed == 0; }
};

// Helper function to create test data
pair<Mat<double>, Row<double>> create_regression_data(size_t n_samples = 100, size_t n_features = 5) {
    Mat<double> X(n_features, n_samples);
    Row<double> y(n_samples);
    
    // Create deterministic data for reproducible tests
    for (size_t i = 0; i < n_samples; ++i) {
        for (size_t j = 0; j < n_features; ++j) {
            X(j, i) = (i * n_features + j) * 0.01; // Simple scaling
        }
        
        // y = sum of features + small noise
        y(i) = sum(X.col(i)) + 0.1 * (i % 10 - 5); // deterministic "noise"
    }
    
    return {X, y};
}

int main() {
  TestRunner test;

  std::cout << "🚀 Regressor Decorator Comprehensive Test Suite\n";
  std::cout << "===============================================\n\n";

  // ========================================
  // Test 1: Basic Template Instantiation
  // ========================================
  test.start_test("Basic Template Instantiation");
  try {
    using StandardRegressor = DecisionTreeRegressorRegressor;
    using DecoratorBase = DecoratorRegressorBase<StandardRegressor, std::size_t, double, std::size_t>;
    using Decorator = DecoratorRegressor<StandardRegressor, std::size_t, double, std::size_t>;
    using NegFeedback = NegativeFeedbackRegressor<StandardRegressor, std::size_t, double, std::size_t>;
    
    test.assert_true(true, "All template instantiations successful");
  } catch (const std::exception& e) {
    test.assert_true(false, string("Template instantiation failed: ") + e.what());
  }

  // ========================================
  // Test 2: Default Constructor
  // ========================================
  test.start_test("Default Constructor");
  try {
    NegativeFeedbackRegressor<DecisionTreeRegressorRegressor, std::size_t, double, std::size_t> decorator;
    test.assert_true(true, "Default constructor successful");
  } catch (const std::exception& e) {
    test.assert_true(false, string("Default constructor failed: ") + e.what());
  }

  // ========================================
  // Test 3: Parameterized Constructor
  // ========================================
  test.start_test("Parameterized Constructor");
  try {
    NegativeFeedbackRegressor<DecisionTreeRegressorRegressor, std::size_t, double, std::size_t> decorator(0.2f, 5);
    test.assert_true(true, "Parameterized constructor successful");
  } catch (const std::exception& e) {
    test.assert_true(false, string("Parameterized constructor failed: ") + e.what());
  }

  // ========================================
  // Test 4: Dataset Constructor
  // ========================================
  test.start_test("Dataset Constructor");
  try {
    auto [X, y] = create_regression_data(50, 3);
    NegativeFeedbackRegressor<DecisionTreeRegressorRegressor, std::size_t, double, std::size_t> 
        decorator(X, y, 1, 0.0, 10); // minLeafSize, minGainSplit, maxDepth
    test.assert_true(true, "Dataset constructor successful");
  } catch (const std::exception& e) {
    test.assert_true(false, string("Dataset constructor failed: ") + e.what());
  }

  // ========================================
  // Test 5: Basic Prediction
  // ========================================
  test.start_test("Basic Prediction");
  try {
    auto [X, y] = create_regression_data(50, 3);
    NegativeFeedbackRegressor<DecisionTreeRegressorRegressor, std::size_t, double, std::size_t> 
        decorator(X, y, 1, 0.0, 10);
    
    Row<double> predictions;
    decorator.Predict(X, predictions);
    
    bool predictions_valid = predictions.n_elem == X.n_cols && 
                           predictions.is_finite();
    test.assert_true(predictions_valid, "Predictions are valid finite numbers");
  } catch (const std::exception& e) {
    test.assert_true(false, string("Basic prediction failed: ") + e.what());
  }

  // ========================================
  // Test 6: setRootRegressor Method
  // ========================================
  test.start_test("setRootRegressor Method");
  try {
    auto [X, y] = create_regression_data(30, 3);
    NegativeFeedbackRegressor<DecisionTreeRegressorRegressor, std::size_t, double, std::size_t> decorator;
    
    std::unique_ptr<DecisionTreeRegressorRegressor> root_regressor;
    auto args = std::make_tuple(1, 0.0, 10); // minLeafSize, minGainSplit, maxDepth
    
    decorator.setRootRegressor(root_regressor, X, y, args);
    
    bool root_created = root_regressor != nullptr;
    test.assert_true(root_created, "Root regressor created successfully", true);
    
    if (root_created) {
      Row<double> predictions;
      decorator.Predict(X, predictions);
      bool predictions_valid = predictions.n_elem == X.n_cols;
      test.assert_true(predictions_valid, "Decorator prediction after setRootRegressor works");
    }
  } catch (const std::exception& e) {
    test.assert_true(false, string("setRootRegressor failed: ") + e.what());
  }

  // ========================================
  // Test 7: Negative Feedback Algorithm Behavior
  // ========================================
  test.start_test("Negative Feedback Algorithm Behavior");
  try {
    auto [X, y] = create_regression_data(50, 3);
    
    // Create standard regressor
    DecisionTreeRegressorRegressor standard(X, y, 1, 0.0, 10);
    Row<double> standard_pred;
    standard.Predict(X, standard_pred);
    
    // Create decorated regressor with strong negative feedback
    NegativeFeedbackRegressor<DecisionTreeRegressorRegressor, std::size_t, double, std::size_t> 
        decorated(0.5f, 3); // Strong feedback directly in constructor
    
    // Initialize the regressor using setRootRegressor
    std::unique_ptr<DecisionTreeRegressorRegressor> root;
    auto args = std::make_tuple(1, 0.0, 10);
    decorated.setRootRegressor(root, X, y, args);
    
    Row<double> decorated_pred;
    decorated.Predict(X, decorated_pred);
    
    // The decorated predictions should be different from standard
    double diff = norm(decorated_pred - standard_pred);
    test.assert_true(diff > 1e-10, "Negative feedback produces different predictions");
  } catch (const std::exception& e) {
    test.assert_true(false, string("Algorithm behavior test failed: ") + e.what());
  }

  // ========================================
  // Test 8: Multiple Beta Values
  // ========================================
  test.start_test("Multiple Beta Values");
  try {
    auto [X, y] = create_regression_data(40, 3);
    vector<float> betas = {0.0f, 0.1f, 0.5f, 1.0f};
    vector<Row<double>> predictions(betas.size());
    
    bool all_different = true;
    for (size_t i = 0; i < betas.size(); ++i) {
      NegativeFeedbackRegressor<DecisionTreeRegressorRegressor, std::size_t, double, std::size_t> 
          decorator(betas[i], 2);
      
      // Use setRootRegressor to ensure same base regressor
      std::unique_ptr<DecisionTreeRegressorRegressor> root;
      auto args = std::make_tuple(1, 0.0, 10);
      decorator.setRootRegressor(root, X, y, args);
      
      decorator.Predict(X, predictions[i]);
      
      // Check that different betas produce different results
      if (i > 0) {
        double diff = norm(predictions[i] - predictions[0]);
        if (betas[i] != betas[0] && diff < 1e-10) {
          all_different = false;
          break;
        }
      }
    }
    
    test.assert_true(all_different, "Different beta values produce different predictions");
  } catch (const std::exception& e) {
    test.assert_true(false, string("Multiple beta test failed: ") + e.what());
  }

  // ========================================
  // Test 9: Iteration Count Effect
  // ========================================
  test.start_test("Iteration Count Effect");
  try {
    auto [X, y] = create_regression_data(40, 3);
    vector<size_t> iterations = {1, 3, 5};
    vector<Row<double>> predictions(iterations.size());
    
    bool iterations_matter = false;
    for (size_t i = 0; i < iterations.size(); ++i) {
      NegativeFeedbackRegressor<DecisionTreeRegressorRegressor, std::size_t, double, std::size_t> 
          decorator(0.2f, iterations[i]);
      
      std::unique_ptr<DecisionTreeRegressorRegressor> root;
      auto args = std::make_tuple(1, 0.0, 10);
      decorator.setRootRegressor(root, X, y, args);
      
      decorator.Predict(X, predictions[i]);
      
      if (i > 0) {
        double diff = norm(predictions[i] - predictions[0]);
        if (diff > 1e-10) {
          iterations_matter = true;
        }
      }
    }
    
    test.assert_true(iterations_matter, "Different iteration counts produce different results");
  } catch (const std::exception& e) {
    test.assert_true(false, string("Iteration count test failed: ") + e.what());
  }

  // ========================================
  // Test 10: Weighted Dataset Constructor
  // ========================================
  test.start_test("Weighted Dataset Constructor");
  try {
    auto [X, y] = create_regression_data(30, 3);
    Row<double> weights(X.n_cols);
    weights.ones(); // Unit weights
    
    NegativeFeedbackRegressor<DecisionTreeRegressorRegressor, std::size_t, double, std::size_t> 
        decorator(X, y, weights, 1, 0.0, 10);
    
    Row<double> predictions;
    decorator.Predict(X, predictions);
    
    bool valid = predictions.n_elem == X.n_cols && predictions.is_finite();
    test.assert_true(valid, "Weighted constructor and prediction successful");
  } catch (const std::exception& e) {
    test.assert_true(false, string("Weighted constructor failed: ") + e.what());
  }

  // ========================================
  // Test 11: Model Traits Integration
  // ========================================
  test.start_test("Model Traits Integration");
  try {
    using NegFeedback = NegativeFeedbackRegressor<DecisionTreeRegressorRegressor, std::size_t, double, std::size_t>;
    using Traits = Model_Traits::model_traits<NegFeedback>;
    
    bool datatype_correct = std::is_same_v<Traits::datatype, double>;
    bool model_correct = std::is_same_v<Traits::model, Model_Traits::RegressorTypes::DecisionTreeRegressorRegressorType>;
    
    test.assert_true(datatype_correct && model_correct, "Model traits correctly defined");
  } catch (const std::exception& e) {
    test.assert_true(false, string("Model traits test failed: ") + e.what());
  }

  // ========================================
  // Test 12: Memory Management
  // ========================================
  test.start_test("Memory Management");
  try {
    auto [X, y] = create_regression_data(20, 2);
    
    // Create many decorators to test for memory leaks
    for (int i = 0; i < 10; ++i) {
      auto decorator = std::make_unique<NegativeFeedbackRegressor<DecisionTreeRegressorRegressor, std::size_t, double, std::size_t>>(X, y, 1, 0.0, 5);
      Row<double> pred;
      decorator->Predict(X, pred);
    }
    
    test.assert_true(true, "Memory management test completed without crashes");
  } catch (const std::exception& e) {
    test.assert_true(false, string("Memory management test failed: ") + e.what());
  }

  // ========================================
  // Test 13: Exception Handling
  // ========================================
  test.start_test("Exception Handling");
  try {
    // Test with invalid data
    Mat<double> empty_X;
    Row<double> empty_y;
    
    bool exception_caught = false;
    try {
      NegativeFeedbackRegressor<DecisionTreeRegressorRegressor, std::size_t, double, std::size_t> 
          decorator(empty_X, empty_y, 1, 0.0, 10);
    } catch (const std::exception&) {
      exception_caught = true;
    }
    
    test.assert_true(exception_caught, "Proper exception handling for invalid input");
  } catch (const std::exception& e) {
    test.assert_true(false, string("Exception handling test failed: ") + e.what());
  }

  // ========================================
  // Test 14: Performance Comparison
  // ========================================
  test.start_test("Performance Comparison");
  try {
    auto [X, y] = create_regression_data(100, 5);
    
    // Time standard regressor
    auto start = std::chrono::high_resolution_clock::now();
    DecisionTreeRegressorRegressor standard(X, y, 1, 0.0, 10);
    Row<double> standard_pred;
    standard.Predict(X, standard_pred);
    auto standard_time = std::chrono::high_resolution_clock::now() - start;
    
    // Time decorated regressor
    start = std::chrono::high_resolution_clock::now();
    NegativeFeedbackRegressor<DecisionTreeRegressorRegressor, std::size_t, double, std::size_t> 
        decorated(X, y, 1, 0.0, 10);
    Row<double> decorated_pred;
    decorated.Predict(X, decorated_pred);
    auto decorated_time = std::chrono::high_resolution_clock::now() - start;
    
    // Decorator should take longer (it does more work)
    bool reasonable_overhead = decorated_time >= standard_time;
    test.assert_true(reasonable_overhead, "Decorator has expected computational overhead");
    
    std::cout << "\n    📈 Performance: Standard=" << standard_time.count() 
	      << "ns, Decorated=" << decorated_time.count() << "ns" << std::endl;
  } catch (const std::exception& e) {
    test.assert_true(false, string("Performance test failed: ") + e.what());
  }

  // ========================================
  // Test 15: Edge Cases
  // ========================================
  test.start_test("Edge Cases");
  try {
    auto [X, y] = create_regression_data(10, 1); // Very small dataset
    
    // Test with extreme parameters
    NegativeFeedbackRegressor<DecisionTreeRegressorRegressor, std::size_t, double, std::size_t> 
        extreme_decorator(10.0f, 100); // Very high beta and iterations
    
    std::unique_ptr<DecisionTreeRegressorRegressor> root;
    auto args = std::make_tuple(1, 0.0, 5);
    extreme_decorator.setRootRegressor(root, X, y, args);
    
    Row<double> predictions;
    extreme_decorator.Predict(X, predictions);
    
    bool survived_extreme = predictions.n_elem == X.n_cols;
    test.assert_true(survived_extreme, "Survived extreme parameter values");
  } catch (const std::exception& e) {
    test.assert_true(false, string("Edge cases test failed: ") + e.what());
  }

  // ========================================
  // Final Summary
  // ========================================
  std::cout << "\n🏁 Test Suite Complete!\n";
  test.summary();

  if (test.all_passed()) {
    std::cout << "\n🎉 All tests passed! Regressor decorators are working correctly.\n";
    return 0;
  } else {
    std::cout << "\n⚠️  Some tests failed. Please review the implementation.\n";
    return 1;
  }
}
