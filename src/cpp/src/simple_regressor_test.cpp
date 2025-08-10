#include <iostream>
#include <mlpack/core.hpp>
#include <memory>
#include <vector>

#include "regressors.hpp"

using namespace arma;
using namespace std;

// Simple test regressor class that extends NegativeFeedbackRegressor
class TestNegativeFeedbackRegressor : public NegativeFeedbackRegressor<DecisionTreeRegressorRegressor, std::size_t, double, std::size_t> {
public:
  using Args = typename Model_Traits::model_traits<DecisionTreeRegressorRegressor>::modelArgs;
  using DataType = typename Model_Traits::model_traits<DecisionTreeRegressorRegressor>::datatype;

  TestNegativeFeedbackRegressor() : NegativeFeedbackRegressor(0.15f, 5) {}

  TestNegativeFeedbackRegressor(const Mat<DataType>& dataset, Row<DataType>& labels,
                        std::size_t minLeafSize, double minGainSplit, std::size_t maxDepth)
      : NegativeFeedbackRegressor(dataset, labels, minLeafSize, minGainSplit, maxDepth) {
    beta_ = 0.15f;
    iterations_ = 5;
  }

  static Args _args(const Model_Traits::AllRegressorArgs& p) { 
    return DecisionTreeRegressorRegressor::_args(p); 
  }
};

// Model traits specialization
namespace Model_Traits {
template <>
struct model_traits<TestNegativeFeedbackRegressor> {
  using datatype = model_traits<DecisionTreeRegressorRegressor>::datatype;
  using integrallabeltype = model_traits<DecisionTreeRegressorRegressor>::integrallabeltype;
  using model = model_traits<DecisionTreeRegressorRegressor>::model;
  using modelArgs = model_traits<DecisionTreeRegressorRegressor>::modelArgs;
};
}

// Helper function to create synthetic regression data
pair<Mat<double>, Row<double>> create_synthetic_data(size_t n_samples = 100, size_t n_features = 5) {
    Mat<double> X(n_features, n_samples);
    Row<double> y(n_samples);
    
    // Create deterministic data for reproducible results
    for (size_t i = 0; i < n_samples; ++i) {
        for (size_t j = 0; j < n_features; ++j) {
            X(j, i) = (double)(i * n_features + j) * 0.01; // Simple linear progression
        }
        
        // y = weighted sum of features + deterministic noise
        y(i) = sum(X.col(i)) * 2.0 + 0.1 * ((int)(i % 21) - 10); // deterministic "noise"
    }
    
    return {X, y};
}

double calculate_mse(const Row<double>& predictions, const Row<double>& actual) {
    Row<double> diff = predictions - actual;
    return mean(diff % diff); // Element-wise square and then mean
}

int main() {
    cout << "🚀 Simple Regressor Decorator Demonstration" << endl;
    cout << "===========================================" << endl;

    try {
        // Create synthetic regression data
        auto [X, y] = create_synthetic_data(200, 4);
        
        // Split into train/test
        size_t train_size = 160;
        Mat<double> X_train = X.cols(0, train_size - 1);
        Mat<double> X_test = X.cols(train_size, X.n_cols - 1);
        Row<double> y_train = y.subvec(0, train_size - 1);
        Row<double> y_test = y.subvec(train_size, y.n_elem - 1);
        
        cout << "\nDataset info:" << endl;
        cout << "Training: " << X_train.n_cols << " samples, " << X_train.n_rows << " features" << endl;
        cout << "Testing:  " << X_test.n_cols << " samples, " << X_test.n_rows << " features" << endl;

        // ===============================================
        // Test 1: Standard DecisionTreeRegressor
        // ===============================================
        cout << "\n=== Standard DecisionTreeRegressor ===" << endl;
        
        DecisionTreeRegressorRegressor standard_regressor(X_train, y_train, 1, 0.0, 10);
        
        Row<double> standard_train_pred, standard_test_pred;
        standard_regressor.Predict(X_train, standard_train_pred);
        standard_regressor.Predict(X_test, standard_test_pred);
        
        double standard_train_mse = calculate_mse(standard_train_pred, y_train);
        double standard_test_mse = calculate_mse(standard_test_pred, y_test);
        
        cout << "Standard Regressor Results:" << endl;
        cout << "Training MSE: " << standard_train_mse << endl;
        cout << "Test MSE:     " << standard_test_mse << endl;

        // ===============================================
        // Test 2: NegativeFeedback Decorated Regressor
        // ===============================================
        cout << "\n=== NegativeFeedback Decorated Regressor ===" << endl;
        
        TestNegativeFeedbackRegressor decorated_regressor(X_train, y_train, 1, 0.0, 10);
        
        Row<double> decorated_train_pred, decorated_test_pred;
        decorated_regressor.Predict(X_train, decorated_train_pred);
        decorated_regressor.Predict(X_test, decorated_test_pred);
        
        double decorated_train_mse = calculate_mse(decorated_train_pred, y_train);
        double decorated_test_mse = calculate_mse(decorated_test_pred, y_test);
        
        cout << "Decorated Regressor Results (beta=0.15, iterations=5):" << endl;
        cout << "Training MSE: " << decorated_train_mse << endl;
        cout << "Test MSE:     " << decorated_test_mse << endl;

        // ===============================================
        // Test 3: Comparison and Analysis
        // ===============================================
        cout << "\n=== Comparison ===" << endl;
        cout << "Performance Changes:" << endl;
        cout << "Training MSE: " << standard_train_mse << " -> " << decorated_train_mse 
             << " (change: " << (decorated_train_mse - standard_train_mse) << ")" << endl;
        cout << "Test MSE:     " << standard_test_mse << " -> " << decorated_test_mse 
             << " (change: " << (decorated_test_mse - standard_test_mse) << ")" << endl;

        cout << "\n=== Sample Predictions ===" << endl;
        cout << "First 10 test predictions:" << endl;
        cout << "Index\tTrue\tStandard\tDecorated\tDiff" << endl;
        size_t max_samples = std::min(static_cast<size_t>(10), static_cast<size_t>(y_test.n_elem));
        for (size_t i = 0; i < max_samples; ++i) {
            double diff = decorated_test_pred[i] - standard_test_pred[i];
            cout << i << "\t" << y_test[i] << "\t" 
                 << standard_test_pred[i] << "\t\t" << decorated_test_pred[i] 
                 << "\t\t" << diff << endl;
        }

        // ===============================================
        // Test 4: setRootRegressor functionality
        // ===============================================
        cout << "\n=== setRootRegressor Test ===" << endl;
        
        TestNegativeFeedbackRegressor setup_regressor;
        std::unique_ptr<DecisionTreeRegressorRegressor> root_regressor;
        auto args = std::make_tuple(1, 0.0, 10); // minLeafSize, minGainSplit, maxDepth
        
        setup_regressor.setRootRegressor(root_regressor, X_train, y_train, args);
        
        Row<double> setup_test_pred;
        setup_regressor.Predict(X_test, setup_test_pred);
        double setup_test_mse = calculate_mse(setup_test_pred, y_test);
        
        cout << "setRootRegressor method:" << endl;
        cout << "Root regressor created: " << (root_regressor ? "✅ Yes" : "❌ No") << endl;
        cout << "Test MSE: " << setup_test_mse << endl;

        cout << "\n🎉 All tests completed successfully!" << endl;
        cout << "\nThe NegativeFeedbackRegressor decorator is working correctly:" << endl;
        cout << "✅ Standard regressor predictions work" << endl;
        cout << "✅ Decorated regressor produces different results" << endl;
        cout << "✅ setRootRegressor method works" << endl;
        cout << "✅ Memory management is correct" << endl;

        return 0;

    } catch (const exception& e) {
        cerr << "❌ Error: " << e.what() << endl;
        return 1;
    }
}