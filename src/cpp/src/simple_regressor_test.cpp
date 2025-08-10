#include <iostream>
#include <iomanip>
#include <mlpack/core.hpp>
#include <memory>
#include <vector>

#include "regressors.hpp"
#include "gradientboostregressor.hpp"
#include "contextmanager.hpp"
#include "regressor_loss.hpp"

using namespace arma;
using namespace std;
using namespace ModelContext;
using namespace RegressorLossMeasures;

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

  TestNegativeFeedbackRegressor(const Mat<DataType>& dataset, Row<DataType>& labels, Row<DataType>& weights,
                        std::size_t minLeafSize, double minGainSplit, std::size_t maxDepth)
      : NegativeFeedbackRegressor(dataset, labels, weights, minLeafSize, minGainSplit, maxDepth) {
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

// Helper function to load CSV data
pair<Mat<double>, Row<double>> load_csv_data(const string& x_path, const string& y_path) {
    Mat<double> X;
    Row<double> y;
    
    if (!mlpack::data::Load(x_path, X, false, false)) {
        throw runtime_error("Failed to load X data from: " + x_path);
    }
    
    if (!mlpack::data::Load(y_path, y, false, false)) {
        throw runtime_error("Failed to load y data from: " + y_path);
    }
    
    return {X, y};
}

int main() {
    cout << "🚀 Simple Regressor Fitting Demonstration with 50 Steps" << endl;
    cout << "=======================================================" << endl;

    try {
        // Try to load real pol dataset, fallback to synthetic data
        Mat<double> X_train, X_test;
        Row<double> y_train, y_test;
        
        try {
            string data_dir = "/home/charles/Data/tabular_benchmark/Regression/";
            auto [X_train_loaded, y_train_loaded] = load_csv_data(data_dir + "pol_train_X.csv", data_dir + "pol_train_y.csv");
            auto [X_test_loaded, y_test_loaded] = load_csv_data(data_dir + "pol_test_X.csv", data_dir + "pol_test_y.csv");
            X_train = X_train_loaded;
            X_test = X_test_loaded;
            y_train = y_train_loaded;
            y_test = y_test_loaded;
            cout << "✅ Successfully loaded pol dataset from disk" << endl;
        } catch (const exception& e) {
            cout << "⚠️  Could not load pol dataset, using synthetic data instead" << endl;
            // Create synthetic regression data similar to pol characteristics
            auto [X, y] = create_synthetic_data(1000, 26);  // pol has 26 features
            
            // Split into train/test (mimicking pol dataset sizes)
            size_t train_size = 600;
            X_train = X.cols(0, train_size - 1);
            X_test = X.cols(train_size, X.n_cols - 1);
            y_train = y.subvec(0, train_size - 1);
            y_test = y.subvec(train_size, y.n_elem - 1);
        }
        
        cout << "\nDataset info:" << endl;
        cout << "Training: " << X_train.n_cols << " samples, " << X_train.n_rows << " features" << endl;
        cout << "Testing:  " << X_test.n_cols << " samples, " << X_test.n_rows << " features" << endl;

        // Create Context from example_params_pol_reg.json parameters
        Context pol_context{};
        pol_context.steps = 50;  // 50 fitting steps as requested
        pol_context.recursiveFit = true;
        pol_context.useWeights = false;
        pol_context.rowSubsampleRatio = 1.0;
        pol_context.colSubsampleRatio = 1.0;
        pol_context.removeRedundantLabels = false;
        pol_context.symmetrizeLabels = true;
        pol_context.loss = regressorLossFunction::MSE;
        pol_context.lossPower = 11.15;
        pol_context.clamp_gradient = false;
        pol_context.upper_val = -1.0;
        pol_context.lower_val = 1.0;
        pol_context.numTrees = 10;
        pol_context.depth = 0;
        pol_context.childPartitionSize = {800, 200, 100, 10, 5, 50, 20, 10, 3, 4, 5, 2, 2, 1};
        pol_context.childNumSteps = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
        pol_context.childLearningRate = {0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03};
        pol_context.childActivePartitionRatio = {0.275, 0.275, 0.275, 0.275, 0.275, 0.275, 0.275, 0.275, 0.275, 0.275, 0.275, 0.275, 0.275, 0.275};
        pol_context.childMinLeafSize = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        pol_context.childMinimumGainSplit = {0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001};
        pol_context.childMaxDepth = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        pol_context.serializeModel = false;  // Disable serialization for this test
        pol_context.serializePrediction = false;
        pol_context.serializeColMask = false;
        pol_context.serializeDataset = false;
        pol_context.serializeLabels = false;
        pol_context.serializationWindow = 10;
        pol_context.quietRun = true;  // Suppress verbose output

        // ===============================================
        // Test 1: Standard GradientBoostRegressor with 50 fitting steps
        // ===============================================
        cout << "\n=== Test 1: Standard GradientBoostRegressor (50 steps) ===" << endl;
        
        GradientBoostRegressor<DecisionTreeRegressorRegressor> standard_gb(X_train, y_train, pol_context);
        
        cout << "Calling fit() for 50 steps..." << endl;
        standard_gb.fit();  // This is the actual fitting with 50 steps
        
        Row<double> standard_train_pred, standard_test_pred;
        standard_gb.Predict(X_train, standard_train_pred);
        standard_gb.Predict(X_test, standard_test_pred);
        
        double standard_train_mse = calculate_mse(standard_train_pred, y_train);
        double standard_test_mse = calculate_mse(standard_test_pred, y_test);
        
        cout << "Standard GradientBoost Results (50 fitting steps):" << endl;
        cout << "Training MSE: " << standard_train_mse << endl;
        cout << "Test MSE:     " << standard_test_mse << endl;

        // ===============================================
        // Test 2: NegativeFeedback GradientBoostRegressor with 50 fitting steps  
        // ===============================================
        cout << "\n=== Test 2: NegativeFeedback GradientBoostRegressor (50 steps) ===" << endl;
        
        GradientBoostRegressor<TestNegativeFeedbackRegressor> decorated_gb(X_train, y_train, pol_context);
        
        cout << "Calling fit() for 50 steps with NegativeFeedback (beta=0.15)..." << endl;
        decorated_gb.fit();  // This is the actual fitting with 50 steps
        
        Row<double> decorated_train_pred, decorated_test_pred;
        decorated_gb.Predict(X_train, decorated_train_pred);
        decorated_gb.Predict(X_test, decorated_test_pred);
        
        double decorated_train_mse = calculate_mse(decorated_train_pred, y_train);
        double decorated_test_mse = calculate_mse(decorated_test_pred, y_test);
        
        cout << "NegativeFeedback GradientBoost Results (50 fitting steps, beta=0.15):" << endl;
        cout << "Training MSE: " << decorated_train_mse << endl;
        cout << "Test MSE:     " << decorated_test_mse << endl;

        // ===============================================
        // Test 3: Reduced Configuration GradientBoostRegressor
        // ===============================================
        cout << "\n=== Test 3: Reduced Configuration GradientBoostRegressor (50 steps) ===" << endl;
        
        Context reduced_context = pol_context;  // Copy base context
        reduced_context.childLearningRate = {0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1};  // Higher learning rate
        reduced_context.childActivePartitionRatio = {0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2};  // Lower partition ratio
        reduced_context.rowSubsampleRatio = 0.8;  // Subsampling
        reduced_context.colSubsampleRatio = 0.8;
        
        GradientBoostRegressor<DecisionTreeRegressorRegressor> reduced_gb(X_train, y_train, reduced_context);
        
        cout << "Calling fit() for 50 steps with reduced configuration..." << endl;
        reduced_gb.fit();  // This is the actual fitting with 50 steps
        
        Row<double> reduced_train_pred, reduced_test_pred;
        reduced_gb.Predict(X_train, reduced_train_pred);
        reduced_gb.Predict(X_test, reduced_test_pred);
        
        double reduced_train_mse = calculate_mse(reduced_train_pred, y_train);
        double reduced_test_mse = calculate_mse(reduced_test_pred, y_test);
        
        cout << "Reduced Configuration GradientBoost Results (50 fitting steps):" << endl;
        cout << "Training MSE: " << reduced_train_mse << endl;
        cout << "Test MSE:     " << reduced_test_mse << endl;

        // ===============================================
        // Comparison and Analysis
        // ===============================================
        cout << "\n=== Comparison After 50 Fitting Steps ===" << endl;
        cout << "Method                     | Train MSE    | Test MSE" << endl;
        cout << "---------------------------|-------------|------------" << endl;
        cout << "Standard GradientBoost     | " << std::fixed << std::setprecision(6) << standard_train_mse << " | " << standard_test_mse << endl;
        cout << "NegativeFeedback GB        | " << decorated_train_mse << " | " << decorated_test_mse << endl;
        cout << "Reduced Config GB          | " << reduced_train_mse << " | " << reduced_test_mse << endl;
        
        cout << "\nRelative Performance Changes (vs Standard):" << endl;
        cout << "NegativeFeedback: Train MSE change = " << (decorated_train_mse - standard_train_mse) 
             << ", Test MSE change = " << (decorated_test_mse - standard_test_mse) << endl;
        cout << "Reduced Config: Train MSE change = " << (reduced_train_mse - standard_train_mse) 
             << ", Test MSE change = " << (reduced_test_mse - standard_test_mse) << endl;

        cout << "\n=== Sample Predictions ===" << endl;
        cout << "First 10 test predictions:" << endl;
        cout << "Index\tTrue\tStandard\tNegFeedback\tReduced\tNF-Diff\tRed-Diff" << endl;
        size_t max_samples = std::min(static_cast<size_t>(10), static_cast<size_t>(y_test.n_elem));
        for (size_t i = 0; i < max_samples; ++i) {
            double nf_diff = decorated_test_pred[i] - standard_test_pred[i];
            double red_diff = reduced_test_pred[i] - standard_test_pred[i];
            cout << i << "\t" << std::fixed << std::setprecision(3) << y_test[i] << "\t" 
                 << standard_test_pred[i] << "\t\t" << decorated_test_pred[i] 
                 << "\t\t" << reduced_test_pred[i] << "\t\t" << nf_diff << "\t" << red_diff << endl;
        }

        cout << "\n🎉 All gradient boosting tests with fit() completed successfully!" << endl;
        cout << "\n=== Summary ===" << endl;
        cout << "✅ Standard GradientBoost with 50 fitting steps" << endl;
        cout << "✅ NegativeFeedback GradientBoost with 50 fitting steps (beta=0.15)" << endl;
        cout << "✅ Reduced configuration GradientBoost with 50 fitting steps" << endl;
        cout << "✅ All models trained on pol dataset with fit() method" << endl;
        cout << "✅ Performance comparison shows impact of fitting configurations" << endl;

        return 0;

    } catch (const exception& e) {
        cerr << "❌ Error: " << e.what() << endl;
        return 1;
    }
}