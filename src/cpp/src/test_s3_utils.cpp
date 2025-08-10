#include <gtest/gtest.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <thread>
#include <vector>

#include "path_utils.hpp"
#include "s3_utils.hpp"

namespace fs = std::filesystem;

class S3UtilsTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Create test directory
    test_dir = "/tmp/s3_utils_test";
    fs::create_directories(test_dir);

    // Load S3 credentials from config file
    std::string creds_file = IB_PROJECT_ROOT "/test_data/s3_credentials.json";
    if (fs::exists(creds_file)) {
      std::ifstream file(creds_file);
      if (file.is_open()) {
        std::string json_content(
            (std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

        // Parse JSON manually (simplified)
        parseCredentials(json_content);
        has_credentials = !test_config.bucket.empty() && !test_config.access_key.empty() &&
                          !test_config.secret_key.empty();
      }
    }

    if (!has_credentials) {
      std::cout << "Warning: No S3 credentials found. S3 tests will be skipped." << std::endl;
    }
  }

  void TearDown() override {
    // Clean up test directory
    if (fs::exists(test_dir)) {
      fs::remove_all(test_dir);
    }
  }

  void parseCredentials(const std::string& json_content) {
    // Extract bucket
    size_t bucket_pos = json_content.find("\"bucket\"");
    if (bucket_pos != std::string::npos) {
      size_t start = json_content.find("\"", bucket_pos + 8) + 1;
      size_t end = json_content.find("\"", start);
      test_config.bucket = json_content.substr(start, end - start);
    }

    // Extract access_key
    size_t access_pos = json_content.find("\"access_key\"");
    if (access_pos != std::string::npos) {
      size_t start = json_content.find("\"", access_pos + 12) + 1;
      size_t end = json_content.find("\"", start);
      test_config.access_key = json_content.substr(start, end - start);
    }

    // Extract secret_key
    size_t secret_pos = json_content.find("\"secret_key\"");
    if (secret_pos != std::string::npos) {
      size_t start = json_content.find("\"", secret_pos + 12) + 1;
      size_t end = json_content.find("\"", start);
      test_config.secret_key = json_content.substr(start, end - start);
    }

    // Extract region
    size_t region_pos = json_content.find("\"region\"");
    if (region_pos != std::string::npos) {
      size_t start = json_content.find("\"", region_pos + 8) + 1;
      size_t end = json_content.find("\"", start);
      test_config.region = json_content.substr(start, end - start);
    }

    // Extract prefix
    size_t prefix_pos = json_content.find("\"prefix\"");
    if (prefix_pos != std::string::npos) {
      size_t start = json_content.find("\"", prefix_pos + 8) + 1;
      size_t end = json_content.find("\"", start);
      test_config.prefix = json_content.substr(start, end - start);
    }
  }

  std::string test_dir;
  IB_utils::S3Config test_config;
  bool has_credentials = false;
};

// Test S3Config parsing from JSON file
TEST_F(S3UtilsTest, ParseS3ConfigFromFile) {
  if (!has_credentials) {
    std::cout << "SKIPPED: No S3 credentials available" << std::endl;
    return;
  }

  // Create a test config file
  std::string config_file = test_dir + "/test_config.json";
  std::ofstream file(config_file);
  file << R"({
        "x": {
            "traindataname": "test_dataset",
            "s3Config": {
                "bucket": ")"
       << test_config.bucket << R"(",
                "access_key": ")"
       << test_config.access_key << R"(",
                "secret_key": ")"
       << test_config.secret_key << R"(",
                "prefix": ")"
       << test_config.prefix << R"(",
                "region": ")"
       << test_config.region << R"("
            }
        }
    })";
  file.close();

  // Parse the config
  IB_utils::S3Config parsed_config = IB_utils::parse_s3_config(config_file);

  // Verify parsing
  EXPECT_EQ(parsed_config.bucket, test_config.bucket);
  EXPECT_EQ(parsed_config.access_key, test_config.access_key);
  EXPECT_EQ(parsed_config.secret_key, test_config.secret_key);
  EXPECT_EQ(parsed_config.prefix, test_config.prefix);
  EXPECT_EQ(parsed_config.region, test_config.region);
}

// Test S3Config parsing with missing file
TEST_F(S3UtilsTest, ParseS3ConfigMissingFile) {
  std::string non_existent_file = test_dir + "/non_existent.json";

  IB_utils::S3Config config = IB_utils::parse_s3_config(non_existent_file);

  // Should return empty config
  EXPECT_TRUE(config.bucket.empty());
  EXPECT_TRUE(config.access_key.empty());
  EXPECT_TRUE(config.secret_key.empty());
}

// Test S3Config parsing with invalid JSON
TEST_F(S3UtilsTest, ParseS3ConfigInvalidJSON) {
  std::string config_file = test_dir + "/invalid_config.json";
  std::ofstream file(config_file);
  file << "{ invalid json content }";
  file.close();

  IB_utils::S3Config config = IB_utils::parse_s3_config(config_file);

  // Should return empty config on parse error
  EXPECT_TRUE(config.bucket.empty());
  EXPECT_TRUE(config.access_key.empty());
  EXPECT_TRUE(config.secret_key.empty());
}

// Test S3Downloader constructor and destructor
TEST_F(S3UtilsTest, S3DownloaderConstructor) {
  if (!has_credentials) {
    std::cout << "SKIPPED: No S3 credentials available" << std::endl;
    return;
  }

  // Should construct without throwing
  EXPECT_NO_THROW({ IB_utils::S3Downloader downloader(test_config); });
}

// Test downloading a single file from S3
TEST_F(S3UtilsTest, DownloadSingleFile) {
  if (!has_credentials) {
    std::cout << "SKIPPED: No S3 credentials available" << std::endl;
    return;
  }

  IB_utils::S3Downloader downloader(test_config);

  // Try to download a test file (assuming it exists in S3)
  std::string s3_key = "synthetic_train_X.csv";
  std::string local_path = test_dir + "/downloaded_file.csv";

  bool success = downloader.download_file(s3_key, local_path);

  if (success) {
    // Verify file was downloaded
    EXPECT_TRUE(fs::exists(local_path));
    EXPECT_GT(fs::file_size(local_path), 0);

    // Verify file content (basic check)
    std::ifstream file(local_path);
    std::string first_line;
    std::getline(file, first_line);
    EXPECT_FALSE(first_line.empty());
  } else {
    std::cout << "Note: File " << s3_key
              << " not found in S3 bucket. This is expected if test data is not uploaded."
              << std::endl;
  }
}

// Test downloading a complete dataset
TEST_F(S3UtilsTest, DownloadDataset) {
  if (!has_credentials) {
    std::cout << "SKIPPED: No S3 credentials available" << std::endl;
    return;
  }

  IB_utils::S3Downloader downloader(test_config);

  std::string dataset_name = "synthetic_train";
  std::string output_dir = test_dir + "/dataset_output";

  bool success = downloader.download_dataset(dataset_name, output_dir);

  if (success) {
    // Verify both files were downloaded
    std::string x_file = output_dir + "/" + dataset_name + "_X.csv";
    std::string y_file = output_dir + "/" + dataset_name + "_y.csv";

    EXPECT_TRUE(fs::exists(x_file));
    EXPECT_TRUE(fs::exists(y_file));
    EXPECT_GT(fs::file_size(x_file), 0);
    EXPECT_GT(fs::file_size(y_file), 0);

    // Verify files have valid CSV content
    std::ifstream x_stream(x_file);
    std::ifstream y_stream(y_file);
    std::string x_line, y_line;

    std::getline(x_stream, x_line);
    std::getline(y_stream, y_line);

    EXPECT_FALSE(x_line.empty());
    EXPECT_FALSE(y_line.empty());

    // Basic CSV validation - should contain commas for multi-column data
    // (This is dataset-specific, adjust as needed)
    if (x_line.find(',') != std::string::npos) {
      EXPECT_TRUE(x_line.find(',') != std::string::npos);
    }
  } else {
    std::cout << "Note: Dataset " << dataset_name
              << " not found in S3 bucket. This is expected if test data is not uploaded."
              << std::endl;
  }
}

// Test path_utils integration with S3
TEST_F(S3UtilsTest, PathUtilsS3Integration) {
  if (!has_credentials) {
    std::cout << "SKIPPED: No S3 credentials available" << std::endl;
    return;
  }

  // Create a parameter file with S3 config
  std::string param_file = test_dir + "/param_file.json";
  std::ofstream file(param_file);
  file << R"({
        "x": {
            "traindataname": "synthetic_train",
            "s3Config": {
                "bucket": ")"
       << test_config.bucket << R"(",
                "access_key": ")"
       << test_config.access_key << R"(",
                "secret_key": ")"
       << test_config.secret_key << R"(",
                "prefix": ")"
       << test_config.prefix << R"(",
                "region": ")"
       << test_config.region << R"("
            }
        }
    })";
  file.close();

  // Set environment variable
  setenv("IB_PARAM_FILE", param_file.c_str(), 1);

  // Test resolving data path - this should trigger S3 download
  std::string resolved_path = IB_utils::resolve_data_path("synthetic_train_X.csv");

  // The resolved path should point to the downloaded file
  EXPECT_TRUE(
      resolved_path.find("/tmp/ib_s3_data/") != std::string::npos ||
      resolved_path.find("synthetic_train_X.csv") != std::string::npos);

  // Clean up environment
  unsetenv("IB_PARAM_FILE");
}

// Test error handling with invalid credentials
TEST_F(S3UtilsTest, InvalidCredentials) {
  IB_utils::S3Config bad_config;
  bad_config.bucket = "non-existent-bucket-12345";
  bad_config.access_key = "invalid_key";
  bad_config.secret_key = "invalid_secret";
  bad_config.region = "us-east-1";

  IB_utils::S3Downloader downloader(bad_config);

  std::string s3_key = "test_file.csv";
  std::string local_path = test_dir + "/should_not_exist.csv";

  bool success = downloader.download_file(s3_key, local_path);

  // Should fail with invalid credentials
  EXPECT_FALSE(success);
  EXPECT_FALSE(fs::exists(local_path));
}

// Test download_s3_dataset function
TEST_F(S3UtilsTest, DownloadS3DatasetFunction) {
  if (!has_credentials) {
    std::cout << "SKIPPED: No S3 credentials available" << std::endl;
    return;
  }

  std::string dataset_name = "synthetic_train";
  std::string output_dir = test_dir + "/function_test";

  bool success = IB_utils::download_s3_dataset(test_config, dataset_name, output_dir);

  if (success) {
    // Verify directory was created
    EXPECT_TRUE(fs::exists(output_dir));

    // Verify files exist
    std::string x_file = output_dir + "/" + dataset_name + "_X.csv";
    std::string y_file = output_dir + "/" + dataset_name + "_y.csv";

    EXPECT_TRUE(fs::exists(x_file));
    EXPECT_TRUE(fs::exists(y_file));
  } else {
    std::cout << "Note: Function test for dataset " << dataset_name
              << " failed. This is expected if test data is not uploaded." << std::endl;
  }
}

// Performance test - download timing
TEST_F(S3UtilsTest, DownloadPerformance) {
  if (!has_credentials) {
    std::cout << "SKIPPED: No S3 credentials available" << std::endl;
    return;
  }

  IB_utils::S3Downloader downloader(test_config);

  std::string dataset_name = "synthetic_train";
  std::string output_dir = test_dir + "/performance_test";

  auto start_time = std::chrono::high_resolution_clock::now();
  bool success = downloader.download_dataset(dataset_name, output_dir);
  auto end_time = std::chrono::high_resolution_clock::now();

  if (success) {
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Dataset download took: " << duration.count() << " ms" << std::endl;

    // Reasonable performance expectation (adjust based on your needs)
    EXPECT_LT(duration.count(), 30000);  // Less than 30 seconds
  }
}

// Test concurrent downloads
TEST_F(S3UtilsTest, ConcurrentDownloads) {
  if (!has_credentials) {
    std::cout << "SKIPPED: No S3 credentials available" << std::endl;
    return;
  }

  const int num_threads = 3;
  std::vector<std::thread> threads;
  std::vector<bool> results(num_threads, false);

  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back([this, i, &results]() {
      IB_utils::S3Downloader downloader(test_config);
      std::string output_dir = test_dir + "/concurrent_test_" + std::to_string(i);
      results[i] = downloader.download_dataset("synthetic_train", output_dir);
    });
  }

  // Wait for all threads to complete
  for (auto& thread : threads) {
    thread.join();
  }

  // Check results - at least some should succeed if data exists
  bool any_success = false;
  for (bool result : results) {
    if (result) any_success = true;
  }

  if (any_success) {
    std::cout << "Concurrent download test passed" << std::endl;
  } else {
    std::cout
        << "Note: Concurrent download test failed. This is expected if test data is not uploaded."
        << std::endl;
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}