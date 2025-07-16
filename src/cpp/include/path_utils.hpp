#ifndef PATH_UTILS_HPP
#define PATH_UTILS_HPP

#include <filesystem>
#include <string>
#include <string_view>
#include <cstdlib>
#include <iostream>

namespace IB_utils {

/**
 * Resolves a path relative to the project root.
 * Uses the IB_PROJECT_ROOT preprocessor define set by CMake.
 *
 * @param relative_path The path relative to project root
 * @return The absolute path
 */
inline std::string resolve_path(std::string_view relative_path) {
  static const std::string project_root = IB_PROJECT_ROOT;

  // If the path is already absolute, return it as is
  if (!relative_path.empty() && relative_path[0] == '/') {
    return std::string(relative_path);
  }

  const std::filesystem::path base_path(project_root);
  const std::filesystem::path rel_path(relative_path);

  return (base_path / rel_path).string();
}

/**
 * Downloads dataset from S3 if S3 configuration is available and local file doesn't exist.
 * 
 * @param dataset_base_name Base name of dataset (e.g., "synthetic_train")
 * @param temp_dir Directory to download files to
 * @return true if download successful or not needed, false on error
 */
inline bool try_s3_download(const std::string& dataset_base_name, const std::string& temp_dir) {
  // Check if S3 config file exists (passed from Python API)
  const char* s3_config_file = std::getenv("IB_S3_CONFIG_FILE");
  if (!s3_config_file || !std::filesystem::exists(s3_config_file)) {
    return false; // No S3 config available
  }

  // Get project root for helper script
  const char* project_root = std::getenv("IB_PROJECT_ROOT");
  if (!project_root) {
    project_root = IB_PROJECT_ROOT;
  }

  // Build command to call S3 download helper
  std::string helper_script = std::string(project_root) + "/scripts/s3_download_helper.py";
  std::string command = "python3 " + helper_script + " \"" + s3_config_file + "\" \"" + 
                       dataset_base_name + "\" \"" + temp_dir + "\"";

  // Execute the helper script
  int result = std::system(command.c_str());
  return result == 0;
}

/**
 * Resolves a path to a data file.
 * If S3 config is available, downloads from S3 and returns that path.
 * Otherwise, falls back to local data directory.
 *
 * @param filename The name of the data file
 * @return The absolute path to the data file
 */
inline std::string resolve_data_path(std::string_view filename) {
  std::string filename_str(filename);
  
  // Check if S3 config is available - if so, prioritize S3 download
  const char* s3_config_file = std::getenv("IB_S3_CONFIG_FILE");
  if (s3_config_file && std::filesystem::exists(s3_config_file)) {
    // Extract dataset base name from filename (remove _X.csv or _y.csv suffix)
    std::string dataset_base_name = filename_str;
    if (dataset_base_name.ends_with("_X.csv")) {
      dataset_base_name = dataset_base_name.substr(0, dataset_base_name.length() - 6);
    } else if (dataset_base_name.ends_with("_y.csv")) {
      dataset_base_name = dataset_base_name.substr(0, dataset_base_name.length() - 6);
    }

    // Check if this is the first file of a dataset pair (X file)
    if (filename_str.ends_with("_X.csv")) {
      // Create temporary directory for S3 download
      std::string temp_dir = "/tmp/ib_s3_data";
      std::filesystem::create_directories(temp_dir);
      
      // Try to download dataset from S3
      if (try_s3_download(dataset_base_name, temp_dir)) {
        // Successfully downloaded - return path to file in temp directory
        return temp_dir + "/" + filename_str;
      }
    } else if (filename_str.ends_with("_y.csv")) {
      // For Y files, check if corresponding X file was already downloaded to temp
      std::string temp_path = "/tmp/ib_s3_data/" + filename_str;
      if (std::filesystem::exists(temp_path)) {
        return temp_path;
      }
    }
  }

  // Fall back to local data directory
  std::string data_dir;
  if (const char* data_dir_env = std::getenv("IB_DATA_DIR")) {
    data_dir = data_dir_env;
  } else if (const char* home_dir = std::getenv("HOME")) {
    data_dir = std::string(home_dir) + "/Data";
  } else {
    // If all else fails, use current directory
    data_dir = ".";
  }

  // Construct and return full path
  const std::filesystem::path data_path(data_dir);
  const std::filesystem::path file_path(filename);
  return (data_path / file_path).string();
}

/**
 * Resolves a path to a test data file.
 * Test data files are expected to be in the test_data directory defined by the build
 * or a default location.
 *
 * @param filename The name of the test data file
 * @return The absolute path to the test data file
 */
inline std::string resolve_test_data_path(std::string_view filename) {
  // First check if we have test data in the user's data directory
  std::string data_path = resolve_data_path(filename);
  if (std::filesystem::exists(data_path)) {
    return data_path;
  }

  // If not found in data dir, use the test data directory from build
  static const std::string test_data_dir =
#ifdef IB_TEST_DATA_DIR
      IB_TEST_DATA_DIR;
#else
      (std::filesystem::path(IB_PROJECT_ROOT) / "test_data").string();
#endif

  std::filesystem::path test_path(test_data_dir);
  std::filesystem::path file_path(filename);
  return (test_path / file_path).string();
}

}  // namespace IB_utils

#endif  // PATH_UTILS_HPP