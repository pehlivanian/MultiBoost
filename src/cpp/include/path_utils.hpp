#ifndef PATH_UTILS_HPP
#define PATH_UTILS_HPP

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>

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
 * Resolves a path to a data file.
 * Data files are expected to be in the data directory defined by the environment
 * or a default location.
 *
 * @param filename The name of the data file
 * @return The absolute path to the data file
 */
inline std::string resolve_data_path(std::string_view filename) {
  // Try to get the data directory from environment variable
  if (const char* data_dir_env = std::getenv("IB_DATA_DIR")) {
    const std::filesystem::path data_path(data_dir_env);
    const std::filesystem::path file_path(filename);
    return (data_path / file_path).string();
  }

  // Default to ~/Data if environment variable not set
  if (const char* home_dir = std::getenv("HOME")) {
    const std::filesystem::path home_path(home_dir);
    const std::filesystem::path data_path = home_path / "Data";
    const std::filesystem::path file_path(filename);
    return (data_path / file_path).string();
  }

  // If all else fails, just return the filename
  return std::string(filename);
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