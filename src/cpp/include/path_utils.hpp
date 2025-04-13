#ifndef PATH_UTILS_HPP
#define PATH_UTILS_HPP

#include <filesystem>
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
  
  std::filesystem::path base_path(project_root);
  std::filesystem::path rel_path(relative_path);
  
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
  const char* data_dir_env = std::getenv("IB_DATA_DIR");
  
  if (data_dir_env != nullptr) {
    std::filesystem::path data_path(data_dir_env);
    std::filesystem::path file_path(filename);
    return (data_path / file_path).string();
  }
  
  // Default to ~/Data if environment variable not set
  const char* home_dir = std::getenv("HOME");
  if (home_dir != nullptr) {
    std::filesystem::path home_path(home_dir);
    std::filesystem::path data_path = home_path / "Data";
    std::filesystem::path file_path(filename);
    return (data_path / file_path).string();
  }
  
  // If all else fails, just return the filename
  return std::string(filename);
}

} // namespace IB_utils

#endif // PATH_UTILS_HPP