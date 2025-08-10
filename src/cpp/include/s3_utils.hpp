#ifndef S3_UTILS_HPP
#define S3_UTILS_HPP

#include <curl/curl.h>
#include <openssl/evp.h>
#include <openssl/hmac.h>
#include <openssl/sha.h>

#include <ctime>
#include <filesystem>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace IB_utils {

struct S3Config {
  std::string bucket;
  std::string access_key;
  std::string secret_key;
  std::string region = "us-east-1";
  std::string prefix = "";
};

/**
 * S3 download utility class using libcurl and AWS Signature Version 4
 */
class S3Downloader {
private:
  S3Config config_;
  CURL* curl_;

  // Helper struct for HTTP response
  struct HTTPResponse {
    std::string data;
    long response_code;
  };

  // Callback function for libcurl to write data
  static size_t WriteCallback(void* contents, size_t size, size_t nmemb, HTTPResponse* response);

  // AWS Signature Version 4 helpers
  std::string sha256_hex(const std::string& data);
  std::string hmac_sha256(const std::string& key, const std::string& data);
  std::string get_canonical_request(
      const std::string& method,
      const std::string& uri,
      const std::string& query_string,
      const std::string& headers,
      const std::string& signed_headers,
      const std::string& payload_hash);
  std::string get_string_to_sign(
      const std::string& algorithm,
      const std::string& request_date,
      const std::string& credential_scope,
      const std::string& canonical_request);
  std::string get_signature_key(
      const std::string& key,
      const std::string& date_stamp,
      const std::string& region_name,
      const std::string& service_name);
  std::string get_authorization_header(
      const std::string& access_key,
      const std::string& credential_scope,
      const std::string& signed_headers,
      const std::string& signature);

  // Get current timestamp in ISO8601 format
  std::string get_iso8601_timestamp();
  std::string get_date_stamp();

public:
  explicit S3Downloader(const S3Config& config);
  ~S3Downloader();

  // Download a file from S3 to local path
  bool download_file(const std::string& s3_key, const std::string& local_path);

  // Download dataset files (X and y) for a given dataset name
  bool download_dataset(const std::string& dataset_name, const std::string& output_dir);

  // Check if a file exists in S3
  bool file_exists(const std::string& s3_key);
};

/**
 * Parse S3 configuration from JSON file
 */
S3Config parse_s3_config(const std::string& json_file_path);

/**
 * Download dataset from S3 using configuration
 */
bool download_s3_dataset(
    const S3Config& config, const std::string& dataset_name, const std::string& output_dir);

}  // namespace IB_utils

#endif  // S3_UTILS_HPP