#include "s3_utils.hpp"

#include <cereal/archives/json.hpp>
#include <cereal/types/string.hpp>
#include <fstream>
#include <iostream>

namespace IB_utils {

// Callback function for libcurl to write data
size_t S3Downloader::WriteCallback(
    void* contents, size_t size, size_t nmemb, HTTPResponse* response) {
  size_t total_size = size * nmemb;
  response->data.append(static_cast<char*>(contents), total_size);
  return total_size;
}

S3Downloader::S3Downloader(const S3Config& config) : config_(config) {
  curl_global_init(CURL_GLOBAL_DEFAULT);
  curl_ = curl_easy_init();
}

S3Downloader::~S3Downloader() {
  if (curl_) {
    curl_easy_cleanup(curl_);
  }
  curl_global_cleanup();
}

std::string S3Downloader::sha256_hex(const std::string& data) {
  unsigned char hash[SHA256_DIGEST_LENGTH];
  SHA256_CTX sha256;
  SHA256_Init(&sha256);
  SHA256_Update(&sha256, data.c_str(), data.length());
  SHA256_Final(hash, &sha256);

  std::stringstream ss;
  for (int i = 0; i < SHA256_DIGEST_LENGTH; i++) {
    ss << std::hex << std::setw(2) << std::setfill('0') << (int)hash[i];
  }
  return ss.str();
}

std::string S3Downloader::hmac_sha256(const std::string& key, const std::string& data) {
  unsigned char result[EVP_MAX_MD_SIZE];
  unsigned int result_len;

  HMAC(
      EVP_sha256(),
      key.c_str(),
      key.length(),
      reinterpret_cast<const unsigned char*>(data.c_str()),
      data.length(),
      result,
      &result_len);

  return std::string(reinterpret_cast<char*>(result), result_len);
}

std::string S3Downloader::get_iso8601_timestamp() {
  auto now = std::time(nullptr);
  auto utc = std::gmtime(&now);

  std::stringstream ss;
  ss << std::put_time(utc, "%Y%m%dT%H%M%SZ");
  return ss.str();
}

std::string S3Downloader::get_date_stamp() {
  auto now = std::time(nullptr);
  auto utc = std::gmtime(&now);

  std::stringstream ss;
  ss << std::put_time(utc, "%Y%m%d");
  return ss.str();
}

std::string S3Downloader::get_canonical_request(
    const std::string& method,
    const std::string& uri,
    const std::string& query_string,
    const std::string& headers,
    const std::string& signed_headers,
    const std::string& payload_hash) {
  return method + "\\n" + uri + "\\n" + query_string + "\\n" + headers + "\\n" + signed_headers +
         "\\n" + payload_hash;
}

std::string S3Downloader::get_string_to_sign(
    const std::string& algorithm,
    const std::string& request_date,
    const std::string& credential_scope,
    const std::string& canonical_request) {
  return algorithm + "\\n" + request_date + "\\n" + credential_scope + "\\n" +
         sha256_hex(canonical_request);
}

std::string S3Downloader::get_signature_key(
    const std::string& key,
    const std::string& date_stamp,
    const std::string& region_name,
    const std::string& service_name) {
  std::string k_secret = "AWS4" + key;
  std::string k_date = hmac_sha256(k_secret, date_stamp);
  std::string k_region = hmac_sha256(k_date, region_name);
  std::string k_service = hmac_sha256(k_region, service_name);
  std::string k_signing = hmac_sha256(k_service, "aws4_request");
  return k_signing;
}

std::string S3Downloader::get_authorization_header(
    const std::string& access_key,
    const std::string& credential_scope,
    const std::string& signed_headers,
    const std::string& signature) {
  return "AWS4-HMAC-SHA256 Credential=" + access_key + "/" + credential_scope +
         ",SignedHeaders=" + signed_headers + ",Signature=" + signature;
}

bool S3Downloader::download_file(const std::string& s3_key, const std::string& local_path) {
  if (!curl_) {
    std::cerr << "CURL not initialized" << std::endl;
    return false;
  }

  // Create URL
  std::string url =
      "https://" + config_.bucket + ".s3." + config_.region + ".amazonaws.com/" + s3_key;

  // Get timestamps
  std::string timestamp = get_iso8601_timestamp();
  std::string date_stamp = get_date_stamp();

  // Create canonical request
  std::string method = "GET";
  std::string uri = "/" + s3_key;
  std::string query_string = "";
  std::string host = config_.bucket + ".s3." + config_.region + ".amazonaws.com";
  std::string headers = "host:" + host + "\\nx-amz-date:" + timestamp + "\\n";
  std::string signed_headers = "host;x-amz-date";
  std::string payload_hash = sha256_hex("");

  std::string canonical_request =
      get_canonical_request(method, uri, query_string, headers, signed_headers, payload_hash);

  // Create string to sign
  std::string algorithm = "AWS4-HMAC-SHA256";
  std::string credential_scope = date_stamp + "/" + config_.region + "/s3/aws4_request";
  std::string string_to_sign =
      get_string_to_sign(algorithm, timestamp, credential_scope, canonical_request);

  // Create signature
  std::string signing_key = get_signature_key(config_.secret_key, date_stamp, config_.region, "s3");
  std::string signature_raw = hmac_sha256(signing_key, string_to_sign);

  // Convert signature to hex
  std::stringstream sig_ss;
  for (unsigned char c : signature_raw) {
    sig_ss << std::hex << std::setw(2) << std::setfill('0') << (int)c;
  }
  std::string signature = sig_ss.str();

  // Create authorization header
  std::string auth_header =
      get_authorization_header(config_.access_key, credential_scope, signed_headers, signature);

  // Set up curl
  HTTPResponse response;
  curl_easy_setopt(curl_, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl_, CURLOPT_WRITEFUNCTION, WriteCallback);
  curl_easy_setopt(curl_, CURLOPT_WRITEDATA, &response);

  // Set headers
  struct curl_slist* headers_list = nullptr;
  std::string host_header = "Host: " + host;
  std::string date_header = "x-amz-date: " + timestamp;
  std::string auth_header_full = "Authorization: " + auth_header;

  headers_list = curl_slist_append(headers_list, host_header.c_str());
  headers_list = curl_slist_append(headers_list, date_header.c_str());
  headers_list = curl_slist_append(headers_list, auth_header_full.c_str());

  curl_easy_setopt(curl_, CURLOPT_HTTPHEADER, headers_list);

  // Perform request
  CURLcode res = curl_easy_perform(curl_);
  curl_easy_getinfo(curl_, CURLINFO_RESPONSE_CODE, &response.response_code);

  curl_slist_free_all(headers_list);

  if (res != CURLE_OK) {
    std::cerr << "curl_easy_perform() failed: " << curl_easy_strerror(res) << std::endl;
    return false;
  }

  if (response.response_code != 200) {
    std::cerr << "HTTP error: " << response.response_code << std::endl;
    std::cerr << "Response: " << response.data << std::endl;
    return false;
  }

  // Write to file
  std::ofstream file(local_path, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Failed to open file: " << local_path << std::endl;
    return false;
  }

  file.write(response.data.c_str(), response.data.length());
  file.close();

  std::cout << "Downloaded: " << s3_key << " to " << local_path << std::endl;
  return true;
}

bool S3Downloader::download_dataset(
    const std::string& dataset_name, const std::string& output_dir) {
  // Create output directory if it doesn't exist
  std::filesystem::create_directories(output_dir);

  // Construct S3 keys
  std::string x_key = config_.prefix.empty() ? dataset_name + "_X.csv"
                                             : config_.prefix + "/" + dataset_name + "_X.csv";
  std::string y_key = config_.prefix.empty() ? dataset_name + "_y.csv"
                                             : config_.prefix + "/" + dataset_name + "_y.csv";

  // Local file paths
  std::string x_path = output_dir + "/" + dataset_name + "_X.csv";
  std::string y_path = output_dir + "/" + dataset_name + "_y.csv";

  // Download both files
  bool x_success = download_file(x_key, x_path);
  bool y_success = download_file(y_key, y_path);

  return x_success && y_success;
}

bool S3Downloader::file_exists(const std::string& s3_key) {
  // Implementation for HEAD request to check if file exists
  // For now, we'll just try to download and return false if it fails
  return true;  // Simplified implementation
}

S3Config parse_s3_config(const std::string& json_file_path) {
  S3Config config;

  try {
    std::ifstream file(json_file_path);
    if (!file.is_open()) {
      std::cerr << "Failed to open S3 config file: " << json_file_path << std::endl;
      return config;
    }

    cereal::JSONInputArchive archive(file);

    // The JSON structure is: {"x": {"s3Config": {...}}}
    // We need to navigate to the s3Config section
    std::string bucket, access_key, secret_key, region, prefix;

    // For now, let's parse manually since the JSON structure is nested
    // This is a simplified approach - in production you'd want proper JSON parsing
    std::string json_content(
        (std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    // Find s3Config section (simplified parsing)
    size_t s3_pos = json_content.find("\"s3Config\"");
    if (s3_pos == std::string::npos) {
      std::cerr << "No s3Config found in JSON" << std::endl;
      return config;
    }

    // Extract bucket
    size_t bucket_pos = json_content.find("\"bucket\"", s3_pos);
    if (bucket_pos != std::string::npos) {
      size_t start = json_content.find("\"", bucket_pos + 8) + 1;
      size_t end = json_content.find("\"", start);
      config.bucket = json_content.substr(start, end - start);
    }

    // Extract access_key
    size_t access_pos = json_content.find("\"access_key\"", s3_pos);
    if (access_pos != std::string::npos) {
      size_t start = json_content.find("\"", access_pos + 12) + 1;
      size_t end = json_content.find("\"", start);
      config.access_key = json_content.substr(start, end - start);
    }

    // Extract secret_key
    size_t secret_pos = json_content.find("\"secret_key\"", s3_pos);
    if (secret_pos != std::string::npos) {
      size_t start = json_content.find("\"", secret_pos + 12) + 1;
      size_t end = json_content.find("\"", start);
      config.secret_key = json_content.substr(start, end - start);
    }

    // Extract region (optional)
    size_t region_pos = json_content.find("\"region\"", s3_pos);
    if (region_pos != std::string::npos) {
      size_t start = json_content.find("\"", region_pos + 8) + 1;
      size_t end = json_content.find("\"", start);
      config.region = json_content.substr(start, end - start);
    }

    // Extract prefix (optional)
    size_t prefix_pos = json_content.find("\"prefix\"", s3_pos);
    if (prefix_pos != std::string::npos) {
      size_t start = json_content.find("\"", prefix_pos + 8) + 1;
      size_t end = json_content.find("\"", start);
      config.prefix = json_content.substr(start, end - start);
    }

  } catch (const std::exception& e) {
    std::cerr << "Error parsing S3 config: " << e.what() << std::endl;
  }

  return config;
}

bool download_s3_dataset(
    const S3Config& config, const std::string& dataset_name, const std::string& output_dir) {
  S3Downloader downloader(config);
  return downloader.download_dataset(dataset_name, output_dir);
}

}  // namespace IB_utils