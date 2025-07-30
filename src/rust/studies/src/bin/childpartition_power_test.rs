use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::io::Write;

#[derive(Serialize, Deserialize, Debug, Clone)]
struct S3Config {
    bucket: String,
    access_key: String,
    secret_key: String,
    prefix: String,
    region: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct Loss {
    index: i32,
    data: i32,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct ModelParams {
    #[serde(rename = "traindataname")]
    train_data_name: String,
    #[serde(rename = "testdataname")]
    test_data_name: String,
    #[serde(rename = "s3Config")]
    s3_config: Option<S3Config>,
    steps: i32,
    #[serde(rename = "recursiveFit")]
    recursive_fit: bool,
    #[serde(rename = "useWeights")]
    use_weights: bool,
    #[serde(rename = "rowSubsampleRatio")]
    row_subsample_ratio: f64,
    #[serde(rename = "colSubsampleRatio")]
    col_subsample_ratio: f64,
    #[serde(rename = "removeRedundantLabels")]
    remove_redundant_labels: bool,
    #[serde(rename = "symmetrizeLabels")]
    symmetrize_labels: bool,
    loss: Loss,
    #[serde(rename = "lossPower")]
    loss_power: f64,
    clamp_gradient: bool,
    upper_val: f64,
    lower_val: f64,
    #[serde(rename = "numTrees")]
    num_trees: i32,
    depth: i32,
    #[serde(rename = "childPartitionSize")]
    child_partition_size: Vec<i32>,
    #[serde(rename = "childNumSteps")]
    child_num_steps: Vec<i32>,
    #[serde(rename = "childLearningRate")]
    child_learning_rate: Vec<f64>,
    #[serde(rename = "childActivePartitionRatio")]
    child_active_partition_ratio: Vec<f64>,
    #[serde(rename = "childMinLeafSize")]
    child_min_leaf_size: Vec<i32>,
    #[serde(rename = "childMinimumGainSplit")]
    child_minimum_gain_split: Vec<f64>,
    #[serde(rename = "childMaxDepth")]
    child_max_depth: Vec<i32>,
    #[serde(rename = "serializeModel")]
    serialize_model: bool,
    #[serde(rename = "serializePrediction")]
    serialize_prediction: bool,
    #[serde(rename = "serializeColMask")]
    serialize_col_mask: bool,
    #[serde(rename = "serializeDataset")]
    serialize_dataset: bool,
    #[serde(rename = "serializeLabels")]
    serialize_labels: bool,
    #[serde(rename = "serializationWindow")]
    serialization_window: i32,
    #[serde(rename = "showISEachStep")]
    show_is_each_step: bool,
    #[serde(rename = "showOOSEachStep")]
    show_oos_each_step: bool,
    #[serde(rename = "quietRun", default)]
    quiet_run: bool,
}

#[derive(Serialize, Deserialize, Debug)]
struct ConfigWrapper {
    x: ModelParams,
}


fn extract_oos_r_squared_from_api_response(response: &str) -> Option<f64> {
    // First try to parse as JSON to get structured data
    if let Ok(json_value) = serde_json::from_str::<serde_json::Value>(response) {
        // Look for metrics in the structured data
        if let Some(data) = json_value.get("data") {
            if let Some(iterations) = data.get("iterations").and_then(|i| i.as_array()) {
                // Look for the last iteration with metrics for house_sales_test
                for iteration in iterations.iter().rev() {
                    if let Some(dataset) = iteration.get("dataset").and_then(|d| d.as_str()) {
                        if dataset == "house_sales_test" {
                            if let Some(metrics) = iteration.get("metrics").and_then(|m| m.as_object()) {
                                if let Some(r_squared) = metrics.get("r_squared").and_then(|r| r.as_f64()) {
                                    return Some(r_squared);
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // If no structured metrics, look in stdout/stderr fields
        if let Some(stdout) = json_value.get("stdout").and_then(|s| s.as_str()) {
            if let Some(r_squared) = extract_oos_r_squared_from_text(stdout) {
                return Some(r_squared);
            }
        }
        
        if let Some(stderr) = json_value.get("stderr").and_then(|s| s.as_str()) {
            if let Some(r_squared) = extract_oos_r_squared_from_text(stderr) {
                return Some(r_squared);
            }
        }
    }
    
    // Fallback: treat entire response as text
    extract_oos_r_squared_from_text(response)
}

fn extract_oos_r_squared_from_text(text: &str) -> Option<f64> {
    let lines: Vec<&str> = text.lines().collect();
    
    // Look for patterns that might contain R² values
    for line in lines.iter().rev() {
        // Pattern 1: Look for "OOS:" followed by r_squared in parentheses
        if line.contains("OOS:") && line.contains("(r_squared)") {
            // Parse format like: "[house_sales_test] OOS: (r_squared): (0.789012)"
            if let Some(start) = line.rfind("(r_squared): (") {
                let r_squared_str = &line[start + 14..];
                if let Some(end) = r_squared_str.find(')') {
                    let r_squared_str = &r_squared_str[..end];
                    if let Ok(r_squared) = r_squared_str.trim().parse::<f64>() {
                        return Some(r_squared);
                    }
                }
            }
        }
        
        // Pattern 2: Look for "R² = " format
        if line.contains("R² = ") {
            if let Some(r_squared_start) = line.find("R² = ") {
                let r_squared_str = &line[r_squared_start + 5..];
                if let Some(end) = r_squared_str.find(',').or_else(|| r_squared_str.find(' ')) {
                    let r_squared_str = &r_squared_str[..end];
                    if let Ok(r_squared) = r_squared_str.trim().parse::<f64>() {
                        return Some(r_squared);
                    }
                } else {
                    // R² value might be at the end of the line
                    if let Ok(r_squared) = r_squared_str.trim().parse::<f64>() {
                        return Some(r_squared);
                    }
                }
            }
        }
        
        // Pattern 3: Look for "R^2:" format (as seen in the shell script)
        if line.contains("R^2:") {
            if let Some(r_squared_start) = line.find("R^2:") {
                let r_squared_str = &line[r_squared_start + 4..];
                // Remove any leading/trailing whitespace and extract the number
                let r_squared_str = r_squared_str.trim();
                if let Some(end) = r_squared_str.find(' ') {
                    let r_squared_str = &r_squared_str[..end];
                    if let Ok(r_squared) = r_squared_str.trim().parse::<f64>() {
                        return Some(r_squared);
                    }
                } else {
                    if let Ok(r_squared) = r_squared_str.parse::<f64>() {
                        return Some(r_squared);
                    }
                }
            }
        }
    }
    None
}

async fn run_experiment_with_api(json_str: &str) -> Result<Option<f64>, Box<dyn std::error::Error>> {
    let api_url = "https://multiboost-api-production.up.railway.app/regression-fit";
    
    println!("Making API request to: {}", api_url);
    
    let client = reqwest::Client::new();
    let response = client
        .post(api_url)
        .header("Content-Type", "application/json")
        .body(json_str.to_string())
        .send()
        .await?;
    
    if !response.status().is_success() {
        eprintln!("API request failed with status: {}", response.status());
        let error_text = response.text().await?;
        eprintln!("Error response: {}", error_text);
        return Ok(None);
    }
    
    let response_text = response.text().await?;
    println!("API Response:");
    println!("{}", response_text);
    
    // Extract the final OOS R² value from the response
    let r_squared = extract_oos_r_squared_from_api_response(&response_text);
    
    if r_squared.is_none() {
        println!("DEBUG: Failed to extract R² value from API response.");
        println!("Looking for patterns in response...");
        let lines: Vec<&str> = response_text.lines().collect();
        for line in lines.iter().rev().take(20) {
            if line.contains("OOS") || line.contains("R") || line.contains("test") || line.contains("house_sales") || line.contains("r_squared") {
                println!("  Potential line: {}", line);
            }
        }
    }
    
    Ok(r_squared)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <base_params_file>", args[0]);
        std::process::exit(1);
    }
    
    let base_params_file = &args[1];
    
    // Read the base parameters file as raw JSON
    let base_content = fs::read_to_string(base_params_file)?;
    let base_json: serde_json::Value = serde_json::from_str(&base_content)?;
    
    println!("Base parameters loaded from: {}", base_params_file);
    
    // Generate range of childActivePartitionRatio values from 0.0 to 0.75 with 0.01 increments
    let mut results: Vec<(f64, Option<f64>)> = Vec::new();
    
    let start_ratio = 0.0;
    let end_ratio = 0.75;
    let increment = 0.01;
    
    let mut current_ratio = start_ratio;
    while current_ratio <= end_ratio {
        let rounded_ratio = (current_ratio * 100.0_f64).round() / 100.0_f64; // Round to 2 decimal places
        
        println!("\n=== Testing childActivePartitionRatio = {:.2} ===", rounded_ratio);
        
        // Modify the JSON directly
        let mut test_json = base_json.clone();
        if let Some(x) = test_json.get_mut("x") {
            if let Some(child_ratios) = x.get_mut("childActivePartitionRatio") {
                if let Some(ratios_array) = child_ratios.as_array_mut() {
                    for ratio_val in ratios_array.iter_mut() {
                        *ratio_val = serde_json::json!(rounded_ratio);
                    }
                }
            }
        }
        
        let json_str = serde_json::to_string(&test_json)?;
        
        // Run the experiment via API
        match run_experiment_with_api(&json_str).await {
            Ok(Some(r_squared)) => {
                println!("✓ childActivePartitionRatio {:.2} -> OOS R² = {:.6}", rounded_ratio, r_squared);
                results.push((rounded_ratio, Some(r_squared)));
            }
            Ok(None) => {
                println!("✗ childActivePartitionRatio {:.2} -> Failed to extract R² value", rounded_ratio);
                results.push((rounded_ratio, None));
            }
            Err(e) => {
                println!("✗ childActivePartitionRatio {:.2} -> Error: {}", rounded_ratio, e);
                results.push((rounded_ratio, None));
            }
        }
        
        current_ratio += increment;
    }
    
    // Save results to file
    let results_filename = format!("childpartition_power_results_{}.csv", 
                                  chrono::Utc::now().format("%Y%m%d_%H%M%S"));
    
    let mut results_file = fs::File::create(&results_filename)?;
    writeln!(results_file, "childActivePartitionRatio,oos_r_squared")?;
    
    for (ratio, r_squared_opt) in &results {
        match r_squared_opt {
            Some(r_squared) => writeln!(results_file, "{:.2},{:.6}", ratio, r_squared)?,
            None => writeln!(results_file, "{:.2},", ratio)?, // Empty value for failed experiments
        }
    }
    
    println!("\n=== EXPERIMENT COMPLETE ===");
    println!("Results saved to: {}", results_filename);
    println!("Summary:");
    
    let successful_results: Vec<_> = results.iter()
        .filter_map(|(ratio, r_squared_opt)| r_squared_opt.map(|r2| (*ratio, r2)))
        .collect();
    
    if !successful_results.is_empty() {
        let best_result = successful_results.iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        
        if let Some((best_ratio, best_r_squared)) = best_result {
            println!("Best result: childActivePartitionRatio = {:.2}, OOS R² = {:.6}", 
                    best_ratio, best_r_squared);
        }
        
        println!("Successful experiments: {}/{}", successful_results.len(), results.len());
    } else {
        println!("No successful experiments completed.");
    }
    
    Ok(())
}