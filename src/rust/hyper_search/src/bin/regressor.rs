#[macro_use]
extern crate lazy_static;

use rand::Rng;
use run_script::ScriptOptions;
use mysql::*;
use mysql::prelude::*;
use regex::Regex;
use std::env;
use std::assert;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::cmp;
use std::ops::MulAssign;


pub mod mongodbext;
pub mod mariadbext;
pub mod dataset;
pub mod model;

// Utilities
fn round(x: f64, decimals: u32) -> f64 {
    let y = 10u32.pow(decimals) as f64;
    (x * y).round() / y
}

fn calculate_hash<T: Hash>(t: &T) -> u64 {
    let mut s = DefaultHasher::new();
    t.hash(&mut s);
    s.finish()
}

struct S<T> {
    elements: Vec<T>,
}

impl<T> std::ops::MulAssign<&T> for S<T>
where
    for<'a> T: std::ops::MulAssign<&'a T>,
{
    fn mul_assign(&mut self, val: &T) {
        for i in self.elements.iter_mut() {
            *i *= val;
        }
    }
}

#[tokio::main]
pub async fn main() -> Result<(), Box<dyn std::error::Error>> {

    let args: Vec<String> = env::args().collect();
	
    let model_type: model::ModelType = model::ModelType::regressor;

    let mut trial_num: i32 = 1;
    let mut rng = rand::thread_rng();

    // Mongodb connection uri
    let mongodb_uri: &str = "mongodb://localhost:27017";
    let creds = mongodbext::Credentials::get_credentials(mongodb_uri).await.unwrap();

    // Mariadb connection uri
    let database_name = model::ModelType::database_name(&model_type);
    let mariadb_uri: &str = &format!("mysql://{}:{}@localhost:3306/{}", creds.0, creds.1, database_name);
    let pool = Pool::new(mariadb_uri)?;
    let mut conn = pool.get_conn()?;

    // =========================
    // == REGRESSOR DATASETS == 
    // =========================
    let datasetnames: Vec<String> = vec![
        // String::from("tabular_benchmark/Regression/pol"),
       	// String::from("tabular_benchmark/Regression/cpu_act"),
        // String::from("tabular_benchmark/Regression/elevators"),
        // String::from("tabular_benchmark/Regression/wine_quality"),
        // String::from("tabular_benchmark/Regression/Ailerons"),
        // String::from("tabular_benchmark/Regression/houses"),
        // String::from("tabular_benchmark/Regression/house_16H"),
        // String::from("tabular_benchmark/Regression/diamonds"),
        // String::from("tabular_benchmark/Regression/Brazilian_houses"),
        // String::from("tabular_benchmark/Regression/Bike_Sharing_Demand"),
        // // String::from("tabular_benchmark/Regression/nyc-taxi-green-dec-2016"),
        // String::from("tabular_benchmark/Regression/house_sales"),
        String::from("tabular_benchmark/Regression/sulfur"),
        String::from("tabular_benchmark/Regression/medical_charges"),
        String::from("tabular_benchmark/Regression/MiamiHousing2016"),
        String::from("tabular_benchmark/Regression/superconduct"),
        String::from("tabular_benchmark/Regression/yprop_4_1"),
        String::from("tabular_benchmark/Regression/abalone"),
        // String::from("tabular_benchmark/Regression/delays_zurich_transport"),
        String::from("tabular_benchmark/Regression/Mixed/analcatdata_supreme"),
        String::from("tabular_benchmark/Regression/Mixed/visualizing_soil"),
        String::from("tabular_benchmark/Regression/Mixed/diamonds"),
        String::from("tabular_benchmark/Regression/Mixed/Mercedes_Benz_Greener_Manufacturing"),
        String::from("tabular_benchmark/Regression/Mixed/Brazilian_houses"),
        String::from("tabular_benchmark/Regression/Mixed/Bike_Sharing_Demand"),
        String::from("tabular_benchmark/Regression/Mixed/nyc-taxi-green-dec-2016"),
        String::from("tabular_benchmark/Regression/Mixed/house_sales"),
        String::from("tabular_benchmark/Regression/Mixed/particulate-matter-ukair-2017"),
        String::from("tabular_benchmark/Regression/Mixed/SGEMM_GPU_kernel_performance"),
        String::from("tabular_benchmark/Regression/Mixed/topo_2_1"),
        String::from("tabular_benchmark/Regression/Mixed/abalone"),
        String::from("tabular_benchmark/Regression/Mixed/seattlecrime6"),
        String::from("tabular_benchmark/Regression/Mixed/delays_zurich_transport"),
        String::from("tabular_benchmark/Regression/Mixed/Allstate_Claims_Severity"),
        String::from("tabular_benchmark/Regression/Mixed/Airlines_DepDelay_1M"),
        String::from("tabular_benchmark/Regression/Mixed/medical_charges")];

    // Generate simulation results
    for datasetname in datasetnames {
        let dataset = dataset::ClassificationDataset::new(&datasetname);
        let dataset_shape = dataset.shape();
        let numRows = dataset_shape.0;
        let numCols = dataset_shape.1;
	
        // let mut ratio: f64 = rng.gen::<f64>();    
        let mut ratio: f64 = rng.gen_range(0.5..1.0);
        let mut baseSteps: u32 = 25;
        let loss_fn: u32 = 0;
        let colsubsample_ratio: f32 = 1.0;
        let mut recursivefit: bool = true;

        // Would like to use a Parzen estimator as in TPE, but
        // the interface doesn't support vector inputs

        let basePath: String = String::from("/home/charles/src/C++/sandbox/Inductive-Boost");

        let mut proc: String = String::from("/src/script/");
	proc.push_str(model::ModelType::model_cmd(&model_type));

        for recursivefit in vec![true, false] {

            // Canned parameters
            // XXX
            // Memory footprint too large
            // let mut numGrids: usize = 8;
    	    // let childNumPartitions: Vec<f64>    = vec![1000.0, 500.0, 250.0, 100.0, 20.0, 10.0, 5.0, 1.0];
            // let childNumSteps: Vec<f64>         = vec![1.0, 1.0, 2.0, 3.0, 3.0, 2.0, 1.0, 1.0];
            // let childLearningRate: Vec<f64>     = vec![0.0001, 0.0001, 0.0002, 0.0002, 0.0003, 0.0003, 0.0004, 0.0004];
            // let childMaxDepth: Vec<f64>         = vec![20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0];
            // let childMinLeafSize: Vec<f64>      = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
            // let childMinimumGainSplit: Vec<f64> = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

            let mut numGrids: usize = 4;
            // XXX
            // let mut childNumPartitions: Vec<f64>    = vec![1.0,   0.5,  0.15, 0.01];
            let mut childNumPartitions: Vec<f64>    = vec![1.0, 0.25, 0.05, 0.005];
            let mut childNumSteps: Vec<f64>         = vec![1.0,   1.0,  2.0,  1.0];
            let mut childLearningRate: Vec<f64>     = vec![0.005, 0.01, 0.01, 0.02];
            let mut childMaxDepth: Vec<f64>         = vec![20.0,  20.0, 20.0, 20.0];
            let mut childMinLeafSize: Vec<f64>      = vec![1.0,   1.0,  1.0,  1.0];
            let mut childMinimumGainSplit: Vec<f64> = vec![0.0,   0.0,  0.0,  0.0];

            // childNumPartitions expressed in relative terms, make it absolute
            for elem in &mut childNumPartitions {
                *elem *= numRows as f64;
                *elem = elem.round() as f64;
            }

            // Stripe recursive v. nonrecursive cases
            if !recursivefit {
                // baseSteps = 25;
                for elem in &mut childLearningRate {
                    *elem *= 10.0;
                }
            }

            let specs: Vec<Vec<String>> = vec![
                childNumPartitions,
                childNumSteps,
                childLearningRate,
                childMaxDepth,
                childMinLeafSize,
                childMinimumGainSplit]
                .into_iter()
                .map(|v| v.into_iter().map(|i| i.to_string()).collect()).collect();

            let mut cmd: String = "".to_string();
            cmd.push_str(&basePath);
            cmd.push_str(&proc);
    
            cmd.push_str(" ");
            cmd.push_str(&numGrids.to_string());
            for spec in &specs {
                cmd.push_str(" ");
                let spec_flat = &spec.join(" ");
                cmd.push_str(spec_flat);
            }
            cmd.push_str(" ");
	
            cmd.push_str(&datasetname);                       cmd.push_str(" ");
            cmd.push_str(&baseSteps.to_string());             cmd.push_str(" ");
            cmd.push_str(&loss_fn.to_string());               cmd.push_str(" ");
            cmd.push_str(&colsubsample_ratio.to_string());    cmd.push_str(" ");

            cmd.push_str(&(recursivefit as i32).to_string());

            println!("RUNNING CASE: {}: \n{}", trial_num, cmd);

            let args = vec![];
            let options = ScriptOptions::new();

            let(code, output, _error) = run_script::run(
                &cmd,
                &args,
                &options,
            ).unwrap();

            let lines = output.lines();

            let mut folder = "";
            let mut index = "";
            let mut run_key: u64 = 0;

            let mut it: i32 = 0;

            for line in lines {
                if model::ITER.is_match(&line) {
                    for (_,[item]) in model::ITER.captures_iter(&line)
                        .map(|i| i.extract()) {
                        it = item.parse::<i32>()?;
                    }
                }
                else if model::FOLDER.is_match(&line) {
                    for (_,[item]) in model::FOLDER.captures_iter(&line)
                        .map(|i| i.extract()) {
	                folder = item;
                    }
                }
                else if model::INDEX.is_match(&line) {
                    for (_,[item]) in model::INDEX.captures_iter(&line)
                        .map(|i| i.extract()) {
                        index = item;
                    }
                    // At this point, both folder and index information 
                    // are available.
                    let data = mariadbext::run_key_data{dataset: datasetname.to_string(), 
                        folder: folder.to_string(), index: index.to_string()};
                    run_key = calculate_hash(&data);
                    let query = mariadbext::format_run_specification_query(run_key, &cmd, folder, index, &datasetname,
                        loss_fn, numRows, numCols, baseSteps, colsubsample_ratio, recursivefit, 0.2,
                        &specs);
                    let r = conn.query_drop(query).expect("Failed to insert into run_specification table");
                }
                else if model::OOS.is_match(&line) {
    	            for (_,[vals]) in model::OOS.captures_iter(&line)
                        .map(|i| i.extract()) {
            	        let mut parsed: Vec<String> = vals.split(", ").map(|i| i.to_string()).collect();
                        let query = mariadbext::format_outofsample_query(&model_type, run_key, &datasetname, it, parsed);
                        let r = conn.query_drop(query).expect("Failed to insert into outofsample table");
                    }
                }            
                else if model::IS.is_match(&line) {
                    for(_,[vals]) in model::IS.captures_iter(&line)
                        .map(|i| i.extract()) {
                        let mut parsed: Vec<String> = vals.split(", ").map(|i| i.to_string()).collect();
                        let query = mariadbext::format_insample_query(&model_type, run_key, &datasetname, it, parsed);
                        let r = conn.query_drop(query).expect("Failed to insert into insample table");
                    }
                }
            }
            println!("run_key: {}", run_key);
        }

        trial_num += 1;

    }

    // Never reached
    Ok(())
}
