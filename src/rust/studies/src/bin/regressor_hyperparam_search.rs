#[macro_use]
extern crate lazy_static;

use rand::Rng;
use run_script::ScriptOptions;
use mysql::*;
use mysql::prelude::*;

use std::env;
use std::assert;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::process;

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

#[tokio::main]
pub async fn main() -> Result<(), Box<dyn std::error::Error>> {
    
    let args: Vec<String> = env::args().collect();
    assert!(args.len() == 2);

    let model_type: model::ModelType = model::ModelType::regressor;
    let database_name = model::ModelType::database_name(&model_type);

    let mut trial_num: i32 = 1;
    let mut rng = rand::thread_rng();

    // Mongodb connection uri for credentials
    let mongodb_uri: &str = "mongodb://localhost:27017";
    let creds = mongodbext::Credentials::get_credentials(mongodb_uri).await.unwrap();

    let mariadb_uri: &str = &format!("mysql://{}:{}@localhost:3306/{}", creds.0, creds.1, database_name);

    let datasetname = &args[1];

    // =========================
    // == REGRESSOR DATASETS == 
    // =========================
    // let datasetnames: Vec<String> = vec![
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
        // String::from("tabular_benchmark/Regression/sulfur"),
        // String::from("tabular_benchmark/Regression/medical_charges"),
        // String::from("tabular_benchmark/Regression/MiamiHousing2016"),
        // String::from("tabular_benchmark/Regression/superconduct"),
        // String::from("tabular_benchmark/Regression/yprop_4_1"),
        // String::from("tabular_benchmark/Regression/abalone"),
        // String::from("tabular_benchmark/Regression/delays_zurich_transport"),
        // String::from("tabular_benchmark/Regression/Mixed/analcatdata_supreme"),
        // String::from("tabular_benchmark/Regression/Mixed/visualizing_soil"),
        // String::from("tabular_benchmark/Regression/Mixed/diamonds"),
        // String::from("tabular_benchmark/Regression/Mixed/Mercedes_Benz_Greener_Manufacturing"),
        // String::from("tabular_benchmark/Regression/Mixed/Brazilian_houses"),
        // String::from("tabular_benchmark/Regression/Mixed/Bike_Sharing_Demand"),
        // String::from("tabular_benchmark/Regression/Mixed/nyc-taxi-green-dec-2016"),
        // String::from("tabular_benchmark/Regression/Mixed/house_sales"),
        // String::from("tabular_benchmark/Regression/Mixed/particulate-matter-ukair-2017"),
        // String::from("tabular_benchmark/Regression/Mixed/SGEMM_GPU_kernel_performance"),
        // String::from("tabular_benchmark/Regression/Mixed/topo_2_1"),
        // String::from("tabular_benchmark/Regression/Mixed/abalone"),
        // String::from("tabular_benchmark/Regression/Mixed/seattlecrime6"),
        // String::from("tabular_benchmark/Regression/Mixed/delays_zurich_transport"),
        // String::from("tabular_benchmark/Regression/Mixed/Allstate_Claims_Severity"),
        // String::from("tabular_benchmark/Regression/Mixed/Airlines_DepDelay_1M"),
        // String::from("tabular_benchmark/Regression/Mixed/medical_charges")];
    
    let dataset = dataset::ClassificationDataset::new(&datasetname);

    // Generate simulation results
    loop {
        let dataset_shape = dataset.shape();
        let numRows = dataset_shape.0;
        let numCols = dataset_shape.1;
	
        let mut ratio1: f64 = rng.gen_range(0.0..0.5);
        let mut ratio2: f64 = rng.gen_range(1.0..1.25);
        let numGrids: usize = 2;
	let baseSteps: u32 = 75;
        let loss_fn: u32  = 0;
        let colsubsample_ratio: f32 = 1.0;
	let clampGradient: usize = 0;
        let upperVal: f32 = 1000.0;
        let lowerVal: f32 = -1000.0;
        let splitRatio: f32 = 0.20;

        // Would like to use a Parzen estimator as in TPE, but
        // the interface doesn't support vector inputs

        let mut childNumPartitions:    Vec<f64> = vec![0.0; numGrids];
        let mut childNumSteps:         Vec<f64> = vec![0.0; numGrids];
        let mut childLearningRate:     Vec<f64> = vec![0.0; numGrids];
        let childMaxDepth:             Vec<f64> = vec![0.0; numGrids];
        let mut childMinLeafSize:      Vec<f64> = vec![0.0; numGrids];
        let mut childMinimumGainSplit: Vec<f64> = vec![0.0; numGrids];
        let mut numPartitions: f64 = numRows as f64;
        let mut numPartitionsBase: f64 = numPartitions;
        let mut numSteps: f64 = 1.;

        for ind in 0..numGrids {
            numPartitionsBase = numPartitions;
            numPartitions *= ratio1;

            numSteps = (numSteps * ratio2).round() as f64;

            numPartitions = (numPartitions / 10.).round() as f64 * 10.;

	    let mut maxDepth:         i32 = (numRows as f64).log2().floor() as i32 + 1;
            // let mut numSteps:         f64 = rng.gen_range(1..4).into();
            let mut learningRate:     f64 = rng.gen_range(0.001..0.2);
            let mut minLeafSize:      f64 = rng.gen_range(1..2).into();
            let mut minimumGainSplit: f64 = 0.0;

            if (ind == 1) {
                numPartitions = 10.;
                learningRate = 0.10;
                numSteps = 1.;
            }
       
            if (false) {
                numPartitions = 10.;
                learningRate = 0.10;
                // learningRate = 0.05410;            
                numSteps = 1.;
            }

            if (ind == 2) {
                // numPartitions = 10.;
                // learningRate = 0.02648;
            }

            if (ind==0) {
                let mut count: u32 = 0;
                let mut r = mariadbext::check_for_existing_run_dim2(mariadb_uri, &creds, numPartitions as i32, &datasetname);
                while (r[0] != 0) {
                    println!("{} bad starting partition; already tested...", numPartitions as i32);
                    ratio1 = rng.gen_range(0.0..1.0);
                    numPartitions = numPartitionsBase * ratio1;
                    numPartitions = (numPartitions / 10.).round() as f64 * 10.;
                    if numPartitions < 1. {
                        numPartitions = 1_f64;
                    }
                    r = mariadbext::check_for_existing_run_dim2(mariadb_uri, &creds, numPartitions as i32, &datasetname);
                    count += 1;
                    if (count > 10000) {
                        process::exit(1);
                    }
                }
                println!("found starting partition {}", numPartitions as i32);
                numSteps = 1.0;
                learningRate = 0.10;
                if (numPartitions == 1.0) {
                    numPartitions = 2.0;
                }
            }

            if numPartitions < 1. {
            	numPartitions = 1_f64;
            }

            // if (ind == 0) {
            //     numPartitions = 31.0;
            //     numSteps = 1.;
	    // 	learningRate = 0.15;
            // }

            if (ind == 1) {
                // numPartitions = 4.0;
                // numSteps = 1.;
                // learningRate = 0.10;
            }
       
            childNumPartitions[ind]       = numPartitions as i32 as f64;
            childNumSteps[ind]            = numSteps;
            childLearningRate[ind]        = learningRate;
            childMinLeafSize[ind]         = minLeafSize;
            childMinimumGainSplit[ind]    = minimumGainSplit;

            ratio1 = rng.gen_range(0.0..1.0);
            ratio2 = rng.gen_range(1.0..1.25);

        }

        let specs: Vec<Vec<String>> = vec![childNumPartitions,
                                   	childNumSteps,
   					childLearningRate,
					childMaxDepth,
					childMinLeafSize,
					childMinimumGainSplit]
            .into_iter()
            .map(|v| v.into_iter().map(|i| i.to_string()).collect()).collect();	
        
        let basePath: String = String::from("/home/charles/src/C++/sandbox/Inductive-Boost");
        
        let mut proc: String = String::from("/src/script/");
        proc.push_str(model::ModelType::model_cmd(&model_type));

        for recursivefit in vec![true] {

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

            let(_code, output, _error) = run_script::run(
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
                    let query = mariadbext::format_run_specification_query(run_key, &cmd, folder, index, datasetname,
                        loss_fn, numRows, numCols, baseSteps, colsubsample_ratio, recursivefit, 
                        clampGradient, upperVal, lowerVal, splitRatio,
                        &specs);
                    mariadbext::insert_to_table(mariadb_uri, &creds, &query);
                }
                else if model::OOS.is_match(&line) {
    	            for (_,[vals]) in model::OOS.captures_iter(&line)
                        .map(|i| i.extract()) {
            	        let parsed: Vec<String> = vals.split(", ").map(|i| i.to_string()).collect();
                        let query = mariadbext::format_outofsample_query(&model_type, run_key, datasetname, it, parsed);
                        mariadbext::insert_to_table(mariadb_uri, &creds, &query);
                    }
                }            
                else if model::IS.is_match(&line) {
                    for(_,[vals]) in model::IS.captures_iter(&line)
                        .map(|i| i.extract()) {
                        let parsed: Vec<String> = vals.split(", ").map(|i| i.to_string()).collect();
                        let query = mariadbext::format_insample_query(&model_type, run_key, datasetname, it, parsed);
                        mariadbext::insert_to_table(mariadb_uri, &creds, &query);
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