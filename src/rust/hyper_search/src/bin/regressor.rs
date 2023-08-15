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
	
        // let mut ratio: f64 = rng.gen::<f64>();    
        let mut ratio: f64 = rng.gen_range(0.5..1.0);
        // XXX
        let mut numGrids: usize = 1;
	let baseSteps: u32 = 50;
        let loss_fn: u32  = 0;
        let colsubsample_ratio: f32 = 0.85;
        let mut recursivefit: bool = true;

        // Would like to use a Parzen estimator as in TPE, but
        // the interface doesn't support vector inputs

        let mut childNumPartitions:    Vec<f64> = vec![0.0; numGrids];
        let mut childNumSteps:         Vec<f64> = vec![0.0; numGrids];
        let mut childLearningRate:     Vec<f64> = vec![0.0; numGrids];
        let mut childMaxDepth:         Vec<f64> = vec![0.0; numGrids];
        let mut childMinLeafSize:      Vec<f64> = vec![0.0; numGrids];
        let mut childMinimumGainSplit: Vec<f64> = vec![0.0; numGrids];
        let mut numPartitions: f64 = numRows as f64;
       
        for ind in 0..numGrids {
            numPartitions *= ratio;
            if numPartitions < 1. {
                numPartitions = 1_f64;
            }
	    let maxDepth:         i32 = (numRows as f64).log2().floor() as i32 + 1;
            let mut numSteps:     f64 = 1.;
            let learningRate:     f64 = rng.gen_range(0.00005..0.2);
            let maxDepth:         f64 = rng.gen_range(maxDepth..maxDepth+1).into();
            let minLeafSize:      f64 = rng.gen_range(1..2).into();
            let minimumGainSplit: f64 = 0.0;

            childNumPartitions[ind]       = round(numPartitions, 0);
            childNumSteps[ind]            = numSteps;
            childLearningRate[ind]        = learningRate;
            childMinLeafSize[ind]         = minLeafSize;
            childMinimumGainSplit[ind]    = minimumGainSplit;

            ratio = rng.gen_range(0.0..1.0);
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

        for recursivefit in vec![true, false] {

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
            
        }        
       

    }

    // Never reached
    Ok(())
}