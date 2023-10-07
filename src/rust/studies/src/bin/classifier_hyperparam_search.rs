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
fn round(x: f32, decimals: u32) -> f32 {
    let y = 10u32.pow(decimals) as f32;
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
	
    let model_type: model::ModelType = model::ModelType::classifier;

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

    let datasetname = &args[1];

    // =========================
    // == CLASSIFIER DATASETS == 
    // =========================
    // let datasetname = "analcatdata_boxing1";
    // let datasetname = "breast";
    // let datasetname = "analcatdata_lawsuit";
    // let datasetname = "analcatdata_asbestos";
    // let datasetname = "analcatdata_boxing2";
    // let datasetname = "analcatdata_creditscore";
    // let datasetname = "analcatdata_cyyoung8092";
    // let datasetname = "analcatdata_cyyoung9302";
    // let datasetname = "bupa";
    // let datasetname = "cleve";
    // let datasetname = "appendicitis";
    // let datasetname = "german";
    // let datasetname = "crx";'
    // let datasetname = "breast_cancer";

    let dataset = dataset::ClassificationDataset::new(&datasetname);

    // Generate simulation results
    loop {
        let dataset_shape = dataset.shape();
        let numRows = dataset_shape.0;
        let numCols = dataset_shape.1;
	
        let mut ratio: f32 = rng.gen_range(0.25..1.0);
        let mut ratio1: f32 = rng.gen_range(0.25..1.0);
        // let numGrids = rng.gen_range(2..6) as usize;
        let numGrids: usize = 2;
	let baseSteps: u32 = 100;
        let loss_fn: u32  = 6;
        let lossPower: f32 = 1.0;
        let colsubsample_ratio: f32 = 0.85;
        let _recursivefit: bool = true;
        let clampGradient: usize = 1;
        let upperVal: f32 = 1.0;
        let lowerVal: f32 = -1.0;
        let testOOS: bool = true;
        let splitRatio: f32 = 0.0;

        // Would like to use a Parzen estimator as in TPE, but
        // the interface doesn't support vector inputs

        let mut childNumPartitions:       Vec<f32> = vec![0.0; numGrids];
        let mut childNumSteps:            Vec<f32> = vec![0.0; numGrids];
        let mut childLearningRate:        Vec<f32> = vec![0.0; numGrids];
        let mut childPartitionUsageRatio: Vec<f32> = vec![0.0; numGrids];
        let mut childMaxDepth:            Vec<f32> = vec![0.0; numGrids];
        let mut childMinLeafSize:         Vec<f32> = vec![0.0; numGrids];
        let mut childMinimumGainSplit:    Vec<f32> = vec![0.0; numGrids];
        let mut numPartitions: f32 = numRows as f32;
        let mut numPartitionsBase: f32 = numPartitions;


        for ind in 0..numGrids {
	    numPartitions *= ratio;
            numPartitions = (numPartitions / 10.).round() as f32 * 10.;
            if numPartitions < 1. {
                numPartitions = 1_f32;
            }
		
            let numSteps:         f32 = 1.;
            let learningRate:     f32 = 0.01;
            let partitionUsageRatio: f32 = 0.5;
            // let maxDepth:         f32 = (numRows as f32).log2().floor() + 1.0;
            let maxDepth:         f32 = 0.0;
            // let minLeafSize:      f32 = rng.gen_range(1.0..2.0).into();
            let minLeafSize:      f32 = 1.0;
            let minimumGainSplit: f32 = 0.0;
 
            if ind==0 {
                let mut count: u32 = 0;
                /*
                let mut r = mariadbext::check_for_existing_run_dim1(mariadb_uri, &creds, numPartitions as i32, &datasetname);
                while r[0] != 0 {
                    println!("{} bad starting partition; already tested...", numPartitions as i32);
                    ratio1 = rng.gen_range(0.0..1.0);
                    numPartitions = numPartitionsBase * ratio1;
                    numPartitions = (numPartitions / 10.).round() as f32 * 10.;
                    if numPartitions < 1. {
                        numPartitions = 1_f32;
                    }
                    r = mariadbext::check_for_existing_run_dim1(mariadb_uri, &creds, numPartitions as i32, &datasetname);
                    count += 1;
                    if count > 100000 {
                        process::exit(1);
                    }
                }
                println!("found starting partition {}", numPartitions as i32);
		*/
                numPartitions = 1820.0;
            }

            childNumPartitions[ind]       = round(numPartitions, 0);
            childNumSteps[ind]            = numSteps;
            childLearningRate[ind]        = learningRate;
            childPartitionUsageRatio[ind] = partitionUsageRatio;
            childMaxDepth[ind]            = maxDepth;
            childMinLeafSize[ind]         = minLeafSize;
            childMinimumGainSplit[ind]    = minimumGainSplit;

	    ratio = rng.gen_range(0.25..1.0);
        }

        let specs: Vec<Vec<String>> = vec![childNumPartitions,
	   			       childNumSteps,
                                       childLearningRate,
                                       childPartitionUsageRatio,
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
            cmd.push_str(&lossPower.to_string());	      cmd.push_str(" ");
            cmd.push_str(&colsubsample_ratio.to_string());    cmd.push_str(" ");
            cmd.push_str(&(recursivefit as i32).to_string()); cmd.push_str(" ");
            cmd.push_str(&clampGradient.to_string());         cmd.push_str(" ");
            cmd.push_str(&upperVal.to_string());              cmd.push_str(" ");
            cmd.push_str(&lowerVal.to_string());              cmd.push_str(" ");
            cmd.push_str(&testOOS.to_string());               cmd.push_str(" "); 
            cmd.push_str(&splitRatio.to_string());

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

            let datasetname_test = datasetname.replace("_train", "_test");

            for line in lines {

                if model::ITER.is_match(&line) {
                    for (_,[item]) in model::ITER.captures_iter(&line)
                        .map(|i| i.extract()) {
                        it = item.parse::<i32>()?;
                        // noop
                    }
                }
                else if model::FOLDER.is_match(&line) {
                    for (_,[item]) in model::FOLDER.captures_iter(&line)
                        .map(|i| i.extract()) {
	                folder = item;
                        // noop
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
                        loss_fn, lossPower, numRows, numCols, baseSteps, colsubsample_ratio, recursivefit, 
                        clampGradient, upperVal, lowerVal, splitRatio,
                        &specs);
                    let _r = conn.query_drop(query).expect("Failed to insert into run_specification table");
                }
                else if model::OOS_patt(&datasetname_test).is_match(&line) {
    	            for (_,[vals]) in model::OOS_patt(&datasetname_test).captures_iter(&line)
                        .map(|i| i.extract()) {
    	                let parsed: Vec<String> = vals.split(", ").map(|i| i.to_string()).collect();
                        let query = mariadbext::format_outofsample_query(&model_type, run_key, datasetname, it, parsed);
                        let _r = conn.query_drop(query).expect("Failed to insert into outofsample table");
                    }
                }            
                else if model::IS_patt(datasetname).is_match(&line) {
                    for(_,[vals]) in model::IS_patt(datasetname).captures_iter(&line)
                        .map(|i| i.extract()) {
                        let parsed: Vec<String> = vals.split(", ").map(|i| i.to_string()).collect();
                        let query = mariadbext::format_insample_query(&model_type, run_key, datasetname, it, parsed);
                        let _r = conn.query_drop(query).expect("Failed to insert into insample table");
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
