#[macro_use]
extern crate lazy_static;

use std::env;
use std::assert;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::process;
use regex::Regex;

use run_script::ScriptOptions;
use mysql::*;
use mysql::prelude::*;

pub mod mongodbext;
pub mod mariadbext;
pub mod model;
pub mod dataset;

// Utilities
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

    // Mongodb connection uri
    let mongodb_uri: &str = "mongodb://localhost:27017";
    let creds = mongodbext::Credentials::get_credentials(mongodb_uri).await.unwrap();

    // Mariadb connection uri
    let database_name = model::ModelType::database_name(&model_type);
    let mariadb_uri: &str = &format!("mysql://{}:{}@localhost:3306/{}", creds.0, creds.1, database_name);
    let pool = Pool::new(mariadb_uri)?;
    let mut conn = pool.get_conn()?;

    let datasetname = &args[1];
    let datasetname_test = datasetname.replace("_train", "_test");

    // Dataset info
    let dataset = dataset::ClassificationDataset::new(&datasetname);
    let datasetShape = dataset.shape();
    let numRows = datasetShape.0;
    let numCols = datasetShape.1;

    // Vector inputs
    let mut childNumPartitions:		Vec<f32> = vec![1000.0, 100.0];
    let mut childNumSteps:		Vec<f32> = vec![1.00; childNumPartitions.len()];
    let mut childLearningRate:		Vec<f32> = vec![0.001; childNumPartitions.len()];
    let mut childPartitionUsageRatio:	Vec<f32> = vec![0.00; childNumPartitions.len()];
    let mut childMaxDepth:		Vec<f32> = vec![0.00; childNumPartitions.len()];
    let mut childMinLeafSize:		Vec<f32> = vec![1.00; childNumPartitions.len()];
    let mut childMinimumGainSplit:	Vec<f32> = vec![0.00; childNumPartitions.len()];

    // Scalar inputs
    let mut numGrids:		usize = childNumPartitions.len();
    let mut numIterations:	u32   = 50;
    let mut lossFn:		u32   = 12;
    let mut lossPower:		f32   = 1.0;
    let mut colSubsampleRatio:	f32   = 1.0;
    let mut recursiveFit:	bool  = true;
    let mut clampGradient:	usize = 1;
    let mut upperVal:		f32   = 1.0;
    let mut lowerVal:		f32   = -1.0;
    let mut runOnTestData:	usize = 1;
    let mut splitRatio:		f32   = 0.0;

    let specs: Vec<Vec<String>> = vec![ childNumPartitions,
                                        childNumSteps,
		              	        childLearningRate,
					childPartitionUsageRatio,
					childMaxDepth,
					childMinLeafSize,
					childMinimumGainSplit]
    .into_iter()
    .map(|v| v.into_iter().map(|w| w.to_string()).collect()).collect();

    let base_path: String = String::from("/home/charles/src/C++/sandbox/Inductive-Boost");

    let mut proc: String = String::from("/src/script/");
    proc.push_str(model::ModelType::model_cmd(&model_type));

    struct case {
        lossFn: u32,
        lossPower: f32,
        clampGradient: usize,
        upperVal: f32,
        lowerVal: f32,
        childPartitionUsageRatio: Vec<f32>
    }

    /*
    let cases: Vec<case> = vec![
        case{lossFn: 3, lossPower: 1.0, clampGradient: 1, upperVal: 1.0, lowerVal: -1.0, childPartitionUsageRatio: vec![0.5]}, // Exp loss
        case{lossFn: 1, lossPower: 1.0, clampGradient: 1, upperVal: 1.0, lowerVal: -1.0, childPartitionUsageRatio: vec![0.5]}, // Bin dev loss
        case{lossFn: 8, lossPower: 1.0, clampGradient: 1, upperVal: 1.0, lowerVal: -1.0, childPartitionUsageRatio: vec![0.5]}, // Squared loss
        case{lossFn: 6, lossPower: 1.0, clampGradient: 1, upperVal: 1.0, lowerVal: -1.0, childPartitionUsageRatio: vec![0.5]}, // Synth loss - cubic
        case{lossFn: 7, lossPower: 1.0, clampGradient: 1, upperVal: 1.0, lowerVal: -1.0, childPartitionUsageRatio: vec![0.5]}, // Synth loss - cube root
        ];
    */
 
    let mut cases: Vec<case> = Vec::new();
    for i in 0..200 {
        let c = case{lossFn: 12, lossPower: i as f32/20., clampGradient: 1, upperVal: 1.0, lowerVal: -1.0, childPartitionUsageRatio: vec![0.5]};
        cases.push(c);
    }
    let mut run_keys: Vec<u64> = Vec::new();
	
    for case in cases {

        lossFn = case.lossFn;
        lossPower = case.lossPower;
        clampGradient = case.clampGradient;
        upperVal = case.upperVal;
        lowerVal = case.lowerVal;

        let mut cmd: String = "".to_string();
        cmd.push_str(&base_path);
        cmd.push_str(&proc);
        cmd.push_str(" ");
        cmd.push_str(&numGrids.to_string());
 
        for spec in &specs {
            cmd.push_str(" ");
            let spec_flat = &spec.join(" ");
            cmd.push_str(spec_flat);
        }

        cmd.push_str(" ");

        cmd.push_str(&datasetname);                      cmd.push_str(" ");
        cmd.push_str(&numIterations.to_string());        cmd.push_str(" ");
        cmd.push_str(&lossFn.to_string());               cmd.push_str(" ");
	cmd.push_str(&lossPower.to_string());		 cmd.push_str(" ");
        cmd.push_str(&colSubsampleRatio.to_string());    cmd.push_str(" ");
        cmd.push_str(&(recursiveFit as i32).to_string());cmd.push_str(" ");
        cmd.push_str(&clampGradient.to_string());        cmd.push_str(" ");
        cmd.push_str(&upperVal.to_string());             cmd.push_str(" ");
        cmd.push_str(&lowerVal.to_string());		 cmd.push_str(" ");
        cmd.push_str(&runOnTestData.to_string());	 cmd.push_str(" ");
        // cmd.push_str(&splitRatio.to_string());


        println!("call: {} \n", cmd);

        let args = vec![];
        let options = ScriptOptions::new();
        let (_code, output, _error) = run_script::run(
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
                    lossFn, lossPower, numRows, numCols, numIterations, colSubsampleRatio, recursiveFit, clampGradient,
                    upperVal, lowerVal, splitRatio, &specs);
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
        run_keys.push(run_key);
    }

    println!("run keys: {:?}", run_keys);
    // Never reached
    Ok(())
}