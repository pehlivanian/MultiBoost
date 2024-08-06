#[macro_use]
extern crate lazy_static;

use std::env;
use std::assert;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::process::Command;
use regex::Regex;

use run_script::ScriptOptions;
use mysql::*;
use mysql::prelude::*;

pub mod mongodbext;
pub mod mariadbext;
pub mod model;
pub mod dataset;

// static CLASSIFIER_PROBLEM: bool = true;
// const model_type: model::ModelType	= model::ModelType::classifier;

static CLASSIFIER_PROBLEM: bool = false;
const model_type: model::ModelType	= model::ModelType::regressor;

const numIterations:	 u32		= 250;
// CLASSIFIER = TRUE
// const lossFn:		 u32		= 5;
const lossFn:		 u32		= 0;
const lossPower:	 f32		= 1.0;
const colSubsampleRatio: f32		= 1.0;
const recursiveFit:	 bool		= true;
const clampGradient:	 usize		= 1;
const upperVal:		 f32		= 5.0;
const lowerVal:		 f32		= -5.0;
const runOnTestData:	 usize		= 1;
const splitRatio:	 f32		= 0.0;

// Utilities
fn calculate_hash<T: Hash>(t: &T) -> u64 {
    let mut s = DefaultHasher::new();
    t.hash(&mut s);
    s.finish()
}

fn commandStr(specs: &Vec<Vec<String>>, datasetTrainName: &str) -> String {

    let base_path: String = String::from("/home/charles/src/C++/sandbox/Inductive-Boost");

    let mut proc: String = String::from("/src/script/");
    proc.push_str(model::ModelType::model_cmd(&model_type));

    let numGrids: usize = specs[0].len();
    let mut cmd: String = "".to_string();
    cmd.push_str(&base_path);
    cmd.push_str(&proc); cmd.push_str(" ");
    cmd.push_str(&numGrids.to_string()); 

    for spec in specs {
        cmd.push_str(" ");
        let specFlat = spec.join(" ");
        cmd.push_str(&specFlat);
    }

    cmd.push_str(" ");

    cmd.push_str(&datasetTrainName);			cmd.push_str(" ");
    cmd.push_str(&numIterations.to_string());		cmd.push_str(" ");
    cmd.push_str(&lossFn.to_string());			cmd.push_str(" ");
    cmd.push_str(&lossPower.to_string());		cmd.push_str(" ");
    cmd.push_str(&colSubsampleRatio.to_string());	cmd.push_str(" ");
    cmd.push_str(&(recursiveFit as i32).to_string());	cmd.push_str(" ");
    cmd.push_str(&clampGradient.to_string());		cmd.push_str(" ");
    cmd.push_str(&upperVal.to_string());		cmd.push_str(" ");
    cmd.push_str(&lowerVal.to_string());		cmd.push_str(" ");
    cmd.push_str(&runOnTestData.to_string());		cmd.push_str(" ");
    cmd.push_str(&splitRatio.to_string());


        println!("call: {}", cmd);

    cmd
}

#[tokio::main]
pub async fn main() -> Result<(), Box<dyn std::error::Error>> {
    
    let args: Vec<String> = env::args().collect();
    assert!(args.len() == 2);

    // Dataset info
    let datasetTrainName = &args[1];
    let datasetTestName = datasetTrainName.replace("_train", "_test");
    let dataset = dataset::ClassificationDataset::new(&datasetTrainName);
    let datasetShape = dataset.shape();
    let numRows = datasetShape.0;
    let numCols = datasetShape.1;


    // Mongodb connection uri
    let mongodb_uri: &str = "mongodb://localhost:27017";
    let creds = mongodbext::Credentials::get_credentials_static(mongodb_uri).await.unwrap();

    // Mariadb connection uri
    let databaseName = model::ModelType::database_name(&model_type);
    let mariadb_uri: &str = &format!("mysql://{}:{}@localhost:3306/{}", creds.0, creds.1, databaseName);
    let pool = Pool::new(mariadb_uri)?;
    let mut conn = pool.get_conn().expect("Bad mariadb connection");;
    
    // Vector inputs
    let mut childNumPartitions:		Vec<f32> = vec![500.0, 25.0];
    let mut childNumSteps:		Vec<f32> = vec![1.00; childNumPartitions.len()];
    let mut childLearningRate:		Vec<f32> = vec![0.01; childNumPartitions.len()];
    let mut childPartitionUsageRatio:	Vec<f32> = vec![0.50; childNumPartitions.len()];
    let mut childMaxDepth:		Vec<f32> = vec![0.00; childNumPartitions.len()];
    let mut childMinLeafSize:		Vec<f32> = vec![1.00; childNumPartitions.len()];
    let mut childMinimumGainSplit:	Vec<f32> = vec![0.00; childNumPartitions.len()];

    let numGrids: usize = childNumPartitions.len();

    struct case {
        childLearningRate: Vec<f32>,
        childPartitionUsageRatio: Vec<f32>
    }

    let mut cases: Vec<case> = Vec::new();
    for i in 0..51 {
        for j in 0..51 {
            let c = case{childLearningRate: vec![i as f32 / (50.0 * 5.0); numGrids],
                         childPartitionUsageRatio: vec![j as f32 / 50.0; numGrids]};
            cases.push(c);
        }
    }

    let mut run_keys: Vec<u64> = Vec::new();

    for case in cases {

        let mut caseVar0           = case.childLearningRate;
        let mut caseVar1           = case.childPartitionUsageRatio;


        if (caseVar0[0] == 0. || caseVar1[0] == 0.) {
            continue;
        }

	let mut r = mariadbext::run_exists(mariadb_uri, &creds, caseVar0[0], caseVar1[0], &datasetTrainName);
	if (r[0] > 0) {
            println!("Case childLearningRate: {} childPartitionUsageRatio: {} exists",
                caseVar0[0], caseVar1[0]);
            continue;
        }
   
    
        let mut vecSpecs: Vec<Vec<String>> = vec![ childNumPartitions.clone(),
         	                               childNumSteps.clone(),
			             	        caseVar0,
				   		caseVar1,
						childMaxDepth.clone(),
						childMinLeafSize.clone(),
						childMinimumGainSplit.clone()]
            .into_iter()
            .map(|v| v.into_iter().map(|w| w.to_string()).collect()).collect();                

        let cmd: String = commandStr(&vecSpecs, &datasetTrainName);

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
                    Command::new("rm")
                        .args(["-rf", folder])
                        .output()
                        .expect("Failed to remove digest.");
                }
            }
            else if model::INDEX.is_match(&line) {
                for (_,[item]) in model::INDEX.captures_iter(&line)
                    .map(|i| i.extract()) {
                    index = item;
                }
                // At this point, both folder and index information 
                // are available.
                let data = mariadbext::run_key_data{dataset: datasetTrainName.to_string(), 
                    folder: folder.to_string(), index: index.to_string()};
                run_key = calculate_hash(&data);
                let query = mariadbext::format_run_specification_query(run_key, &cmd, folder, index, datasetTrainName,
                    lossFn, lossPower, numRows, numCols, numIterations, colSubsampleRatio, recursiveFit, clampGradient,
                    upperVal, lowerVal, splitRatio, &vecSpecs);
                let _r = conn.query_drop(query).expect("Failed to insert into run_specification table");
            }
            else if model::OOS_patt(&datasetTestName).is_match(&line) {
                for (_,[vals]) in model::OOS_patt(&datasetTestName).captures_iter(&line)
                    .map(|i| i.extract()) {
    	            let parsed: Vec<String> = vals.split(", ").map(|i| i.to_string()).collect();
                    let query = mariadbext::format_outofsample_query(&model_type, run_key, datasetTrainName, it, parsed);
                    let _r = conn.query_drop(query).expect("Failed to insert into outofsample table");
                }
            }                        
            else if model::IS_patt(datasetTrainName).is_match(&line) {
                for(_,[vals]) in model::IS_patt(datasetTrainName).captures_iter(&line)
                    .map(|i| i.extract()) {
                    let parsed: Vec<String> = vals.split(", ").map(|i| i.to_string()).collect();
                    let query = mariadbext::format_insample_query(&model_type, run_key, datasetTrainName, it, parsed);
                    let _r = conn.query_drop(query).expect("Failed to insert into insample table");
                }
            }
        }
        
        run_keys.push(run_key);

    }


    // Never reached
    Ok(())
}