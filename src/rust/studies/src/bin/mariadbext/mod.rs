use mysql::*;
use mysql::prelude::*;

use crate::model;

#[derive(Hash)]
#[allow(non_camel_case_types)]
pub struct run_key_data {
    pub dataset: String,
    pub folder: String,
    pub index: String
}

pub fn get_connection(uri: &str, _creds: &(String, String)) -> Result<mysql::PooledConn, mysql::Error> {
    let pool = Pool::new(uri).unwrap();
    pool.get_conn()
}

pub fn insert_to_table(uri: &str, creds: &(String, String), query: &str) {
    let mut conn = get_connection(uri, creds).unwrap();
    let _r = conn.query_drop(query).expect("Failed to insert to table");
}

pub fn run_exists(uri: &str, creds: &(String, String), learning_rate: f32, active_partition_ratio: f32, 
    datasetname: &str) -> Vec<i32> {
    let mut conn = get_connection(uri, creds).unwrap();
    let query = format!("select count(*) from run_specification where CAST(learning_rate0 as FLOAT) = CAST({} as FLOAT)  and CAST(active_partition_ratio0 as FLOAT) = CAST({} as FLOAT) and dataset_name=\"{}\"",
        learning_rate.to_string(), active_partition_ratio.to_string(), datasetname);
    let r = conn.query(query).expect("Failed to select from table");
    r
}

pub fn check_for_existing_run_dim1(uri: &str, creds: &(String, String), num_partitions: i32, datasetname: &str) -> Vec<i32> {
    let mut conn = get_connection(uri, creds).unwrap();
    let query = format!("select count(*) from run_specification where loss_fn = 6 and num_partitions0 = {} and num_partitions1 = 0 and dataset_name=\"{}\"",
        num_partitions.to_string(),
        datasetname);
    let r = conn.query(query).expect("Failed to select from table");
    r
}

pub fn check_for_existing_run_dim2(uri: &str, creds: &(String, String), num_partitions: i32, datasetname: &str) -> Vec<i32> {
    let mut conn = get_connection(uri, creds).unwrap();
    let query = format!("select count(*) from run_specification where num_partitions1 = 10 and num_partitions0 = {} and dataset_name=\"{}\"",
        num_partitions.to_string(),
        datasetname);
    let r = conn.query(query).expect("Failed to select from table");
    r
}

pub fn check_for_existing_run_dim3(uri: &str, creds: &(String, String), num_partitions: i32, datasetname: &str) -> Vec<i32> {
    let mut conn = get_connection(uri, creds).unwrap();
    let query = format!("select count(*) from run_specification where num_partitions0 = 40 and num_partitions1 = 4 and num_partitions2 = {} and dataset_name=\"{}\"",
        num_partitions.to_string(),
        datasetname);
    let r = conn.query(query).expect("Failed to select from table");
    r
}

pub fn format_run_specification_query(run_key: u64, cmd: &str, folder: &str, index: &str, datasetname: &str,
    loss_fn: u32, loss_power: f32, n_rows: usize, n_cols: usize, basesteps: u32, colsubsample_ratio: f32, 
    rcsive: bool, clamp_gradient: usize, upper_val: f32, lower_val: f32, split_ratio: f32, specs: &Vec<Vec<String>>) -> String {
    let mut query = format!("INSERT INTO run_specification (run_key, cmd, folder, idx, dataset_name, loss_fn, loss_power, n_rows, n_cols, basesteps, colsubsample_ratio, rcsive, clamp_gradient, upper_val, lower_val, split_ratio, num_partitions0, num_partitions1, num_partitions2, num_partitions3, num_partitions4, num_partitions5, num_partitions6, num_partitions7, num_partitions8, num_partitions9, num_steps0, num_steps1, num_steps2, num_steps3, num_steps4, num_steps5, num_steps6, num_steps7, num_steps8, num_steps9, learning_rate0, learning_rate1, learning_rate2, learning_rate3, learning_rate4, learning_rate5, learning_rate6, learning_rate7, learning_rate8, learning_rate9, active_partition_ratio0, active_partition_ratio1, active_partition_ratio2, active_partition_ratio3, active_partition_ratio4, active_partition_ratio5, active_partition_ratio6, active_partition_ratio7, active_partition_ratio8, active_partition_ratio9, max_depth0, max_depth1, max_depth2, max_depth3, max_depth4, max_depth5, max_depth6, max_depth7, max_depth8, max_depth9, min_leafsize0, min_leafsize1, min_leafsize2, min_leafsize3, min_leafsize4, min_leafsize5, min_leafsize6, min_leafsize7, min_leafsize8, min_leafsize9, min_gainsplit0, min_gainsplit1, min_gainsplit2, min_gainsplit3, min_gainsplit4, min_gainsplit5, min_gainsplit6, min_gainsplit7, min_gainsplit8, min_gainsplit9) VALUES (\"{}\", \"{}\", \"{}\", \"{}\", \"{}\", {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}",
        run_key.to_string(),
        cmd,
        folder,
        index,
        datasetname,
        loss_fn.to_string(),
        loss_power.to_string(),
        n_rows.to_string(),
        n_cols.to_string(),
        basesteps.to_string(),
        colsubsample_ratio.to_string(),
        rcsive.to_string(),
        clamp_gradient.to_string(),
        upper_val.to_string(),
        lower_val.to_string(),        
        split_ratio.to_string());

    for spec in specs {
        let sz = spec.len();
        let mut suffix: String = ", ".to_string();
        suffix.push_str(&vec!["0"; 10-sz].join(", "));
        let mut spec_flat: String = ", ".to_string();
        spec_flat.push_str(&spec.join(", "));
        spec_flat.push_str(&suffix);
        
        query.push_str(&spec_flat);
    }

    query.push_str(")");
    query

}

pub fn format_insample_query(m: &model::ModelType, run_key: u64, datasetname: &str, it: i32, parsed: Vec<String>) -> String {
    match m {
        model::ModelType::classifier => {
            insample_classifier_query(run_key, datasetname, it, parsed)
        }
        model::ModelType::regressor => {
            insample_regressor_query(run_key, datasetname, it, parsed)
        }
        model::ModelType::other => {
            "Error".to_string()
        }
    }
}

pub fn format_outofsample_query(m: &model::ModelType, run_key: u64, datasetname: &str, it: i32, parsed: Vec<String>) -> String {
    match m {
        model::ModelType::classifier => {
            outofsample_classifier_query(run_key, datasetname, it, parsed)
        }
        model::ModelType::regressor => {
            outofsample_regressor_query(run_key, datasetname, it, parsed)
        }
        model::ModelType::other => {
            "Error".to_string()
        }
    }
}

fn insample_classifier_query(run_key: u64, datasetname: &str, it: i32, mut parsed: Vec<String>) -> String {
    let _imb = parsed.pop();
    let f1 = parsed.pop().unwrap().parse::<f32>().unwrap();
    let rec = parsed.pop().unwrap().parse::<f32>().unwrap();
    let prec = parsed.pop().unwrap().parse::<f32>().unwrap();
    let err = parsed.pop().unwrap().parse::<f32>().unwrap();              
    // At this point, iteration number is known
    format!("INSERT INTO insample (run_key, dataset_name, iteration, err, prcsn, recall, F1) VALUES ({}, \"{}\", {}, {}, {}, {}, {})",
        run_key, datasetname, it, err, prec, rec, f1)
}

fn insample_regressor_query(run_key: u64, datasetname: &str, it: i32, mut parsed: Vec<String>) -> String {
    let rho = parsed.pop().unwrap().parse::<f32>().unwrap();
    let tau = parsed.pop().unwrap().parse::<f32>().unwrap();
    let r2 = parsed.pop().unwrap().parse::<f32>().unwrap();
    let loss = parsed.pop().unwrap().parse::<f32>().unwrap();
    format!("INSERT INTO insample (run_key, dataset_name, iteration, loss, r2, tau, rho) VALUES ({}, \"{}\", {}, {}, {}, {}, {})",
        run_key, datasetname, it, loss, r2, tau, rho)
}

fn outofsample_classifier_query(run_key: u64, datasetname: &str, it: i32, mut parsed: Vec<String>) -> String {
    let _imb = parsed.pop();
    let f1 = parsed.pop().unwrap().parse::<f32>().unwrap();
    let rec = parsed.pop().unwrap().parse::<f32>().unwrap();
    let prec = parsed.pop().unwrap().parse::<f32>().unwrap();
    let err = parsed.pop().unwrap().parse::<f32>().unwrap();              
    // At this point, iteration number is known
    format!("INSERT INTO outofsample (run_key, dataset_name, iteration, err, prcsn, recall, F1) VALUES ({}, \"{}\", {}, {}, {}, {}, {})",
        run_key, datasetname, it, err, prec, rec, f1)
}

fn outofsample_regressor_query(run_key: u64, datasetname: &str, it: i32, mut parsed: Vec<String>) -> String {
    let rho = parsed.pop().unwrap().parse::<f32>().unwrap();
    let tau = parsed.pop().unwrap().parse::<f32>().unwrap();
    let r2 = parsed.pop().unwrap().parse::<f32>().unwrap();
    let loss = parsed.pop().unwrap().parse::<f32>().unwrap();
    format!("INSERT INTO outofsample (run_key, dataset_name, iteration, loss, r2, tau, rho) VALUES ({}, \"{}\", {}, {}, {}, {}, {})",
        run_key, datasetname, it, loss, r2, tau, rho)
}
