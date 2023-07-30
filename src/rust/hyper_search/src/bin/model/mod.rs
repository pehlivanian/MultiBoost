use lazy_static;
use regex::Regex;

lazy_static!{
    pub static ref ITER: Regex = Regex::new(r"[\s]+ITER[\s]*:[\s]+([0-9]+)").unwrap();
}

lazy_static!{
    pub static ref FOLDER: Regex = Regex::new(r"[\s]+FOLDER[\s]*:[\s]+(.+)").unwrap();
}

lazy_static!{
    pub static ref INDEX: Regex = Regex::new(r"[\s]+INDEX[\s]*:[\s]+(.+)").unwrap();
}

lazy_static!{
    pub static ref IS: Regex = Regex::new(r"[\s]IS[\s]*:[\s]*.*:[\s]+\((.*)\)").unwrap();
}

lazy_static!{
    pub static ref OOS: Regex = Regex::new(r"[\s]OOS[\s]*:[\s]*.*:[\s]+\((.*)\)").unwrap();
}


pub enum ModelType {
    classifier,
    regressor,
    other,
}

impl ModelType {
    pub fn model_cmd(m: &ModelType) -> &'static str {
        match m {
            ModelType::classifier => "incremental_classifier_fit.sh",
            ModelType::regressor => "incremental_regressor_fit.sh",
            ModelType::other => "Error!",
        }
    }

    pub fn database_name(m: &ModelType) -> &'static str {
        match m {
            ModelType::classifier => "MULTISCALEGB_CLASS",
            ModelType::regressor => "MULTISCALEGB_REG",
            ModelType::other => "ERROR",
        }
    }
}