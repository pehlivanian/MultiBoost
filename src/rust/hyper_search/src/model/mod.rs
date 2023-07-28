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