use std::fs::File;
use ndarray::{Array, Array1, Array2};
use linfa::Dataset;
use std::io::{BufRead, BufReader};

pub struct ClassificationDataset {
    dataset: Dataset<f32, i32, ndarray::Dim<[usize; 1]>>,
}

#[allow(non_camel_case_types)]
enum DataType {
    X,
    y,
}

impl ClassificationDataset {

   pub fn new(path: &str) -> Self {
        let dataset = ClassificationDataset::get_dataset(path);
	ClassificationDataset{dataset: dataset}        
    }

   pub fn shape(&self) -> (usize, usize) {
       (self.dataset.records.nrows(), self.dataset.records.ncols())
   }

    fn suffix_path(dt: DataType) -> &'static str {
        match dt {
            DataType::X => "_X.csv",
            DataType::y => "_y.csv",
        }
    }

    fn get_path(dataset_name: &str, dtype: DataType) -> String {
        let mut full_path: String = String::from("/home/charles/Data/");
        let suffix = ClassificationDataset::suffix_path(dtype);
        full_path.push_str(dataset_name);
        full_path.push_str(suffix);

        full_path
    }

    fn get_Xdata(path: &str) -> Array2<f32> {
        let mut reader = csv::ReaderBuilder::new()
            .has_headers(false)
            .from_path(path).unwrap();

        let data: Vec<Vec<f32>> = reader
            .records()
            .map(|r|
            r
            .unwrap().iter()
            .map(|field| field.parse::<f32>().unwrap())
            .collect::<Vec<f32>>()
            ).collect::<Vec<Vec<f32>>>();

        let m = data.len();
        let n = data[0].len();

        let mut records: Vec<f32> = vec![];
        for record in data.iter() {
            records.extend_from_slice(record);
        }

        Array::from(records).into_shape((m,n)).unwrap()
    }

    fn get_ydata(path: &str) -> Array1<i32> {
        let file = File::open(path).unwrap();
        let br = BufReader::new(file);
        let mut r = Vec::new();
    
        for line in br.lines() {
            let line = line.unwrap();
            let n = line
                .trim()
                .parse::<f32>().unwrap() as i32;
            r.push(n);
        }
        Array::from(r)
    }

    fn get_dataset(dataset_name: &str) -> Dataset<f32, i32, ndarray::Dim<[usize; 1]>> {
        let Xpath = ClassificationDataset::get_path(dataset_name, DataType::X);
        let ypath = ClassificationDataset::get_path(dataset_name, DataType::y);

        let X = ClassificationDataset::get_Xdata(&Xpath);
        let y = ClassificationDataset::get_ydata(&ypath);

        Dataset::new(X, y)
    }
}
