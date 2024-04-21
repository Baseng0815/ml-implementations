use std::{path::Path, error::Error, fs::File, io::Read};

use nalgebra::{DVector, SVector};

use crate::{util::DataParseError, algos::Datapoint};

#[derive(Debug)]
pub struct HousingDatapoint {
    pub crim: f64, // per capita crime rate by town
    pub zn: f64, // proportion of residential land zoned for lots over 25,000 sq.ft.
    pub indus: f64, // proportion of non-retail business acres per town
    pub chas: f64, // Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
    pub nox: f64, // nitric oxides concentration (parts per 10 million)
    pub rm: f64, // average number of rooms per dwelling
    pub age: f64, // proportion of owner-occupied units built prior to 1940
    pub dis: f64, // weighted distances to ﬁve Boston employment centers
    pub rad: f64, // index of accessibility to radial highways
    pub tax: f64, // full-value property-tax rate per $10,000
    pub ptratio: f64, // pupil-teacher ratio by town 12. B: 1000(Bk−0.63)2 where Bk is the proportion of blacks by town 13. LSTAT: % lower status of the population
    pub b: f64, // 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
    pub lstat: f64, // % lower status of the population
    pub medv: f64, // Median value of owner-occupied homes in $1000s
}

#[derive(Debug, Clone)]
pub struct MailDatapoint {
    pub index: String,
    pub word_count: Vec<usize>,
    pub is_spam: bool
}

impl HousingDatapoint {
    pub fn from_line(line: &str) -> Result<HousingDatapoint, Box<dyn Error>> {
        let split: Vec<String> = line.split_whitespace().map(|s| s.to_owned()).collect();
        if split.len() != 14 {
            Err(DataParseError::new())?
        }
        Ok(HousingDatapoint {
            crim: split[0].parse()?,
            zn: split[1].parse()?,
            indus: split[2].parse()?,
            chas: split[3].parse()?,
            nox: split[4].parse()?,
            rm: split[5].parse()?,
            age: split[6].parse()?,
            dis: split[7].parse()?,
            rad: split[8].parse()?,
            tax: split[9].parse()?,
            ptratio: split[10].parse()?,
            b: split[11].parse()?,
            lstat: split[12].parse()?,
            medv: split[13].parse()?
        })
    }
}

impl MailDatapoint {
    pub fn from_line(line: &str) -> Result<MailDatapoint, Box<dyn Error>> {
        let split: Vec<String> = line.split(",").map(|s| s.to_owned()).collect();
        if split.len() != 3002 {
            Err(DataParseError::new())?
        }
        let mut word_count: Vec<usize> = Vec::new();
        for str in split.iter().skip(1).take(3000) {
            word_count.push(str.parse()?);
        }
        Ok(MailDatapoint {
            index: split[0].clone(),
            word_count,
            is_spam: split[3001] == "0"
        })
    }
}

impl From<&HousingDatapoint> for Datapoint {
    fn from(value: &HousingDatapoint) -> Self {
        let features = DVector::from_row_slice(&[
            1.0, value.crim, value.zn, value.indus, value.chas, value.nox, value.rm,
            value.age, value.dis, value.rad, value.tax, value.ptratio, value.b, value.lstat ]);

        Datapoint {
            features,
            value: value.medv
        }
    }
}

pub fn read_housing_data(path: &Path) -> Result<Vec<HousingDatapoint>, Box<dyn Error>> {
    let mut file = File::open(path)?;
    let mut content = String::new();
    file.read_to_string(&mut content)?;
    let data: Result<Vec<_>, _> = content.split("\n")
        .filter(|line| !line.is_empty())
        .map(|line| HousingDatapoint::from_line(&line))
        .collect();

    data
}
