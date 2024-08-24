use core::fmt;
use std::{error::Error, f64::consts::E};

pub struct Quantiles {
    pub min: f64,
    pub pointzerofive: f64,
    pub pointtwofive: f64,
    pub pointfive: f64,
    pub pointsevenfive: f64,
    pub pointninefive: f64,
    pub max: f64,
}

pub struct DistributionProperties {
    pub quantiles: Quantiles,
    mean: f64,
    variance: f64,
    standard_deviation: f64
}

pub fn distribution_properties(data: &Vec<f64>) -> Option<DistributionProperties> {
    if data.is_empty() {
        return None
    }

    let mut sorted = data.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    sorted.dedup();

    let quantiles = Quantiles {
        min: sorted[0],
        pointzerofive: sorted[(sorted.len() as f64 * 0.05) as usize],
        pointtwofive: sorted[(sorted.len() as f64 * 0.25) as usize],
        pointfive: sorted[(sorted.len() as f64 * 0.5) as usize],
        pointsevenfive: sorted[(sorted.len() as f64 * 0.75) as usize],
        pointninefive: sorted[(sorted.len() as f64 * 0.95) as usize],
        max: sorted[sorted.len() - 1]
    };

    let mean = sorted.iter().sum::<f64>() / sorted.len() as f64;
    let variance = sorted.iter().copied().reduce(|acc, e| acc + (e - mean).powi(2)).unwrap();

    Some(DistributionProperties {
        quantiles,
        mean,
        variance,
        standard_deviation: variance.sqrt()
    })
}

#[derive(Debug)]
pub struct DataParseError {  }

impl fmt::Display for DataParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Couldn't parse data")
    }
}

impl Error for DataParseError { }

impl DataParseError {
    pub fn new() -> Self {
        DataParseError { }
    }
}

pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + E.powf(-x))
}

pub fn sigmoid_derivative(x: f64) -> f64 {
    let y = sigmoid(x);
    y * (1.0 - y)
}
