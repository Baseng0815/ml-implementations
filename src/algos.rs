use std::ops::Mul;
use rand::prelude::*;

use nalgebra::{SVector, SMatrix, DMatrix, DVector, RowVector, RowDVector};

use crate::util::DataParseError;

#[derive(Debug)]
pub struct LinearRegressionModel {
    pub parameters: DVector<f64>
}

impl LinearRegressionModel {
    pub fn new(dim_features: usize) -> Self {
        LinearRegressionModel { parameters: DVector::zeros(dim_features) }
    }

    pub fn predict(&self, features: &DVector<f64>) -> f64 {
        features.dot(&self.parameters)
    }

    pub fn predict_multiple(&self, features: &DMatrix<f64>) -> DVector<f64> {
        features * &self.parameters
    }

    pub fn loss(&self, dp: &Datapoint) -> f64 {
        let prediction = self.predict(&dp.features);
        (prediction - dp.value).powi(2)
    }

    pub fn loss_multiple(&self, ds: &Dataset) -> f64 {
        let prediction = self.predict_multiple(&ds.features);
        let delta = prediction - &ds.values;
        delta.dot(&delta) / ds.features.nrows() as f64
    }
}

#[derive(Debug)]
pub struct Datapoint {
    pub features: DVector<f64>,
    pub value: f64
}

#[derive(Debug)]
pub struct Dataset {
    pub features: DMatrix<f64>,
    pub values: DVector<f64>
}

impl TryFrom<&[Datapoint]> for Dataset {
    type Error = DataParseError;

    fn try_from(value: &[Datapoint]) -> Result<Self, Self::Error> {
        if value.is_empty() {
            Err(DataParseError::new())?
        }

        let features: DMatrix<f64> = DMatrix::from_rows(&value.iter().map(|dp| dp.features.transpose()).collect::<Vec<_>>());
        let values: DVector<f64> = DVector::from_row_iterator(value.len(),  value.iter().map(|dp| dp.value));

        Ok(Dataset { features , values })
    }
}

impl Dataset {
    // scale all features to [a, b]
    pub fn normalize(&mut self, a: f64, b: f64) {
        let r = self.features.nrows();
        let c = self.features.ncols();

        for ci in 0..self.features.ncols() {
            let mut col = self.features.column(ci).clone_owned();
            let min = col.min();
            let max = col.max();
            col -= DVector::repeat(r, min);
            if max != min {
                col *= (b - a) / (max - min);
            }
            col += DVector::repeat(r, a);
            self.features.set_column(ci, &col);
        }
    }

    // scale all features to a normal distribution
    pub fn standardize(&mut self, mean: f64, sd: f64) {

    }
}

impl LinearRegressionModel {
    pub fn lr_bgd(&mut self, ds: &Dataset, alpha: f64, max_iter: u32, epsilon: f64) {
        for _ in 0..max_iter {
            let total_loss = self.loss_multiple(&ds);
            if total_loss <= epsilon {
                break;
            }

            let gradient = (2.0 / ds.features.nrows() as f64) * &ds.features.transpose() * (&ds.features * &self.parameters - &ds.values);
            self.parameters -= alpha * gradient;
        }
    }

    pub fn lr_sgd(&mut self, ds: &Dataset, alpha: f64, batch_ratio: f64, max_iter: u32, epsilon: f64) {
        let batch_size = (ds.features.nrows() as f64 * batch_ratio) as usize;

        for _ in 0..max_iter {
            let mut permutation = (0..ds.features.nrows()).collect::<Vec<_>>();
            permutation.shuffle(&mut thread_rng());

            let mut features_new = DMatrix::zeros(batch_size, ds.features.ncols());
            let mut values_new = DVector::zeros(batch_size);

            for (di, &si) in permutation.iter().take(batch_size).enumerate() {
                features_new.set_row(di, &ds.features.row(si));
                values_new.set_row(di, &ds.values.row(si));
            }

            let ds_batch = Dataset {
                features: features_new,
                values: values_new
            };

            let total_loss = self.loss_multiple(&ds_batch);
            if total_loss <= epsilon {
                break;
            }

            let gradient = (2.0 / ds_batch.features.nrows() as f64) * &ds_batch.features.transpose() * (&ds_batch.features * &self.parameters - &ds_batch.values);
            self.parameters -= alpha * gradient;
        }
    }

    pub fn lr_normaleqn(&mut self, ds: &Dataset) {
        self.parameters = (ds.features.transpose() * &ds.features).pseudo_inverse(0.001).unwrap() * ds.features.transpose() * &ds.values;
    }
}
