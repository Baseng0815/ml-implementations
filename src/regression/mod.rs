use nalgebra::{DMatrix, DVector};
use rand::prelude::*;

use crate::data::{Datapoint, Dataset};

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
