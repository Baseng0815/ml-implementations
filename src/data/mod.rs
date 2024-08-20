use nalgebra::{DMatrix, DVector};

use crate::util::DataParseError;

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

