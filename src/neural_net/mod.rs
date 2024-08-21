use nalgebra::{DMatrix, DVector, RowDVector};
use rand::{distributions::Uniform, Rng};

#[derive(Debug)]
struct Layer {
    weights: DMatrix<f64>,
    bias: DVector<f64>,
}

impl Layer {
    pub fn new(weights: DMatrix<f64>, bias: DVector<f64>) -> Result<Self, ()> {
        if weights.nrows() != bias.nrows() {
            Err(())?;
        }

        Ok(Self {
            weights,
            bias
        })
    }
}

#[derive(Debug)]
pub struct NeuralNetwork {
    layers: Vec<Layer>,
}

impl NeuralNetwork {
    pub fn new(layer_sizes: Vec<usize>) -> Result<Self, ()> {
        // weights and biases are sampled uniformly from [0,1]
        let dist = Uniform::new(0.0, 1.0);
        let layers: Result<Vec<_>, _> = layer_sizes.iter().zip(layer_sizes.iter().skip(1)).map(|(&prevsize, &size)| {
            let weight_rows = (0..size).map(|_| {
                RowDVector::from_iterator(prevsize, (0..prevsize).map(|_| rand::thread_rng().sample(dist)))
            }).collect::<Vec<_>>();
            let weights = DMatrix::from_rows(&weight_rows);
            let bias = DVector::from_iterator(size, (0..size).map(|_| rand::thread_rng().sample(dist)));
            Layer::new(weights, bias)
        }).collect();

        Ok(Self {
            layers: layers?
        })
    }

    /// Calculate an output value for an input value
    pub fn feedforward(&self, input: DVector<f64>, activation: fn(f64) -> f64) -> DVector<f64> {
        let mut result = input;
        for layer in &self.layers {
            result = layer.weights.clone() * result + layer.bias.clone();
            for e in result.iter_mut() {
                *e = activation(*e);
            }
        }
        result
    }

    /// Calculate the gradient of the cost function for an input value
    pub fn backpropagate(&self, input: DVector<f64>) {

    }
}
