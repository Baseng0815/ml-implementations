use std::f64::consts::E;

use nalgebra::{DMatrix, DVector, RowDVector};
use rand::{distributions::Uniform, Rng};
use rand_distr::Normal;
use serde::{Deserialize, Serialize};

use crate::{
    util::{sigmoid, sigmoid_derivative},
};

#[derive(Debug, Deserialize, Serialize)]
struct Layer {
    weights: DMatrix<f64>,
    bias: DVector<f64>,
}

impl Layer {
    pub fn new(weights: DMatrix<f64>, bias: DVector<f64>) -> Result<Self, ()> {
        if weights.nrows() != bias.nrows() {
            Err(())?;
        }

        Ok(Self { weights, bias })
    }
}

#[derive(Debug, Clone)]
pub struct FeedforwardLayer {
    pub weighted_inputs: DVector<f64>,
    pub activation: DVector<f64>,
}

impl FeedforwardLayer {
    fn new(weighted_inputs: DVector<f64>, activation: DVector<f64>) -> Self {
        Self {
            weighted_inputs,
            activation,
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct FeedforwardResult {
    pub layers: Vec<FeedforwardLayer>,
}

impl FeedforwardResult {
    pub fn new(layers: Vec<FeedforwardLayer>) -> Self {
        Self { layers }
    }

    pub fn final_result(&self) -> &DVector<f64> {
        &self
            .layers
            .last()
            .expect("There has to be an output layer")
            .activation
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct NeuralNetwork {
    layers: Vec<Layer>,
}

impl NeuralNetwork {
    pub fn new(layer_sizes: Vec<usize>) -> Result<Self, ()> {
        // weights and biases are sampled uniformly from [0,1]
        let dist = Normal::new(0.0, 1.0).unwrap();
        // let dist = Uniform::new(0.0, 1.0);
        let layers: Result<Vec<_>, _> = layer_sizes
            .iter()
            .zip(layer_sizes.iter().skip(1))
            .map(|(&prevsize, &size)| {
                let weight_rows = (0..size)
                    .map(|_| {
                        RowDVector::from_iterator(
                            prevsize,
                            (0..prevsize).map(|_| rand::thread_rng().sample(dist)),
                        )
                    })
                    .collect::<Vec<_>>();
                let weights = DMatrix::from_rows(&weight_rows);
                let bias = DVector::from_iterator(
                    size,
                    (0..size).map(|_| rand::thread_rng().sample(dist)),
                );
                Layer::new(weights, bias)
            })
            .collect();

        Ok(Self { layers: layers? })
    }

    /// Calculate an output value for an input value
    pub fn feedforward(&self, inputs: &[DVector<f64>], indices: &[usize]) -> Vec<FeedforwardResult> {
        indices.iter().map(|index| &inputs[*index]).map(|input| {
            let mut layers = vec![FeedforwardLayer::new(
                DVector::zeros(input.ncols()),
                input.clone(),
            )];
            for layer in &self.layers {
                let prev = &layers.last().unwrap().activation;
                let weighted_inputs = layer.weights.clone() * prev + layer.bias.clone();
                let activation = weighted_inputs.map(sigmoid);
                layers.push(FeedforwardLayer::new(weighted_inputs, activation));
            }

            FeedforwardResult::new(layers)
        }).collect::<Vec<_>>()
    }

    /// Calculate the gradient of the cost function for an input value and adjust parameters
    pub fn backpropagate(
        &mut self,
        feedforward_multi: &[FeedforwardResult],
        output_multi: &[DVector<f64>],
        indices: &[usize],
        eta: f64,
    ) {
        let errors_multi: Vec<_> = feedforward_multi.iter()
            .zip(indices.iter().map(|index| &output_multi[*index]))
            .map(|(ff, output)| {
                // calculate output error
                let last = ff.layers.last().expect("There has to be an output");
                let (a_last, z_last) = (last.activation.clone(), last.weighted_inputs.clone());
                let z_last_derivative = z_last.map(sigmoid_derivative);
                let error_last = (a_last - output).component_mul(&z_last_derivative);

                // backpropagate the error through all layers
                let mut errors = vec![error_last];

                for ffi in (1..(ff.layers.len() - 1)).rev() {
                    let weighted_inputs = &ff.layers[ffi].weighted_inputs;
                    let weights_next = &self.layers[ffi].weights;

                    let error_prev = errors.last().expect("There has to be a previous error");
                    let z_derivative = weighted_inputs.map(sigmoid_derivative);
                    let error =
                        (weights_next.transpose() * error_prev).component_mul(&z_derivative);
                    errors.push(error);
                }

                errors.into_iter().rev().collect::<Vec<_>>()
            })
            .collect();

        for (layeri, layer) in self.layers.iter_mut().enumerate() {
            let (sum_gradient_weights, sum_gradient_bias) =
                feedforward_multi.iter().zip(errors_multi.iter()).fold(
                    (
                        DMatrix::zeros(layer.weights.nrows(), layer.weights.ncols()),
                        DVector::zeros(layer.bias.nrows()),
                    ),
                    |acc, (ff, errors)| {
                        // calculate gradient and adjust weights and biases for each layer
                        let activation_pre = ff.layers[layeri].activation.clone();
                        // eprintln!("activation_pre.nrows() = {:#?}", activation_pre.nrows());
                        // eprintln!("errors[layeri].nrows() = {:#?}", errors[layeri].nrows());
                        // eprintln!("errors[layeri].ncols() = {:#?}", errors[layeri].ncols());
                        // eprintln!("acc.0.nrows() = {:#?}", acc.0.nrows());
                        // eprintln!("acc.0.ncols() = {:#?}", acc.0.ncols());
                        // eprintln!("acc.1.nrows() = {:#?}", acc.1.nrows());
                        // eprintln!("acc.1.ncols() = {:#?}", acc.1.ncols());
                        (
                            acc.0 + errors[layeri].clone() * activation_pre.transpose(),
                            acc.1 + errors[layeri].clone(),
                        )
                    },
                );

            // eprintln!(
            //     "sum_gradient_weights.nrows() = {:#?}",
            //     sum_gradient_weights.nrows()
            // );
            // eprintln!(
            //     "sum_gradient_bias.nrows() = {:#?}",
            //     sum_gradient_bias.nrows()
            // );
            // eprintln!("layer.weights.nrows() = {:#?}", layer.weights.nrows());
            // eprintln!("layer.weights.ncols() = {:#?}", layer.weights.ncols());
            // eprintln!("layer.bias.nrows() = {:#?}", layer.bias.nrows());
            layer.weights -= eta / (output_multi.len() as f64) * sum_gradient_weights;
            layer.bias -= eta / (output_multi.len() as f64) * sum_gradient_bias;
        }
    }
}
