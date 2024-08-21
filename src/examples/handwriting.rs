use std::f64::consts::E;

use nalgebra::{DVector, SVector};

use crate::neural_net::NeuralNetwork;

pub fn run_sample() -> Result<(), ()> {
    // 28x28 pixels => 784 inputs (greyscale), 1 hidden layer with 15 neurons, 10 outputs
    // let net = NeuralNetwork::new(vec![784, 15, 10]);
    let net = NeuralNetwork::new(vec![1, 2, 3])?;
    let input = DVector::from_vec(vec![5.0]);
    let activation = |x: f64| {
        1.0 / (1.0 + E.powf(-x))
    };

    eprintln!("net = {:#?}", net);
    eprintln!("net.feedforward(input) = {:#?}", net.feedforward(input, activation));

    Ok(())
}
