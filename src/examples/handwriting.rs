use std::{f64::consts::E, fs, io::Cursor};

use image::{open, GenericImageView, ImageFormat, ImageReader};
use nalgebra::{DVector, SVector};
use serde::Serialize;

use crate::neural_net::NeuralNetwork;

#[derive(Debug)]
pub struct MNISTDataset {
    pub training_images: Vec<DVector<f64>>,
    pub training_labels: Vec<DVector<f64>>,
    pub test_images: Vec<DVector<f64>>,
    pub test_labels: Vec<DVector<f64>>,
}

impl MNISTDataset {
    pub fn new(images: Vec<Vec<f64>>, labels: Vec<u32>, split: f64) -> Self {
        // This is extremely inefficient. Too bad!
        let images = images.into_iter().map(|pixels| DVector::from_column_slice(&pixels)).collect::<Vec<_>>();
        let labels = labels.into_iter().map(|label| {
            let mut vec: DVector<f64> = DVector::zeros(10);
            vec[label as usize] = 1.0;
            vec
        }).collect::<Vec<_>>();
        let cutoff = (images.len() as f64 * split) as usize;
        Self {
            training_images: images[..cutoff].to_vec(),
            training_labels: labels[..cutoff].to_vec(),
            test_images: images[cutoff..].to_vec(),
            test_labels: labels[cutoff..].to_vec(),
        }
    }
}

fn load_mnist_data() -> MNISTDataset {
    let bytes = fs::read("./data/mnist").unwrap();
    let (images, labels): (Vec<Vec<f64>>, Vec<u32>) = serde_pickle::from_slice(&bytes, Default::default()).unwrap();
    MNISTDataset::new(images, labels, 0.9)
}

fn load_image(path: &str) -> DVector<f64> {
    let img = open(path).unwrap().into_luma8();
    let pixels: Vec<_> = img.pixels().map(|pix| {
        pix.0[0] as f64 / 255.0
    }).collect();

    DVector::from_vec(pixels)
}

fn load_net(path: &str) -> NeuralNetwork {
    let net: NeuralNetwork = ron::from_str(&fs::read_to_string(path).unwrap()).unwrap();
    net
}

fn vec_max_with_index(v: DVector<f64>) -> (f64, usize) {
    let mut max = 0.0;
    let mut maxi = 0;
    for (rowi, row) in v.row_iter().enumerate() {
        if row[0] >= max {
            max = row[0];
            maxi = rowi;
        }
    }

    (max, maxi)
}

pub fn run_sample() -> Result<(), ()> {
    // // 28x28 pixels => 784 inputs (greyscale), 1 hidden layer with 15 neurons, 10 outputs
    // // let mut net = NeuralNetwork::new(vec![784, 15, 10])?;
    // let mut net: NeuralNetwork = ron::from_str(&fs::read_to_string("./net_pretty.ron").unwrap()).unwrap();
    // // eprintln!("net = {:#?}", net);

    // let data = load_mnist_data();

    // for i in 0..5000 {
    //     // train
    //     let feedforward = net.feedforward(&data.training_images);
    //     net.backpropagate(&feedforward, &data.training_labels);

    //     // evaluate
    //     let feedforward = net.feedforward(&data.test_images);
    //     let mut mse = 0.0;
    //     let mut correctly_classified = 0;
    //     for (label, ff) in data.test_labels.iter().zip(feedforward.iter()) {
    //         let (_, correct_label) = vec_max_with_index(label.clone());
    //         let result = ff.final_result();
    //         let (_, predicted_label) = vec_max_with_index(result.clone());

    //         if correct_label == predicted_label {
    //             correctly_classified += 1;
    //         }

    //         mse += (label - result).magnitude_squared();
    //     }
    //     mse = mse / data.test_labels.len() as f64;
    //     eprintln!("Iteration {}: MSE={}; correctly classified: {}/{}", i, mse, correctly_classified, data.test_labels.len());
    // }

    // let net_ron = ron::ser::to_string_pretty(&net, ron::ser::PrettyConfig::default()).unwrap();
    // fs::write("./net_pretty.ron", &net_ron).unwrap();

    let net = load_net("./net_pretty.ron");
    let img = [load_image("./data/zero.png")];
    let ff = net.feedforward(&img);
    eprintln!("ff = {:#?}", ff);
    let (_, predicted_label) = vec_max_with_index(ff[0].final_result().clone());
    eprintln!("predicted_label = {:#?}", predicted_label);

    Ok(())
}
