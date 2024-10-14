use std::{f64::consts::E, fs, io::Cursor};

use egui::{emath::RectTransform, pos2, vec2, Color32, Rect, Rounding, Sense};
use image::{open, GenericImageView, ImageFormat, ImageReader};
use nalgebra::{DVector, SVector};
use rand::Rng;
use serde::Serialize;

use crate::{neural_net::{CostFunction, FeedforwardResult, NeuralNetwork}, util::sigmoid_derivative};

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
    let (mut images, labels): (Vec<Vec<f64>>, Vec<u32>) = serde_pickle::from_slice(&bytes, Default::default()).unwrap();
    // for image in images.iter_mut() {
    //     for value in image.iter_mut() {
    //         if *value >= 0.5 {
    //             *value = 1.0;
    //         } else {
    //             *value = 0.0
    //         }
    //     }
    // }
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
    let mut max = v[0];
    let mut maxi = 0;
    for (rowi, row) in v.row_iter().enumerate() {
        if row[0] >= max {
            max = row[0];
            maxi = rowi;
        }
    }

    (max, maxi)
}

fn nan_to_num(x: f64) -> f64 {
    if x.is_infinite() {
        if x.is_sign_negative() {
            f64::MIN
        } else {
            f64::MAX
        }
    } else if x.is_nan() {
        0.0
    } else {
        x
    }
}

fn train_and_save() -> Result<(), ()> {
    // 28x28 pixels => 784 inputs (greyscale), 1 hidden layer with 15 neurons, 10 outputs
    let mut net = NeuralNetwork::new(vec![784, 30, 10])?;
    // let mut net = load_net("./net_pretty.ron");
    let data = load_mnist_data();
    eprintln!("data.training_images.len() = {:#?}", data.training_images.len());
    eprintln!("data.test_images.len() = {:#?}", data.test_images.len());

    let quadratic_cost = CostFunction {
        cost: |a, y| 0.5 * (a - y).magnitude_squared(),
        delta: |z, a, y| (a - y).component_mul(&z.map(sigmoid_derivative)),
    };

    let cross_entropy = CostFunction {
        cost: |a, y| -(1.0 / y.len() as f64) * a.iter().zip(y.iter()).map(|(a, y)| y * a.ln() + (1.0 - y) * (1.0 - a).ln()).sum::<f64>(),
        delta: |z, a, y| a - y,
    };

    // hyperparameters
    let eta = 0.5; // learning rate
    let batch_size = 10;
    let lambda = 0.1; // regularization parameter
    let epochs = data.training_images.len() / batch_size;
    let cost_function = cross_entropy;

    let mut indices = (0..data.training_images.len()).collect::<Vec<_>>();
    for i in 0..300 {
        // train (stochastic gradient descent)
        for i in 0..batch_size {
            let j = rand::thread_rng().gen_range(0..indices.len());
            indices.swap(i, j);
        }

        for epoch in 0..epochs {
            let batch_indices = &indices[(epoch * batch_size)..((epoch + 1) * batch_size)];
            let feedforward = net.feedforward(&data.training_images, batch_indices);
            net.backpropagate(&feedforward, &data.training_labels, batch_indices, eta, lambda, &cost_function);
            // eprintln!("   Batch iteration {}/{}...", batch_iteration, batch_iterations);
        }

        // evaluate
        let feedforward = net.feedforward(&data.test_images, &(0..data.test_images.len()).collect::<Vec<_>>());
        let mut cost = 0.0;
        let mut correctly_classified = 0;
        for (label, ff) in data.test_labels.iter().zip(feedforward.iter()) {
            let (_, correct_label) = vec_max_with_index(label.clone());
            let result = ff.final_result();
            let (_, predicted_label) = vec_max_with_index(result.clone());

            if correct_label == predicted_label {
                correctly_classified += 1;
            }

            cost += (cost_function.cost)(result.clone(), label.clone());
        }
        cost = cost / data.test_labels.len() as f64;
        eprintln!("Iteration {}: cost={}; correctly classified: {}/{}", i, cost, correctly_classified, data.test_labels.len());
    }

    let net_ron = ron::ser::to_string_pretty(&net, ron::ser::PrettyConfig::default()).unwrap();
    fs::write("./net_pretty.ron", &net_ron).unwrap();

    Ok(())
}

enum DragMode {
    Fill,
    Erase,
    Nothing,
}

struct FrontendApp {
    pixels_w: usize,
    pixels_h: usize,
    mode: DragMode, // this could probably be done using egui drag function but w/e
    pixels: DVector<f64>,

    net: NeuralNetwork,
    last_ff: FeedforwardResult,
}

impl FrontendApp {
    pub fn new(cc: &eframe::CreationContext<'_>, pixels_w: usize, pixels_h: usize) -> Self {
        let data = load_mnist_data();
        let img = [load_image("./data/eight.png")];

        Self {
            pixels_w,
            pixels_h,
            mode: DragMode::Fill,
            // pixels: DVector::repeat(pixels_w * pixels_h, 0.0),
            pixels: data.test_images[1].clone(),
            net: load_net("./net_pretty.ron"),
            last_ff: FeedforwardResult::default(),
        }
    }
}

impl eframe::App for FrontendApp {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        egui::SidePanel::right("side_panel").show(ctx, |ui| {
            ui.heading("Controls");
            if ui.button("Reset").clicked() {
                self.pixels = DVector::repeat(self.pixels.nrows(), 0.0);
            }

            ui.separator();

            ui.heading("Result");

            if let Some(last_layer) = self.last_ff.layers.last() {
                ui.label(format!("Feedforward result: {:#?}", last_layer.activation.data));
            }
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            let (response, painter) = ui.allocate_painter(ui.available_size(), Sense::drag());
            let to_screen = RectTransform::from_to(Rect::from_min_size(pos2(0.0, 0.0), vec2(self.pixels_w as f32, self.pixels_h as f32)), response.rect);

            if response.drag_stopped() {
                let ff = self.net.feedforward(&[self.pixels.clone()], &[0]);
                self.last_ff = ff.last().expect("A feedforward must have a result").clone();

                self.mode = DragMode::Nothing;
            } else if response.dragged() {
                if let Some(cursor_pos) = response.hover_pos() {
                    let pixel_coords = to_screen.inverse().transform_pos(cursor_pos);
                    let x = pixel_coords.x as usize;
                    let y = pixel_coords.y as usize;
                    let index = y as usize * self.pixels_w + x;
                    if index < self.pixels.nrows() {
                        if matches!(self.mode, DragMode::Nothing) {
                            if self.pixels[index] > 0.5 {
                                self.mode = DragMode::Erase;
                            } else {
                                self.mode = DragMode::Fill;
                            }

                        }

                        match self.mode {
                            DragMode::Fill => self.pixels[index] = 1.0,
                            DragMode::Erase => self.pixels[index] = 0.0,
                            _ => {  },
                        }
                    }
                }
            }

            for y in 0..self.pixels_h {
                for x in 0..self.pixels_w {
                    let rect = Rect::from_min_size(pos2(x as f32, y as f32), vec2(1.0, 1.0));
                    let rect_screencoords = to_screen.transform_rect(rect);
                    let pixel_value = self.pixels[y * self.pixels_w + x];
                    let color = if pixel_value < 0.5 { Color32::BLACK } else { Color32::WHITE };
                    painter.rect_filled(rect_screencoords, Rounding::ZERO, color);
                }
            }

            ui.heading("Hello World!");
        });
    }
}

pub fn run_sample() -> Result<(), ()> {
    train_and_save()?;

    // let native_options = eframe::NativeOptions::default();
    // eframe::run_native("Digit recognition", native_options, Box::new(|cc| Ok(Box::new(FrontendApp::new(cc, 28, 28))))).unwrap();

    // let net = load_net("./net_pretty.ron");
    // let img = [load_image("./data/eight.png")];
    // let ff = net.feedforward(&img);
    // eprintln!("ff = {:#?}", ff);
    // let (_, predicted_label) = vec_max_with_index(ff[0].final_result().clone());
    // eprintln!("predicted_label = {:#?}", predicted_label);

    Ok(())
}
