use std::{f64::consts::E, fs, io::Cursor};

use egui::{emath::RectTransform, pos2, vec2, Color32, Rect, Rounding, Sense};
use image::{open, GenericImageView, ImageFormat, ImageReader};
use nalgebra::{DVector, SVector};
use serde::Serialize;

use crate::neural_net::{FeedforwardResult, NeuralNetwork};

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
    for image in images.iter_mut() {
        for value in image.iter_mut() {
            if *value >= 0.5 {
                *value = 1.0;
            } else {
                *value = 0.0
            }
        }
    }
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

fn train_and_save() -> Result<(), ()> {
    // 28x28 pixels => 784 inputs (greyscale), 1 hidden layer with 15 neurons, 10 outputs
    // let mut net = NeuralNetwork::new(vec![784, 15, 10])?;
    let mut net = load_net("./net_pretty.ron");
    let data = load_mnist_data();

    for i in 0..5000 {
        // train
        let feedforward = net.feedforward(&data.training_images);
        net.backpropagate(&feedforward, &data.training_labels);

        // evaluate
        let feedforward = net.feedforward(&data.test_images);
        let mut mse = 0.0;
        let mut correctly_classified = 0;
        for (label, ff) in data.test_labels.iter().zip(feedforward.iter()) {
            let (_, correct_label) = vec_max_with_index(label.clone());
            let result = ff.final_result();
            let (_, predicted_label) = vec_max_with_index(result.clone());

            if correct_label == predicted_label {
                correctly_classified += 1;
            }

            mse += (label - result).magnitude_squared();
        }
        mse = mse / data.test_labels.len() as f64;
        eprintln!("Iteration {}: MSE={}; correctly classified: {}/{}", i, mse, correctly_classified, data.test_labels.len());
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
                let ff = self.net.feedforward(&[self.pixels.clone()]);
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
    train_and_save();

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
