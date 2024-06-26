use std::path::Path;

use data::HousingDatapoint;
use plotters::coord::Shift;
use plotters::prelude::*;
use util::{DataParseError, distribution_properties};

use crate::algos::{Dataset, Datapoint, LinearRegressionModel};
use crate::data::read_housing_data;

mod util;
mod algos;
mod data;

fn plot_housing_feature(
    area: &DrawingArea<BitMapBackend<'_>, Shift>,
    data: Vec<(f64, f64)>,
    label: &str,
    style: ShapeStyle
    ) -> Result<(), Box<dyn std::error::Error>> {
    let q0 = distribution_properties(&data.iter().map(|dp| dp.0).collect::<Vec<_>>()).ok_or(DataParseError::new())?.quantiles;
    let q1 = distribution_properties(&data.iter().map(|dp| dp.1).collect::<Vec<_>>()).ok_or(DataParseError::new())?.quantiles;

    let mut chart = ChartBuilder::on(&area)
        .caption(label, ("sans-serif", 15).into_font())
        .margin(3)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0f32..q0.pointninefive as f32, 0f32..q1.pointninefive as f32)?;

    chart.configure_mesh().draw()?;

    chart.draw_series(
        data
        .iter()
        .filter(|dp| dp.0 > q0.pointzerofive && dp.0 < q0.pointninefive)
        .filter(|dp| dp.1 > q1.pointzerofive && dp.1 < q1.pointninefive)
        .map(|dp| Circle::new((dp.0 as f32, dp.1 as f32), 2, style))
        )?;

    Ok(())
}

fn eval_housing_features(data: &Vec<HousingDatapoint>) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new("housing_features.jpg", (1080, 720)).into_drawing_area();
    root.fill(&WHITE)?;

    let areas = root.split_evenly((2, 3));

    // (crim, medv)
    plot_housing_feature(&areas[0], data.iter().map(|dp| (dp.crim, dp.medv)).collect::<Vec<_>>(), "Crime - Median Housing Value", GREEN.filled())?;
    // (crim, nox)
    plot_housing_feature(&areas[1], data.iter().map(|dp| (dp.nox, dp.medv)).collect::<Vec<_>>(), "Nitric Oxide Concentration - Median Housing Value", RED.filled())?;
    // (crim, rm)
    plot_housing_feature(&areas[2], data.iter().map(|dp| (dp.rm, dp.medv)).collect::<Vec<_>>(), "Number of Rooms - Median Housing Value", BLUE.filled())?;
    // (crim, dis)
    plot_housing_feature(&areas[3], data.iter().map(|dp| (dp.dis, dp.medv)).collect::<Vec<_>>(), "Distance to Employment Centers - Median Housing Value", YELLOW.filled())?;
    // (crim, tax)
    plot_housing_feature(&areas[4], data.iter().map(|dp| (dp.tax, dp.medv)).collect::<Vec<_>>(), "Property Tax Rate - Median Housing Value", BLACK.filled())?;
    // (crim, age)
    plot_housing_feature(&areas[5], data.iter().map(|dp| (dp.age, dp.medv)).collect::<Vec<_>>(), "Proportion of Units built prior to 1940 - Median Housing Value", MAGENTA.filled())?;

    root.present()?;

    Ok(())
}

fn eval_regression_loss(housing_dataset: &Dataset) -> Result<(), Box<dyn std::error::Error>> {
    // batch gradient descent
    let rates = vec![0.2, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001];
    let mut history: Vec<Vec<f64>> = Vec::new();
    let mut models = Vec::with_capacity(rates.len());
    for _ in 0..rates.len() {
        history.push(Vec::new());
        models.push(LinearRegressionModel::new(housing_dataset.features.ncols()));
    }

    for i in 0..1000 {
        eprintln!("i = {:#?}", i);
        for j in 0..rates.len() {
            models[j].lr_bgd(&housing_dataset, rates[j], 1, 0.001);
            history[j].push(models[j].loss_multiple(&housing_dataset));
        }
    }

    let root = BitMapBackend::new("regression_loss_bgd.jpg", (1080, 720)).into_drawing_area();
    root.fill(&WHITE)?;

    let max = history.iter().flatten().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    let mut chart = ChartBuilder::on(&root)
        .caption("Linear Regression with Batch Gradient Descent Loss Rate by Alpha", ("sans-serif", 20).into_font())
        .margin(3)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0f32..history.iter().map(|x| x.len()).max().unwrap() as f32, 0f32..(max as f32 * 1.05))?;

    let colors = vec![GREEN.filled(), RED.filled(), BLUE.filled(), YELLOW.filled(), BLACK.filled(), MAGENTA.filled(), CYAN.filled()];
    for i in 0..rates.len() {
        let color = colors[i].clone();
        chart.draw_series(LineSeries::new(
            history[i]
            .iter()
            .enumerate()
            .map(|dp| (dp.0 as f32, *dp.1 as f32)), color.stroke_width(1)))?
            .label(format!("alpha={}", rates[i]))
            .legend(move |(x, y)| Rectangle::new([(x, y - 5), (x + 10, y + 5)], color.filled()));
    }

    chart.configure_mesh().x_desc("Training Epoch").y_desc("MSE").draw()?;
    chart.configure_series_labels().border_style(BLACK).draw()?;
    root.present()?;

    // stochastic gradient descent with different batch sizes
    let rates = [0.005, 0.1, 0.5];
    history = Vec::new();
    models = Vec::with_capacity(rates.len());
    for _ in 0..rates.len() {
        history.push(Vec::new());
        models.push(LinearRegressionModel::new(housing_dataset.features.ncols()));
    }
    for i in 0..1000 {
        eprintln!("i = {:#?}", i);
        for j in 0..rates.len() {
            models[j].lr_sgd(&housing_dataset, 0.005, rates[j], 1, 0.001);
            history[j].push(models[j].loss_multiple(&housing_dataset));
        }
    }

    let root = BitMapBackend::new("regression_loss_sgd.jpg", (1080, 720)).into_drawing_area();
    root.fill(&WHITE)?;

    let max = history.iter().flatten().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    let mut chart = ChartBuilder::on(&root)
        .caption("Linear Regression with Stochastic Gradient Descent Loss Rate by Batch Size (alpha=0.005)", ("sans-serif", 20).into_font())
        .margin(3)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0f32..history.iter().map(|x| x.len()).max().unwrap() as f32, 0f32..(max as f32 * 1.05))?;

    let colors = vec![GREEN.filled(), RED.filled(), BLUE.filled(), YELLOW.filled(), BLACK.filled(), MAGENTA.filled(), CYAN.filled()];
    for i in 0..rates.len() {
        let color = colors[i].clone();
        chart.draw_series(LineSeries::new(
            history[i]
            .iter()
            .enumerate()
            .map(|dp| (dp.0 as f32, *dp.1 as f32)), color.stroke_width(1)))?
            .label(format!("batch size={}", rates[i]))
            .legend(move |(x, y)| Rectangle::new([(x, y - 5), (x + 10, y + 5)], color.filled()));
    }

    chart.configure_mesh().x_desc("Training Epoch").y_desc("MSE").draw()?;
    chart.configure_series_labels().border_style(BLACK).draw()?;
    root.present()?;

    // best solution through closed-formed normal equation
    let mut best_model = LinearRegressionModel::new(housing_dataset.features.ncols());
    best_model.lr_normaleqn(&housing_dataset);
    eprintln!("best_model.loss_multiple(&housing_dataset) = {:#?}", best_model.loss_multiple(&housing_dataset));

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let housing_data = read_housing_data(Path::new("./data/housing.csv"))?;
    eprintln!("housing_data = {:?}", housing_data);

    let mut housing_dataset = Dataset::try_from(&housing_data.iter().map(|x| Datapoint::from(x)).collect::<Vec<_>>()[..])?;
    eprintln!("housing_dataset = {:#?}", housing_dataset);
    housing_dataset.normalize(0.0, 1.0);
    eprintln!("housing_dataset = {:#?}", housing_dataset);

    eval_housing_features(&housing_data)?;
    eval_regression_loss(&housing_dataset)?;

    Ok(())
}
