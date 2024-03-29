use core::fmt;
use std::{path::Path, fs::File, error::Error};
use std::io::prelude::*;

use plotters::coord::Shift;
use plotters::coord::types::RangedCoordf32;
use plotters::prelude::*;

use crate::util::quantiles;

mod util;
mod algos;

#[derive(Debug)]
pub struct HousingDatapoint {
    crim: f64, // per capita crime rate by town
    zn: f64, // proportion of residential land zoned for lots over 25,000 sq.ft.
    indus: f64, // proportion of non-retail business acres per town
    chas: f64, // Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
    nox: f64, // nitric oxides concentration (parts per 10 million)
    rm: f64, // average number of rooms per dwelling
    age: f64, // proportion of owner-occupied units built prior to 1940
    dis: f64, // weighted distances to ﬁve Boston employment centers
    rad: f64, // index of accessibility to radial highways
    tax: f64, // full-value property-tax rate per $10,000
    ptratio: f64, // pupil-teacher ratio by town 12. B: 1000(Bk−0.63)2 where Bk is the proportion of blacks by town 13. LSTAT: % lower status of the population
    b: f64, // 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
    lstat: f64, // % lower status of the population
    medv: f64, // Median value of owner-occupied homes in $1000s
}

#[derive(Debug)]
struct DataParseError {  }

impl fmt::Display for DataParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Couldn't parse data")
    }
}

impl Error for DataParseError { }

impl DataParseError {
    pub fn new() -> Self {
        DataParseError { }
    }
}

impl HousingDatapoint {
    pub fn from_line(line: &str) -> Result<HousingDatapoint, Box<dyn Error>> {
        let split: Vec<String> = line.split_whitespace().map(|s| s.to_owned()).collect();
        if split.len() != 14 {
            Err(DataParseError::new())?
        }
        Ok(HousingDatapoint {
            crim: split[0].parse()?,
            zn: split[1].parse()?,
            indus: split[2].parse()?,
            chas: split[3].parse()?,
            nox: split[4].parse()?,
            rm: split[5].parse()?,
            age: split[6].parse()?,
            dis: split[7].parse()?,
            rad: split[8].parse()?,
            tax: split[9].parse()?,
            ptratio: split[10].parse()?,
            b: split[11].parse()?,
            lstat: split[12].parse()?,
            medv: split[13].parse()?
        })
    }
}

fn read_data(path: &Path) -> Result<Vec<HousingDatapoint>, Box<dyn Error>> {
    let mut file = File::open(path)?;
    let mut content = String::new();
    file.read_to_string(&mut content)?;
    let data: Result<Vec<_>, _> = content.split("\n")
        .filter(|line| !line.is_empty())
        .map(|line| HousingDatapoint::from_line(&line))
        .collect();

    data
}

fn plot(
    area: &DrawingArea<BitMapBackend<'_>, Shift>,
    data: Vec<(f64, f64)>,
    label: &str,
    style: ShapeStyle
    ) -> Result<(), Box<dyn std::error::Error>> {
    let q0 = quantiles(&data.iter().map(|dp| dp.0).collect::<Vec<_>>()).ok_or(DataParseError::new())?;
    let q1 = quantiles(&data.iter().map(|dp| dp.1).collect::<Vec<_>>()).ok_or(DataParseError::new())?;

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

fn scatter(data: &Vec<HousingDatapoint>) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new("scatter.png", (1080, 720)).into_drawing_area();
    root.fill(&WHITE)?;

    let areas = root.split_evenly((2, 3));

    // (crim, medv)
    plot(&areas[0], data.iter().map(|dp| (dp.crim, dp.medv)).collect::<Vec<_>>(), "Crime - Median Housing Value", GREEN.filled())?;
    // (crim, nox)
    plot(&areas[1], data.iter().map(|dp| (dp.nox, dp.medv)).collect::<Vec<_>>(), "Nitric Oxide Concentration - Median Housing Value", RED.filled())?;
    // (crim, rm)
    plot(&areas[2], data.iter().map(|dp| (dp.rm, dp.medv)).collect::<Vec<_>>(), "Number of Rooms - Median Housing Value", BLUE.filled())?;
    // (crim, dis)
    plot(&areas[3], data.iter().map(|dp| (dp.dis, dp.medv)).collect::<Vec<_>>(), "Distance to Employment Centers - Median Housing Value", YELLOW.filled())?;
    // (crim, tax)
    plot(&areas[4], data.iter().map(|dp| (dp.tax, dp.medv)).collect::<Vec<_>>(), "Property Tax Rate - Median Housing Value", BLACK.filled())?;
    // (crim, age)
    plot(&areas[5], data.iter().map(|dp| (dp.age, dp.medv)).collect::<Vec<_>>(), "Proportion of Units built prior to 1940 - Median Housing Value", MAGENTA.filled())?;

    root.present()?;

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data = read_data(Path::new("./data/housing.csv"))?;
    eprintln!("data = {:#?}", data);

    scatter(&data)?;

    Ok(())
}
