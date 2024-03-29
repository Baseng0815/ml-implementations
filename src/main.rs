use core::fmt;
use std::{path::Path, fs::File, error::Error};
use std::io::prelude::*;

use plotters::prelude::*;

#[derive(Debug)]
struct Datapoint {
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

impl Datapoint {
    pub fn from_line(line: &str) -> Result<Datapoint, Box<dyn Error>> {
        let split: Vec<String> = line.split_whitespace().map(|s| s.to_owned()).collect();
        eprintln!("split = {:#?}", split);
        if split.len() != 14 {
            Err(DataParseError::new())?
        }
        Ok(Datapoint {
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

fn read_data(path: &Path) -> Result<Vec<Datapoint>, Box<dyn Error>> {
    let mut file = File::open(path)?;
    let mut content = String::new();
    file.read_to_string(&mut content)?;
    let data: Result<Vec<_>, _> = content.split("\n")
        .map(|line| Datapoint::from_line(&line))
        .collect();

    data
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data = read_data(Path::new("./data/housing.csv"))?;
    eprintln!("data = {:#?}", data);

    let root = BitMapBackend::new("plot.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption("y=x^2", ("sans-serif", 50).into_font())
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(-1f32..1f32, 0f32..1f32)?;

    chart.configure_mesh().draw()?;
    chart.draw_series(LineSeries::new(
            (-50..=50).map(|x| x as f32 / 50.0).map(|x| (x, x*x)),
            &GREEN))?
        .label("y = x^2")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));
    chart.configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;

    Ok(())
}
