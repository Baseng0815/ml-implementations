use std::{error::Error, f64::consts::E, fs::File, io::Read, path::Path};

use nalgebra::DVector;
use plotters::{backend::BitMapBackend, chart::ChartBuilder, coord::Shift, drawing::{DrawingArea, IntoDrawingArea}, element::{Circle, Rectangle}, series::LineSeries, style::{Color, IntoFont, ShapeStyle, BLACK, BLUE, CYAN, GREEN, MAGENTA, RED, WHITE, YELLOW}};

use crate::{data::{Datapoint, Dataset}, regression::LinearRegressionModel, util::{distribution_properties, DataParseError}};

#[derive(Debug)]
struct HousingDatapoint {
    pub crim: f64, // per capita crime rate by town
    pub zn: f64, // proportion of residential land zoned for lots over 25,000 sq.ft.
    pub indus: f64, // proportion of non-retail business acres per town
    pub chas: f64, // Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
    pub nox: f64, // nitric oxides concentration (parts per 10 million)
    pub rm: f64, // average number of rooms per dwelling
    pub age: f64, // proportion of owner-occupied units built prior to 1940
    pub dis: f64, // weighted distances to ﬁve Boston employment centers
    pub rad: f64, // index of accessibility to radial highways
    pub tax: f64, // full-value property-tax rate per $10,000
    pub ptratio: f64, // pupil-teacher ratio by town 12. B: 1000(Bk−0.63)2 where Bk is the proportion of blacks by town 13. LSTAT: % lower status of the population
    pub b: f64, // 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
    pub lstat: f64, // % lower status of the population
    pub medv: f64, // Median value of owner-occupied homes in $1000s
}

#[derive(Debug, Clone)]
struct MailDatapoint {
    pub index: String,
    pub word_count: Vec<usize>,
    pub is_spam: bool
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

impl MailDatapoint {
    pub fn from_line(line: &str) -> Result<MailDatapoint, Box<dyn Error>> {
        let split: Vec<String> = line.split(",").map(|s| s.to_owned()).collect();
        if split.len() != 3002 {
            Err(DataParseError::new())?
        }
        let mut word_count: Vec<usize> = Vec::new();
        for str in split.iter().skip(1).take(3000) {
            word_count.push(str.parse()?);
        }
        Ok(MailDatapoint {
            index: split[0].clone(),
            word_count,
            is_spam: split[3001] == "0"
        })
    }
}

impl From<&HousingDatapoint> for Datapoint {
    fn from(value: &HousingDatapoint) -> Self {
        let features = DVector::from_row_slice(&[
            1.0, value.crim, value.zn, value.indus, value.chas, value.nox, value.rm,
            value.age, value.dis, value.rad, value.tax, value.ptratio, value.b, value.lstat ]);

        Datapoint {
            features,
            value: value.medv
        }
    }
}

fn read_housing_data(path: &Path) -> Result<Vec<HousingDatapoint>, Box<dyn Error>> {
    let mut file = File::open(path)?;
    let mut content = String::new();
    file.read_to_string(&mut content)?;
    let data: Result<Vec<_>, _> = content.split("\n")
        .filter(|line| !line.is_empty())
        .map(|line| HousingDatapoint::from_line(&line))
        .collect();

    data
}

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

fn spam_filter() -> Result<(), Box<dyn std::error::Error>> {
    let mut file = File::open("./data/emails.csv")?;
    let mut content = String::new();
    file.read_to_string(&mut content)?;
    let dict = content.lines().take(1).collect::<Vec<_>>().get(0).ok_or(DataParseError::new())?.split(",").skip(1).take(3000).map(|s| s.to_owned()).collect::<Vec<_>>();
    eprintln!("dict = {:#?}", dict);
    let data: Result<Vec<_>, _> = content.lines().skip(1)
        .filter(|line| !line.is_empty())
        .map(|line| MailDatapoint::from_line(&line))
        .collect();
    let data = data?;
    let spam_count = data.len();
    let non_spam_count = data.len() - spam_count;

    // probability of spam
    let p_spam = data.iter().filter(|&d| d.is_spam).count() as f64 / spam_count as f64;

    // probability of word k
    let mut p_word_k = vec![0.0; dict.len()];
    for data in data.iter() {
        for (i, &wc) in data.word_count.iter().enumerate() {
            if wc > 0 {
                p_word_k[i] += 1.0;
            }
        }
    }

    for val in p_word_k.iter_mut() {
        *val = (*val + 1.0) / (data.len() as f64 + 1.0);
    }

    // probability of word k in dict appearing in non-spam mail
    let mut p_word_k_nospam: Vec<f64> = vec![0.0; dict.len()];
    for data in data.iter().filter(|&d| !d.is_spam) {
        for (i, &wc) in data.word_count.iter().enumerate() {
            if wc > 0 {
                p_word_k_nospam[i] += 1.0;
            }
        }
    }

    for val in p_word_k_nospam.iter_mut() {
        *val = (*val + 1.0) / (non_spam_count as f64 + 1.0);
    }

    // probability of word k in dict appearing in spam mail
    let mut p_word_k_spam: Vec<f64> = vec![0.0; dict.len()];
    for data in data.iter().filter(|&d| d.is_spam) {
        for (i, &wc) in data.word_count.iter().enumerate() {
            if wc > 0 {
                p_word_k_spam[i] += 1.0;
            }
        }
    }

    for val in p_word_k_spam.iter_mut() {
        *val = (*val + 1.0) / (spam_count as f64 + 1.0);
    }

    // Pr(Spam|Word) = Pr(Word|Spam)*Pr(Spam) / Pr(Word)

    let mut a: Vec<(&f64, &String)> = p_word_k_spam.iter().zip(dict.iter()).collect();
    a.sort_by(|b, c| b.0.partial_cmp(c.0).unwrap());
    // eprintln!("p_spam = {:#?}", p_spam);
    eprintln!("a = {:#?}", a);

    // classify
    let message = "subject not our low up gas forward forwarded";
    let mut bow = vec![0; dict.len()];
    for (i, word) in dict.iter().enumerate() {
        if message.contains(word) {
            bow[i] = 1;
        }
    }

    let eta: f64 = p_word_k_spam.iter().zip(bow.iter()).filter(|(_, &c)| c > 0).map(|(pi, _)| (1.0 - pi).ln() - pi.ln()).sum();
    let p = 1.0 / (1.0 + E.powf(eta));
    eprintln!("eta = {:#?}", eta);
    eprintln!("p = {:#?}", p);

    Ok(())
}

pub fn run_sample() -> Result<(), Box<dyn std::error::Error>> {
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
