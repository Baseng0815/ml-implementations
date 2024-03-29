pub struct Quantiles {
    pub min: f64,
    pub pointzerofive: f64,
    pub pointtwofive: f64,
    pub pointfive: f64,
    pub pointsevenfive: f64,
    pub pointninefive: f64,
    pub max: f64,
}

pub fn quantiles(data: &Vec<f64>) -> Option<Quantiles> {
    if data.is_empty() {
        return None
    }

    let mut sorted = data.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    sorted.dedup();

    Some(Quantiles {
        min: sorted[0],
        pointzerofive: sorted[(sorted.len() as f64 * 0.05) as usize],
        pointtwofive: sorted[(sorted.len() as f64 * 0.25) as usize],
        pointfive: sorted[(sorted.len() as f64 * 0.5) as usize],
        pointsevenfive: sorted[(sorted.len() as f64 * 0.75) as usize],
        pointninefive: sorted[(sorted.len() as f64 * 0.95) as usize],
        max: sorted[sorted.len() - 1]
    })
}
