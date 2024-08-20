use examples::housing::run_sample;

mod util;
mod regression;
mod data;
mod examples;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    run_sample()
}
