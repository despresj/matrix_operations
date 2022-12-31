mod multiplication;
use ndarray::Array2;
use rand::thread_rng;
use rand_distr::{Distribution, Normal};
use std::time::Instant;

fn norm_mat(mu: f64, sigma: f64, nrow: usize, ncol: usize) -> Array2<f64> {
    assert!(sigma > 0., "sigma must be greater than zero");
    assert!(nrow > 0, "must have a non zero number of rows");
    assert!(ncol > 0, "must have a non zero number of columns");

    let mut rng = thread_rng();
    let normal = Normal::new(mu, sigma).expect("mu and sigma are real and sigma is greater than 0");

    let mut matrix = Array2::<f64>::zeros((nrow, ncol));

    for i in 0..nrow {
        for j in 0..ncol {
            matrix[(i, j)] = normal.sample(&mut rng);
        }
    }

    matrix
}

fn main() {
    let dims = 100;
    let start = Instant::now();
    let a = norm_mat(0., 1., dims, dims);
    let b = norm_mat(0., 1., dims, dims);
    let elapsed = start.elapsed();
    println!("Elapsed time to create matrixies : {:?}", elapsed / 2);

    let start = Instant::now();
    let _ = multiplication::matrix_multiply(&a, &b);
    println!("Elapsed time from scratch: {:?}", start.elapsed());

    let start = Instant::now();
    let _ = a.dot(&b);
    println!("Elapsed time from .dot : {:?}", start.elapsed());
}
