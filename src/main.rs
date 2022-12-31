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

// matrix_multiply is a basic matrix multiplication function
// that will take in two Array2<f64> types and return and Array2<f64>
fn matrix_multiply(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let a_rows = a.nrows();
    let b_cols = b.ncols();
    let a_cols = a.ncols();

    assert_eq!(
        a_rows, b_cols,
        "Matrix Multiplication is undefined unles a rows is equal to b cols"
    );

    let mut result = Array2::zeros((a_rows, b_cols));

    for i in 0..a_rows {
        for j in 0..b_cols {
            for k in 0..a_cols {
                result[[i, j]] += a[[i, k]] * b[[k, j]];
            }
        }
    }

    result
}

fn main() {
    let dims = 100;
    let start = Instant::now();
    let a = norm_mat(0., 1., dims, dims);
    let b = norm_mat(0., 1., dims, dims);
    let elapsed = start.elapsed();
    println!("Elapsed time to create matrixies : {:?}", elapsed / 2);

    let start = Instant::now();
    let _ = matrix_multiply(&a, &b);
    println!("Elapsed time from scratch: {:?}", start.elapsed());

    let start = Instant::now();
    let _ = a.dot(&b);
    println!("Elapsed time from .dot : {:?}", start.elapsed());
}
