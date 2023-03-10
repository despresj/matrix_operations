use ndarray::Array2;
use rand::thread_rng;
use rand_distr::{Distribution, Normal};

pub fn norm_mat(mu: f64, sigma: f64, nrow: usize, ncol: usize) -> Array2<f64> {
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

#[test]
fn test_norm_mat() {
    let a = norm_mat(0.0, 1.0, 2, 2);
    assert_eq!(a.nrows(), 2);
    assert_eq!(a.ncols(), 2);

    let b = norm_mat(0.0, 1.0, 9, 1);
    assert_eq!(b.nrows(), 9);
    assert_eq!(b.ncols(), 1);
}

#[test]
#[should_panic]
fn test_input_invalad_norm_mat() {
    let _ = norm_mat(0.0, -1.0, 2, 2);
    let _ = norm_mat(0.0, 0.0, 2, 2);
    let _ = norm_mat(0.0, 1.0, 2, 0);
    let _ = norm_mat(0.0, 1.0, 0, 2);
}
