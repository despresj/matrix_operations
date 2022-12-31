/// this is a slow matrix multiplication function from scratch
use ndarray::Array2;

pub fn matrix_multiply(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
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

#[test]
fn test_matrix_multiplication() {
    use ndarray::arr2;
    let a = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
    let b = arr2(&[[5.0, 6.0], [7.0, 8.0]]);
    let c = matrix_multiply(&a, &b);
    assert_eq!(c[(0, 0)], 19.0);
    assert_eq!(c[(0, 1)], 22.0);
    assert_eq!(c[(1, 0)], 43.0);
    assert_eq!(c[(1, 1)], 50.0);

    let a = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    let b = arr2(&[[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]);
    let c = matrix_multiply(&a, &b);
    assert_eq!(c[(0, 0)], 58.0);
    assert_eq!(c[(0, 1)], 64.0);
    assert_eq!(c[(1, 0)], 139.0);
    assert_eq!(c[(1, 1)], 154.0);
}

#[test]
#[should_panic]
fn test_invalid_matrices_multiplication() {
    use ndarray::arr2;
    let a = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [4.0, 5.0, 6.0]]);
    let b = arr2(&[[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]);
    let _ = matrix_multiply(&a, &b);
}
