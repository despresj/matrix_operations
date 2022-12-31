use std::time::Instant;
mod mat_helpers;
mod multiplication;

fn main() {
    let dims = 100;
    let start = Instant::now();
    let a = mat_helpers::norm_mat(0., 1., dims, dims);
    let b = mat_helpers::norm_mat(0., 1., dims, dims);
    let elapsed = start.elapsed();
    println!("Elapsed time to create matrixies : {:?}", elapsed / 2);

    let start = Instant::now();
    let _ = multiplication::matrix_multiply(&a, &b);
    println!("Elapsed time from .dot : {:?}", start.elapsed());

    let start = Instant::now();
    let _ = a.dot(&b);
    println!("Elapsed time from .dot : {:?}", start.elapsed());
}
