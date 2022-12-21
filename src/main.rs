use ndarray::{arr2, Array2};
use rand::distributions::Standard;
use rand::prelude::*;

fn main() {
    let a = arr2(&[[1, 2, 3], [4, 5, 6]]);

    let b = arr2(&[[6, 5, 4], [3, 2, 1]]);

    start_here(&a, &b);

    let val: [u64; 8] = StdRng::from_entropy().sample(Standard);
    println!("f32 from [0, 1): {:?}", val);
}

fn start_here(a: &Array2<i32>, b: &Array2<i32>) {
    for number in a {
        println!("number A {}", number)
    }
    for number in b {
        println!("number B {}", number)
    }
}
