use ndarray::{arr1, arr2, Array1, Array2};

const ENOCHS: i64 = 4000;
const LEARNING_RATE: f64 = 0.3;

fn main() {
    // Network set up
    let mut weights: Array1<f64> = arr1(&[0.1, 0.2]);
    let mut bias: f64 = 0.3;
    
    // Data set up
    let inputs: Array2<f64> = arr2(&[
        [0.0000, 0.0000], [0.1600, 0.1556], [0.2400, 0.3543], [0.2800, 0.3709],
        [0.3600, 0.4702], [0.4000, 0.4868], [0.5000, 0.5530], [0.5200, 0.6026],
        [0.6000, 0.6358], [0.6200, 0.3212], [0.6600, 0.7185], [0.7000, 0.7351],
        [0.7600, 0.8013], [0.8400, 0.7848], [0.9600, 0.9669], [1.0000, 1.0000]
    ]);

    let outputs: Array1<f64> = arr1(&[230.0, 555.0, 815.0, 860.0, 1140.0, 1085.0, 1200.0, 1330.0, 1290.0, 870.0, 1545.0, 1480.0, 1750.0, 1845.0, 1790.0, 1955.0]);
    
    for _ in 0..ENOCHS {
        let pred: Array1<f64> = inputs.dot(&weights) + bias;
        let _cost: f64 = (&pred - &outputs).iter().map(|x| x.powf(2.0)).collect::<Array1<f64>>().iter().sum::<f64>() / pred.shape()[0] as f64;
        let error_delta: Array1<f64> = 2.0 * (&pred - &outputs);
        let weight_delta: Array1<f64> = inputs.t().dot(&error_delta) / error_delta.len() as f64;

        weights = &weights - (LEARNING_RATE * &weight_delta);
        bias -= LEARNING_RATE * error_delta.iter().sum::<f64>() / error_delta.len() as f64;
    }
    
    let test_inputs: Array2<f64> = arr2(&[
        [0.16, 0.1391], [0.56, 0.3046], [0.76, 0.8013]
    ]);

    let test_pred = test_inputs.dot(&weights) + bias;
    println!("{}", test_pred);

}