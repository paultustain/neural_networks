use ndarray::{arr1, arr2, Array1, Array2, Zip};

const ENOCHS: i64 = 400;
const LEARNING_RATE: f64 = 0.75;

fn activate(x: f64) -> f64{
    1.0 / (1.0 + (-x).exp())
}

fn log_loss(a: &f64, t: &f64) -> f64 {
    -t * a.ln() - (1.0 - t) * (1.0 - a).ln()
}

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

    let outputs: Array1<f64> = arr1(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]); // 0 keep, 1 sell
    
    for e in 0..ENOCHS {
        let pred: Array1<f64> = inputs.dot(&weights) + bias;
        let act: Array1<f64> = pred.iter().map(|x| activate(*x)).collect(); // Activate predictions in sigmoid function 
        let mut cost_array = Array1::zeros(act.shape()[0]);
        
        Zip::from(&mut cost_array)
            .and(&act)
            .and(&outputs)
            .for_each(|c: &mut f64, &a, &o| *c += log_loss(&a, &o)); // cost now uses log losses rather than mean square error 
        
        let cost: f64 = cost_array.iter().sum::<f64>() / act.shape()[0] as f64;
        println!("Enoch: {} Cost {:.2}", e, cost);

        let error_delta: Array1<f64> = &act - &outputs;
        let weight_delta: Array1<f64> = inputs.t().dot(&error_delta) / error_delta.shape()[0] as f64;

        weights = &weights - (LEARNING_RATE * &weight_delta);
        bias -= LEARNING_RATE * error_delta.iter().sum::<f64>() / error_delta.shape()[0] as f64;
    }
    
    let test_inputs: Array2<f64> = arr2(&[
        [0.16, 0.1391], [0.56, 0.3046], [0.76, 0.8013], [0.96, 0.3046], [0.16, 0.7185]
    ]);

    let test_pred: Array1<f64> = test_inputs.dot(&weights) + bias;
    let test_act: Array1<f64> = test_pred.iter().map(|x| activate(*x)).collect();
    println!("{:.0}", test_act);

}