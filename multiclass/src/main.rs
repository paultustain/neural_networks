use ndarray::{arr1, arr2, Array1, Array2, Axis, Zip, s};
use ndarray_stats::QuantileExt;
use std::f64::consts::E;

const EPOCHS: i64 = 5000;
const LEARNING_RATE: f64 = 0.5;
const PURPLE: [f64; 3] = [1., 0., 0.];
const ORANGE: [f64; 3] = [0., 1., 0.];
const GREEN: [f64; 3] = [0., 0., 1.];

fn _log_loss(a: &f64, t: &f64) -> f64 {
    -t * a.ln() - (1.0 - t) * (1.0 - a).ln()
}

fn softmax(predictions: &mut Array2<f64>) -> Array2<f64> {
    let mut act: Array2<f64> = Array2::zeros((predictions.shape()[0], predictions.shape()[1]));
        
    for mut row in predictions.axis_iter_mut(Axis(0)) {
        let max_row = row.max().unwrap();
        row -= *max_row;
    }

    Zip::from(&mut act)
        .and(predictions)
        .for_each(|a: &mut f64, p: &mut f64| {
            *a = E.powf(*p)
        });

    for mut row in act.axis_iter_mut(Axis(0)) {
        let total: f64 = row.iter().sum();
        row /= total;
    }

    act
}

fn main() {
    let inputs: Array2<f64> = arr2(&[
        [0.0000, 0.3929], [0.5484, 0.7500], [0.0645, 0.5714], [0.5806, 0.5714],
        [0.2258, 0.8929], [0.4839, 0.2500], [0.3226, 0.2143], [0.7742, 0.8214],
        [0.4516, 0.5000], [0.4194, 0.0357], [0.4839, 0.2500], [0.3226, 0.7143],
        [0.5806, 0.5000], [0.5484, 0.1071], [0.6129, 0.6429], [0.6774, 0.1786],
        [0.2258, 0.8214], [0.7419, 0.1429], [0.6452, 1.0000], [0.8387, 0.2500],
        [0.9677, 0.3214], [0.3226, 0.4643], [0.3871, 0.5357], [0.3548, 0.1429],
        [0.3548, 0.6429], [0.1935, 0.4643], [0.4516, 0.3929], [0.4839, 0.6071],
        [0.6129, 0.6786], [0.2258, 0.6071], [0.5161, 0.3214], [0.5484, 0.6786],
        [0.3871, 0.8571], [0.6452, 0.6071], [0.1935, 0.3929], [0.6452, 0.3929],
        [0.6774, 0.4643], [0.3226, 0.2857], [0.7419, 0.7143], [0.7419, 0.3214],
        [1.0000, 0.3929], [0.8065, 0.3929], [0.1935, 0.5000], [0.1613, 0.8214],
        [0.2903, 0.9286], [0.3548, 0.0000], [0.2903, 0.6786], [0.5484, 0.9643],
        [0.4194, 0.1786], [0.2581, 0.2500], [0.3226, 0.7143], [0.5161, 0.3929],
        [0.2903, 0.6429], [0.5484, 0.9286], [0.2581, 0.3214], [0.0968, 0.5000],
        [0.6129, 0.7857], [0.0968, 0.3214], [0.6452, 0.9286], [0.8065, 0.7500]
    ]);
    
    let outputs: Array2<f64> = arr2(&[PURPLE, ORANGE, PURPLE, ORANGE, GREEN, PURPLE, PURPLE, GREEN, ORANGE,
        PURPLE, PURPLE, GREEN, ORANGE, PURPLE, ORANGE, PURPLE, GREEN, PURPLE, GREEN,
        PURPLE, PURPLE, ORANGE, ORANGE, PURPLE, ORANGE, PURPLE, ORANGE, ORANGE, ORANGE,
        GREEN, ORANGE, ORANGE, GREEN, ORANGE, PURPLE, ORANGE, ORANGE, PURPLE, ORANGE,
        ORANGE, PURPLE, ORANGE, GREEN, GREEN, GREEN, PURPLE, GREEN, GREEN, PURPLE, PURPLE,
        GREEN, ORANGE, GREEN, GREEN, PURPLE, PURPLE, GREEN, PURPLE, GREEN, GREEN]);
    
    // needs to have n lists that is the same as input points
    // number within each list is the number of output points
    let mut weights: Array2<f64> = arr2(&[
        [0.1, 0.15, 0.18], 
        [0.2, 0.25, 0.1]
    ]);  
    
    let mut bias: Array1<f64> = arr1(&[0.3, 0.4, 0.35]);

    for _ in 0..EPOCHS {
        let pred: Array2<f64> = inputs.dot(&weights) + bias.clone();
        let act: Array2<f64> = softmax(&mut pred.clone());
        
        let error_delta: Array2<f64> = &act - &outputs;
        let weight_delta: Array2<f64> = inputs.t().dot(&error_delta);

        let mut bias_change: Array1<f64> = Array1::zeros(error_delta.t().shape()[0]);
        Zip::from(&mut bias_change)
            .and(error_delta.t().rows())
            .for_each(|b: &mut f64, row| *b += row.iter().sum::<f64>());

        bias_change = LEARNING_RATE * bias_change / error_delta.shape()[0] as f64;
        let weight_change: Array2<f64> = LEARNING_RATE * &weight_delta / inputs.shape()[0] as f64;
        
        weights = weights - weight_change;
        bias = bias - bias_change;
        println!("{}", bias);
    }
     
    let test_inputs: Array2<f64> = arr2(&[
        [0.0000, 0.3929], [0.0645, 0.5714], [0.0968, 0.3214],
        [0.0968, 0.5000], [0.2581, 0.3214], [0.1935, 0.4643], [0.2581, 0.2500],
        [0.1935, 0.3929], [0.3226, 0.2143], [0.4839, 0.2500], [0.3226, 0.4643],
        [0.3871, 0.5357], [0.3548, 0.6429], [0.4516, 0.5000], [0.4516, 0.3929],
        [0.5161, 0.3929], [0.5484, 0.7500], [0.6129, 0.6786], [0.5161, 0.3214],
        [0.5484, 0.6786], [0.1935, 0.5000], [0.2258, 0.6071], [0.3226, 0.7143],
        [0.2903, 0.6786], [0.3226, 0.7143], [0.2258, 0.8214], [0.2903, 0.6429],
        [0.6129, 0.7857], [0.7742, 0.8214], [0.8065, 0.7500]
    ]);
    let test_targets: Array2<f64> = arr2(&[
        PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE,
        PURPLE, PURPLE, ORANGE, ORANGE, ORANGE, ORANGE, ORANGE, ORANGE, 
        ORANGE, ORANGE, ORANGE, ORANGE, GREEN, GREEN, GREEN, GREEN, 
        GREEN, GREEN, GREEN, GREEN, GREEN, GREEN
    ]);
    let test_pred: Array2<f64> = test_inputs.dot(&weights) + bias.clone();
    let mut test_act: Array2<f64> = softmax(&mut test_pred.clone());
    let mut correct: i8 = 0;
    
    for (i, row) in test_act.axis_iter_mut(Axis(0)).enumerate() {
        let max_index_pred: usize = row
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(index, _)| index).unwrap();
        let target_row = test_targets.slice(s![i..i+1, ..]);
        let max_index_target: usize = target_row
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(index, _)| index).unwrap();
        println!("{} {}", max_index_pred, max_index_target);
        if max_index_target == max_index_pred {
            correct += 1;
        }
    }
    println!("{} / {} Correct", correct, test_inputs.shape()[0]);
    // println!("{} {}", test_act, test_targets);
    
    
}
