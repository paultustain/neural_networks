use ndarray::{arr1, arr2, Array1, Array2, Axis, Zip};
use ndarray_stats::QuantileExt;
use std::f64::consts::E;

const EPOCHS: i64 = 1;
const LEARNING_RATE: f64 = 0.1;
const RED: (i8, i8, i8) = (1, 0, 0);
const GREEN: (i8, i8, i8) = (0, 1, 0);
const BLUE: (i8, i8, i8) = (0, 0, 1);

// fn softmax(predictions: Vec<f64>) -> Vec<f64> {
//     let mut temp_vec: Vec<f64> = vec![0.; predictions.len()];

//     for (i, mut temp) in temp_vec.enumerate() { 
//         temp = predictions[i]
//     }
//     temp_vec
// }

fn main() {
    let inputs: Array2<f64> = arr2(&[
        [0.0000, 0.0000], [0.2778, 0.2500], [0.2778, 0.9375], [0.9167, 0.6563],
        [0.4167, 0.2500], [0.3611, 0.3438], [0.3333, 0.4063], [0.9722, 0.3750],
        [0.0833, 0.3438], [0.6389, 0.3438], [0.4167, 0.6875], [0.7500, 0.6875],
        [0.0833, 0.1875], [0.9167, 0.5313], [0.1389, 0.2500], [0.8333, 0.6250],
        [0.8056, 0.6250], [0.1944, 1.0000], [0.8333, 0.5625], [0.4167, 1.0000],
        [1.0000, 0.6875], [0.4722, 0.6563], [0.3611, 0.5625], [0.4722, 0.8438],
        [0.1667, 0.3125], [0.4167, 0.9375], [0.3611, 0.9688], [0.9167, 0.3438],
        [0.0833, 0.0313], [0.3333, 0.8750]
    ]);
    
    let target: Array1<(i8, i8, i8)> = arr1(&[RED, RED, BLUE, GREEN, RED, RED, RED, GREEN, RED, GREEN, BLUE, GREEN, RED,
        GREEN, RED, GREEN, GREEN, BLUE, GREEN, BLUE, GREEN, BLUE, BLUE, BLUE,
        RED, BLUE, BLUE, GREEN, RED, BLUE]);
    
    // needs to have n lists that is the same as input points
    // number within each list is the number of output points
    let weights: Array2<f64> = arr2(&[
        [0.1, 0.15, 0.18], 
        [0.2, 0.25, 0.1]
    ]);  
    
    let mut bias: Array1<f64> = arr1(&[0.3, 0.4, 0.35]);

    for _ in 0..EPOCHS {
        let mut pred: Array2<f64> = inputs.dot(&weights) + bias.clone();
        let mut act: Array2<f64> = Array2::zeros((pred.shape()[0], pred.shape()[1]));

        for mut row in pred.axis_iter_mut(Axis(0)) {
            let max_row = row.max().unwrap();
            row -= *max_row;
        }

        Zip::from(&mut act)
            .and(&pred)
            .for_each(|a: &mut f64, p: &f64| {
                *a = E.powf(*p)
            });

        for mut row in act.axis_iter_mut(Axis(0)) {
            let total: f64 = row.iter().sum();
            row /= total;
        }




        // for (i, mut row) in act.axis_iter_mut(Axis(0)).enumerate() {
        //     let pred_row: Array1<f64> = arr1(pred.slice(s![i..i+1, ..]).into_shape(3).unwrap());
        //     let pred_max = arr1(&[pred_row.max().unwrap(); 3]);
        //     row = &pred_row - &pred_max;
        //     println!("{} {} {}", row, pred_row, pred_max);
            // softmax(&pred_row);
        // }
        // let mut act: Array2<f64> = Array2::zeros([pred.shape()[0], pred.shape()[1]]);
        
        // Zip::from(&mut act)
        //     .and(&pred)
        //     .for_each(|a: &mut f64, row| *a += row);

        // println!("{}", act);
    }
}
