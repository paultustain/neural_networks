use ndarray::{arr2, Array, Array1, Array2, Axis, Zip, s};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::StandardNormal;
use ndarray_stats::QuantileExt;
use std::f64::consts::E;
// use num_traits::Float;


const EPOCHS: i64 = 5000;
const LEARNING_RATE: f64 = 0.4;
const PURPLE: [f64; 3] = [1., 0., 0.];
const ORANGE: [f64; 3] = [0., 1., 0.];
const GREEN: [f64; 3] = [0., 0., 1.];
const INPUT_COUNT: usize = 2;
const HIDDEN_COUNT: usize = 8;
const OUTPUT_COUNT: usize = 3;


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


fn relu_act(array: &mut Array2<f64>) -> Array2<f64> {
    let mut act: Array2<f64> = Array2::zeros((array.shape()[0], array.shape()[1]));
    
    Zip::from(&mut act)
        .and(array)
        .for_each(|a: &mut f64, p: &mut f64| {
            *a += p.max(0.0)
        });
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
    
        
    let mut w_i_h: Array2<f64> = Array::random((HIDDEN_COUNT, INPUT_COUNT),  StandardNormal);
    // arr2(&[ 
    //     [0.1, -0.2], [-0.3, 0.25], [0.12, 0.23], [-0.11, -0.22]
    // ]);


    // println!("{} {}", rand, w_i_h);

    let mut w_h_o: Array2<f64> = Array::random((OUTPUT_COUNT, HIDDEN_COUNT),  StandardNormal);
    // arr2(&[ 
    //     [0.2, 0.17, 0.3, -0.11], [0.3, -0.4, 0.5, -0.22], [0.12, 0.23, 0.15, 0.33]
    // ]);    

    let mut b_i_h: Array1<f64> = Array1::zeros(HIDDEN_COUNT);
    //arr1(&[0.2, 0.34, 0.21, 0.44]);
    let mut b_h_o: Array1<f64> = Array1::zeros(OUTPUT_COUNT); 
    //arr1(&[0.3, 0.29, 0.37]);
     
    for _ in 0..EPOCHS { 
        let mut pred_h: Array2<f64> = inputs.dot(&w_i_h.t()) + b_i_h.clone();
        let act_h = relu_act(&mut pred_h);
        let mut pred_o: Array2<f64> = act_h.dot(&w_h_o.t()) + b_h_o.clone();

        let act_o: Array2<f64> = softmax(&mut pred_o);

        // let mut cost = Array1<f64> = Array1::zeros(act_o.shape()[1]);

        // let cost = Zip::from(& act_o)
        //                 .and(&mut outputs)
        //                 .for_each(|a: &mut f64, o: &mut f64| -> f64 {
        //                     log_loss(a, o)
        //                 });

        // Back prop - error derivatives 

        let error_d_o: Array2<f64> = &act_o - &outputs;
        let mut temp_errors: Array2<f64> = error_d_o.dot(&w_h_o);

        let mut error_d_h: Array2<f64> = Array2::zeros((pred_h.shape()[0], pred_h.shape()[1]));
        Zip::from(&mut error_d_h)
            .and(&mut pred_h)
            .and(&mut temp_errors)
            .for_each(|e: &mut f64, p: &mut f64, t: &mut f64 | {
                if *p >= 0.0 { 
                    *e += *t
                } else {
                    *e = 0.
                }
            });
        
        let w_h_o_d: Array2<f64> = act_h.t().dot(&error_d_o);
        let mut b_h_o_d: Array1<f64> = Array1::zeros(error_d_o.shape()[1]);
        Zip::from(&mut b_h_o_d)
            .and(error_d_o.t().rows())
            .for_each(|b: &mut f64, row| {
                *b += row.iter().sum::<f64>();
            });

        let w_i_h_d: Array2<f64> = inputs.t().dot(&error_d_h);
        let mut b_i_h_d: Array1<f64> = Array1::zeros(error_d_h.shape()[1]);
        Zip::from(&mut b_i_h_d)
            .and(error_d_h.t().rows())
            .for_each(|b: &mut f64, row| {
                *b += row.iter().sum::<f64>();
            });
            

        w_h_o = w_h_o - LEARNING_RATE * &w_h_o_d.t() / inputs.shape()[0] as f64;
        b_h_o = b_h_o - LEARNING_RATE * &b_h_o_d / inputs.shape()[0] as f64;
        w_i_h = w_i_h - LEARNING_RATE * &w_i_h_d.t() / inputs.shape()[0] as f64;
        b_i_h = b_i_h - LEARNING_RATE * &b_i_h_d / inputs.shape()[0] as f64;
        // println!("{} {} {} {}", w_h_o, b_h_o, w_i_h, b_i_h);
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
    
    let mut test_pred_h: Array2<f64> = test_inputs.dot(&w_i_h.t()) + b_i_h.clone();
    let test_act_h = relu_act(&mut test_pred_h);
    let mut test_pred_o: Array2<f64> = test_act_h.dot(&w_h_o.t()) + b_h_o.clone();
    let mut test_act_o: Array2<f64> = softmax(&mut test_pred_o);

    
    let mut correct: i8 = 0;

    for (i, row) in test_act_o.axis_iter_mut(Axis(0)).enumerate() {
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



