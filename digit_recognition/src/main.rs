use ndarray::{Array, Array1, Array2, Axis, Zip, s};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::StandardNormal;
// use ndarray_stats::QuantileExt;
use std::f64::consts::E;
use std::fs;
use rand::seq::SliceRandom;
use rand::thread_rng;

const BATCH_SIZE: usize = 400;
const EPOCHS: i64 = 300;
const LEARNING_RATE: f64 = 0.5;
const INPUT_COUNT: usize = 738;
const HIDDEN_COUNT: usize = 8;
const OUTPUT_COUNT: usize = 10;

// fn _log_loss(a: &f64, t: &f64) -> f64 {
//     -t * a.ln() - (1.0 - t) * (1.0 - a).ln()
// }


fn softmax(predictions: &mut Array2<f64>) -> Array2<f64> {
    let mut act: Array2<f64> = Array2::zeros((predictions.shape()[0], predictions.shape()[1]));
        
    for mut row in predictions.axis_iter_mut(Axis(0)) {
        let mut max_row: f64 = -9999999.;
        for c in &row {
            if *c > max_row { 
                max_row = *c;
            }
        }
        row -= max_row;

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


fn shuffle_data() -> Vec<String> {
    let contents = fs::read_to_string("train.csv").expect("cannot read");
    let mut rng = thread_rng();
    let mut figures = Vec::new();

    for line in contents.lines() { 
        figures.push(line.to_string());
    }

    figures.shuffle(&mut rng);


    figures
}


fn main() {
    let figures = shuffle_data();

    let mut start = 0;
    let mut w_i_h: Array2<f64> = Array::random((HIDDEN_COUNT, INPUT_COUNT),  StandardNormal);
   
    let mut w_h_o: Array2<f64> = Array::random((OUTPUT_COUNT, HIDDEN_COUNT),  StandardNormal);

    let mut b_i_h: Array1<f64> = Array1::zeros(HIDDEN_COUNT);
    let mut b_h_o: Array1<f64> = Array1::zeros(OUTPUT_COUNT); 

    while start < figures.len() { 

        let mut labels_load = Vec::new();
        let mut targets_load: Vec<Vec<&str>> = Vec::new();
        let mut inputs_load: Vec<Vec<&str>> = Vec::new();
        let end = start + BATCH_SIZE;
        println!("{} to {}", start, end);
        for line in &figures[start..end] {
            let collection: Vec<&str> = line.split(",").collect();
            labels_load.push(collection[0]);
            targets_load.push((&collection[1..11]).to_vec());
            inputs_load.push((&collection[12..]).to_vec());
        }             

        let mut inputs: Array2<f64> = Array2::zeros((BATCH_SIZE, INPUT_COUNT));
        for (i, mut row) in inputs.axis_iter_mut(Axis(0)).enumerate() {
            for (j, col) in row.iter_mut().enumerate() {
                *col = inputs_load[i][j].parse::<f64>().unwrap();
            }
        }
        let mut outputs: Array2<f64> = Array2::zeros((BATCH_SIZE, OUTPUT_COUNT));
        for (i, mut row) in outputs.axis_iter_mut(Axis(0)).enumerate() {
            for (j, col) in row.iter_mut().enumerate() {
                *col = targets_load[i][j].parse::<f64>().unwrap();
            }
        }
        // println!("{:#?}", outputs);

        // let mut w_i_h: Array2<f64> = Array::random((HIDDEN_COUNT, INPUT_COUNT),  StandardNormal);
   
        // let mut w_h_o: Array2<f64> = Array::random((OUTPUT_COUNT, HIDDEN_COUNT),  StandardNormal);
    
        // let mut b_i_h: Array1<f64> = Array1::zeros(HIDDEN_COUNT);
        // let mut b_h_o: Array1<f64> = Array1::zeros(OUTPUT_COUNT); 
         
        for _ in 0..EPOCHS { 
            // println!("{}", e);
            let mut pred_h: Array2<f64> = inputs.dot(&w_i_h.t()) + b_i_h.clone();
            let act_h = relu_act(&mut pred_h);
            let mut pred_o: Array2<f64> = act_h.dot(&w_h_o.t()) + b_h_o.clone();    
            let act_o: Array2<f64> = softmax(&mut pred_o);
    
        //     // let mut cost = Array1<f64> = Array1::zeros(act_o.shape()[1]);
    
        //     // let cost = Zip::from(& act_o)
        //     //                 .and(&mut outputs)
        //     //                 .for_each(|a: &mut f64, o: &mut f64| -> f64 {
        //     //                     log_loss(a, o)
        //     //                 });
    
        //     // Back prop - error derivatives 
    
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
    
        start += BATCH_SIZE;

    }


    let contents = fs::read_to_string("test.csv").expect("cannot read");
    let mut test_figures_load = Vec::new();

    for line in contents.lines() { 
        test_figures_load.push(line.to_string());
    }

    let mut test_labels_load = Vec::new();
    let mut test_targets_load: Vec<Vec<&str>> = Vec::new();
    let mut test_inputs_load: Vec<Vec<&str>> = Vec::new();
    for line in &test_figures_load {
        let collection: Vec<&str> = line.split(",").collect();
        test_labels_load.push(collection[0]);
        test_targets_load.push((&collection[1..11]).to_vec());
        test_inputs_load.push((&collection[12..]).to_vec());
    }             

    let mut test_inputs: Array2<f64> = Array2::zeros((1000, INPUT_COUNT));
    for (i, mut row) in test_inputs.axis_iter_mut(Axis(0)).enumerate() {
        for (j, col) in row.iter_mut().enumerate() {
            *col = test_inputs_load[i][j].parse::<f64>().unwrap();
        }
    }
    let mut test_outputs: Array2<f64> = Array2::zeros((1000, OUTPUT_COUNT));
    for (i, mut row) in test_outputs.axis_iter_mut(Axis(0)).enumerate() {
        for (j, col) in row.iter_mut().enumerate() {
            *col = test_targets_load[i][j].parse::<f64>().unwrap();
        }
    }
    let mut test_pred_h: Array2<f64> = test_inputs.dot(&w_i_h.t()) + b_i_h.clone();
    let test_act_h = relu_act(&mut test_pred_h);
    let mut test_pred_o: Array2<f64> = test_act_h.dot(&w_h_o.t()) + b_h_o.clone();    
    let mut test_act_o: Array2<f64> = softmax(&mut test_pred_o);

    let mut correct: i32 = 0;

    for (i, row) in test_act_o.axis_iter_mut(Axis(0)).enumerate() {
        let max_index_pred: usize = row
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(index, _)| index).unwrap();
        let target_row = test_outputs.slice(s![i..i+1, ..]);
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
}
 