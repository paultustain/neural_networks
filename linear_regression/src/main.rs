const ENOCHS: i64 = 4000;
const LEARNING_RATE: f64 = 0.1;

#[derive(Copy, Clone, Debug)]
struct Function {
    w_age: f64, 
    w_miles: f64,
    bias: f64
}

impl Function {
    fn new(w_age: f64, w_miles: f64, bias: f64) -> Self{
        Function { 
            w_age: w_age, 
            w_miles: w_miles,
            bias: bias,
        }
    }

    fn predict(self, age: f64, milage: f64) -> f64 {
        self.w_age * age + self.w_miles * milage + self.bias
    }

    fn update_weight_age(&mut self, new_w: f64) { 
        self.w_age -= new_w;
    }
    fn update_weight_miles(&mut self, new_w: f64) { 
        self.w_miles -= new_w;
    }    
    fn update_bias(&mut self, new_bias: f64) { 
        self.bias -= new_bias;
    }
}

fn main() {
    let mut f: Function = Function::new(0.1, 0.2, 0.3);
    let inputs: Vec<(f64, f64)> = vec![
        (0.0000, 0.0000), (0.1600, 0.1556), (0.2400, 0.3543), (0.2800, 0.3709),
        (0.3600, 0.4702), (0.4000, 0.4868), (0.5000, 0.5530), (0.5200, 0.6026),
        (0.6000, 0.6358), (0.6200, 0.3212), (0.6600, 0.7185), (0.7000, 0.7351),
        (0.7600, 0.8013), (0.8400, 0.7848), (0.9600, 0.9669), (1.0000, 1.0000)
    ];
    let outputs: Vec<f64> = vec![230.0, 555.0, 815.0, 860.0, 1140.0, 1085.0, 1200.0, 1330.0, 1290.0, 870.0, 1545.0, 1480.0, 1750.0, 1845.0, 1790.0, 1955.0];

    // let test_inputs: Vec<f64> = vec![5.0, 6.0];
    // let test_outputs: Vec<f64> = vec![20.0, 22.0];
    let age: Vec<f64> = inputs.iter().map(|i| i.0).collect();
    let miles: Vec<f64> = inputs.iter().map(|i| i.1).collect();
    
    for _ in 0..ENOCHS {
        let pred: Vec<f64> = inputs.iter().map(| i | f.predict(i.0, i.1)).collect();

        // Difference between prediction and output squared.
        let errors: Vec<f64> = pred.iter().zip(outputs.iter()).map(|p: (&f64, &f64)| (p.0 - p.1).powf(2.0)).collect();
        println!("Weight (age): {:.2} Weight (miles): {:.2} Bias: {:.2} Cost: {:.2}", f.w_age, f.w_miles, f.bias, errors.iter().sum::<f64>() / errors.len() as f64 );

        let errors_d: Vec<f64> = pred.iter().zip(outputs.iter()).map(|p: (&f64, &f64)| 2.0 * (p.0 - p.1)).collect();
        
        let weight_age_d: Vec<f64> = errors_d.iter().zip(age.iter()).map(|p: (&f64, &f64)| p.0 * p.1).collect();
        let weight_miles_d: Vec<f64> = errors_d.iter().zip(miles.iter()).map(|p: (&f64, &f64)| p.0 * p.1).collect();

        // let bias_d: Vec<f64> = errors_d.iter().map(|x| x).collect();

        f.update_weight_age(LEARNING_RATE * weight_age_d.iter().sum::<f64>() / weight_age_d.len() as f64);
        f.update_weight_miles(LEARNING_RATE * weight_miles_d.iter().sum::<f64>() / weight_miles_d.len() as f64);
        f.update_bias(LEARNING_RATE * errors_d.iter().sum::<f64>() / errors_d.len() as f64)
    }

    // let predictions: Vec<f64> = test_inputs.iter().map(|i| f.predict(*i)).collect();
    // for n in 0..predictions.len() as usize {
    //     println!("Input: {:.2}, Prediction: {:.4}, Actual: {:.2}", test_inputs[n], predictions[n], test_outputs[n]);
    // }


}

