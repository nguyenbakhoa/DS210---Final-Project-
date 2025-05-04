// Module: model - Implements the logistic regression algorithm and feature scaling.

// Scales features to a range [0, 1] based on training data min/max.
pub struct MinMaxScaler {
    mins: Vec<f32>,
    maxs: Vec<f32>,
}

impl MinMaxScaler {
    // Creates a new, unfitted MinMaxScaler.
    pub fn new() -> Self {
        MinMaxScaler {
            mins: Vec::new(),
            maxs: Vec::new(),
        }
    }

    // Calculates and stores the min/max for each feature from the training data.
    pub fn fit(&mut self, data: &Vec<Vec<f32>>) {
        if data.is_empty() {
            self.mins = Vec::new();
            self.maxs = Vec::new();
            return;
        }
        let num_features = data[0].len();
        // Initialize mins to positive infinity so that any actual value will be smaller    
        self.mins = vec![f32::INFINITY; num_features]; 
        // Initialize maxs to negative infinity so that any actual value will be larger
        self.maxs = vec![f32::NEG_INFINITY; num_features]; 

        // Iterate over the training data and calculate the min and max for each feature.
        // These values will be used to scale the data in the future.
        for features in data {
            for j in 0..num_features {
                self.mins[j] = self.mins[j].min(features[j]);
                self.maxs[j] = self.maxs[j].max(features[j]); 
            }
        }
    }

    // Applies the learned Min-Max scaling to new data.
    // Clamps values to the fitted range before scaling.
    pub fn transform(&self, data: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        if self.mins.is_empty() || data.is_empty() {
            return data.clone(); // Return original data if scaler not fit or data empty
        }
        let num_features = self.mins.len();
        // Iterate over the data and scale each feature.
        // The result is a new vector of scaled features.
        data.iter().map(|features| {
            features.iter().enumerate().map(|(j, &val)| {
                if j >= num_features { 
                    return val; // Should not happen if data structure is consistent
                }
                let range = self.maxs[j] - self.mins[j];
                let clamped_val = val.max(self.mins[j]).min(self.maxs[j]);
                if range == 0.0 {
                    0.0 // Or handle as appropriate, e.g., return 0 or 0.5
                } else {
                    (clamped_val - self.mins[j]) / range
                }
            }).collect()
        }).collect()
    }
}

// Calculates the sigmoid (logistic) function: 1.0 / (1.0 + exp(-z)).
// The sigmoid function is used to squish the output of the dot product of the weights and features
// into a range between 0 and 1, which is useful for classifying a sample as either positive or negative.
pub fn sigmoid(z: f32) -> f32 {
    1.0 / (1.0 + (-z).exp())
}

// Predicts the probability of the positive class (1) for given features and weights.
// Calculates the dot product and applies the sigmoid function.
pub fn predict_probability(features: &Vec<f32>, weights: &Vec<f32>) -> f32 {
    // Calculate z = w0 + w1*x1 + w2*x2 + ... + wn*xn
    let mut z = weights[0]; // Start with bias term (w0)
    
    // Add each weighted feature
    for i in 0..features.len() {
        let feature_value = features[i];
        let weight = weights[i + 1];
        z += feature_value * weight;
    }
    
    // Apply sigmoid function to get probability
    sigmoid(z)
}

// Predicts the binary class label (0.0 or 1.0) by thresholding the predicted probability at 0.5.
pub fn predict_class(features: &Vec<f32>, weights: &Vec<f32>) -> f32 {
    let probability = predict_probability(features, weights);
    
    // Return 1.0 if probability >= 0.5, otherwise 0.0
    if probability >= 0.5 {
        1.0
    } else {
        0.0
    }
}

// Trains a logistic regression model using batch gradient descent  
// -> predict the probability of rain based on weather data.
// Initializes weights to zero and iteratively updates them based on the average gradient over the training data.
pub fn train_logistic_regression(x_train: &Vec<Vec<f32>>, y_train: &Vec<f32>, learning_rate: f32, epochs: usize) -> Vec<f32> {
    // Check if the training data is valid
    if x_train.is_empty() {
        panic!("Training data is empty!");
    }
    
    if x_train.len() != y_train.len() {
        panic!("Number of training samples and target values don't match!");
    }
    
    // Get the number of features and samples
    let number_of_features = x_train[0].len();
    let number_of_samples = x_train.len();
    let number_of_samples_f32 = number_of_samples as f32;
    
    // Create a vector to store our weights (including bias)
    let mut model_weights = Vec::new();
    
    // Initialize all weights to zero (including bias term at index 0)
    for _i in 0..(number_of_features + 1) {
        model_weights.push(0.0);
    }
    
    // Main training loop - run for the specified number of epochs
    let mut epoch_counter = 0;
    while epoch_counter < epochs {
        // Create a vector to store our gradients
        let mut gradient_values = Vec::new();
        
        // Initialize all gradients to zero
        for _i in 0..(number_of_features + 1) {
            gradient_values.push(0.0);
        }
        
        // Loop through each training sample
        for sample_index in 0..number_of_samples {
            // Get the current sample's features and actual target value
            let current_features = &x_train[sample_index];
            let actual_value = y_train[sample_index];
            
            // Make a prediction using current weights
            let predicted_value = predict_probability(current_features, &model_weights);
            
            // Calculate the error (difference between prediction and actual)
            let error_value = predicted_value - actual_value;
            
            // Update the gradient for the bias term
            gradient_values[0] = gradient_values[0] + error_value;
            
            // Update the gradients for each feature
            for feature_index in 0..number_of_features {
                let feature_value = current_features[feature_index];
                gradient_values[feature_index + 1] = gradient_values[feature_index + 1] + (error_value * feature_value);
            }
        }
        
        // Update all weights using the calculated gradients
        for weight_index in 0..model_weights.len() {
            // Calculate the average gradient for this weight
            let average_gradient = gradient_values[weight_index] / number_of_samples_f32;
            
            // Update the weight using gradient descent
            model_weights[weight_index] = model_weights[weight_index] - (learning_rate * average_gradient);
        }
        
        // Increment the epoch counter
        epoch_counter = epoch_counter + 1;
    }
    
    // Return the trained weights
    return model_weights;
}
