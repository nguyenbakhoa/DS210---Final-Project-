// Module: main - Trains and evaluates a logistic regression model.
use std::error::Error;
use rand::seq::SliceRandom; // Import shuffle functionality

mod data;
mod model;
mod tests;

// Entry point: Loads data, shuffles it, scales features, trains a logistic regression model,
// predicts rainfall on a test set, and prints accuracy.
fn main() -> Result<(), Box<dyn Error>> {
    // Parameters & Path Setup
    let file_path = "hanoiweather.csv"; // CSV must be in project root
    let training_ratio = 0.8; // 80% training, 20% testing
    let learning_rate = 0.01; // How quickly the model learns
    let epochs = 1000; // Number of training iterations

    // Load Data from CSV file
    let mut weather_data = data::load_weather_data(&file_path)?;

    // Shuffle Data to avoid bias from data order
    let mut random = rand::rng();
    weather_data.shuffle(&mut random);

    // Split Data into training and testing sets
    let split_index = (weather_data.len() as f32 * training_ratio) as usize; // 80% training, 20% testing
    let (training_set, testing_set) = weather_data.split_at(split_index); 

    // Prepare Feature/Target Vectors
    let mut x_train = Vec::new();
    let mut y_train = Vec::new();
    for row in training_set {
        x_train.push(row.features());
        y_train.push(row.rain as f32);
    } // Prepare training data
    
    let mut x_test = Vec::new();
    let mut y_test = Vec::new();
    for row in testing_set {
        x_test.push(row.features());
        y_test.push(row.rain as f32);
    } // Prepare test data

    // Feature Scaling to normalize values between 0 and 1
    let mut scaler = model::MinMaxScaler::new();
    scaler.fit(&x_train); // Learn min/max from training data
    let x_train_scaled = scaler.transform(&x_train); // Scale training data
    let x_test_scaled = scaler.transform(&x_test);   // Scale test data

    // Train Model using scaled data
    let model_weights = model::train_logistic_regression(
        &x_train_scaled, &y_train, learning_rate, epochs
    );

    // Make Predictions on test data
    let mut correct_count = 0; 
    for i in 0..x_test_scaled.len() {
        let prediction = model::predict_class(&x_test_scaled[i], &model_weights); // Make predictions
        if prediction == y_test[i] {
            correct_count = correct_count + 1;
        }
    } // Count correct predictions

    // Calculate and print accuracy
    let accuracy = (correct_count as f32) / (y_test.len() as f32) * 100.0; // Calculate accuracy
    println!("Test accuracy: {:.2}%", accuracy);

    Ok(())
}
