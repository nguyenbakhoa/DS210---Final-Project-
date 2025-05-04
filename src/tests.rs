#[cfg(test)]
mod tests {
    // Here are the functions and structs I want to test 
    use crate::model::{sigmoid, predict_class, MinMaxScaler};

    // Test 1: Test the sigmoid function
    #[test]
    fn test_sigmoid() {
        // Test case 1: sigmoid(0) should be exactly 0.5
        let result_at_zero = sigmoid(0.0);
        assert_eq!(result_at_zero, 0.5, "Sigmoid at 0 failed");
        
        // Test case 2: sigmoid(large positive) should be close to 1
        let result_at_positive = sigmoid(100.0);
        assert!(result_at_positive > 0.999, "Sigmoid large positive failed");
        
        // Test case 3: sigmoid(large negative) should be close to 0
        let result_at_negative = sigmoid(-100.0);
        assert!(result_at_negative < 0.001, "Sigmoid large negative failed");
    }

    // Test 2: Test predict_class with simple inputs
    #[test]
    fn test_predict_class() {
        // Test case 1: Expected class 1
        let features1 = vec![2.0, 3.0]; 
        let weights1 = vec![0.1, 0.5, -0.2]; // z = 0.1 + 1.0 - 0.6 = 0.5 -> sigmoid > 0.5
        let prediction1 = predict_class(&features1, &weights1);
        assert_eq!(prediction1, 1.0, "Predict class 1 failed");

        // Test case 2: Expected class 0
        let features2 = vec![1.0, 1.0];
        let weights2 = vec![-0.5, 0.1, 0.1]; // z = -0.5 + 0.1 + 0.1 = -0.3 -> sigmoid < 0.5
        let prediction2 = predict_class(&features2, &weights2);
        assert_eq!(prediction2, 0.0, "Predict class 0 failed");
    }

    // Test 3: MinMaxScaler fit and transform
    #[test]
    fn test_min_max_scaler() {
        // Create a new scaler
        let mut scaler = MinMaxScaler::new();
        
        // Create test data with known min and max values
        let test_data = vec![
            vec![1.0, 10.0], // Min values
            vec![3.0, 30.0]  // Max values
        ]; 
        
        // Fit the scaler to learn min and max values
        scaler.fit(&test_data);
        
        // Transform the data to scale it
        let scaled_data = scaler.transform(&test_data);
        
        // Expected: [[0.0, 0.0], [1.0, 1.0]]
        // Check that min values were scaled to 0.0
        assert_eq!(scaled_data[0][0], 0.0, "Scale min failed [0][0]");
        assert_eq!(scaled_data[0][1], 0.0, "Scale min failed [0][1]");
        
        // Check that max values were scaled to 1.0
        assert_eq!(scaled_data[1][0], 1.0, "Scale max failed [1][0]");
        assert_eq!(scaled_data[1][1], 1.0, "Scale max failed [1][1]");
    }
}
