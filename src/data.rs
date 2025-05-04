use serde::Deserialize;
use std::error::Error;
use std::fs::File;

// Defines data structures and functions for loading weather data.

// Represents a single row of weather data from the CSV.
// Handles potential missing values using Option<f32>.
#[derive(Debug, Deserialize, Clone)] 
pub struct WeatherRow {
    #[serde(rename = "temp")]
    pub temp: Option<f32>,
    #[serde(rename = "humidity")]
    pub humidity: Option<f32>,
    #[serde(rename = "windspeed")]
    pub wind: Option<f32>,
    #[serde(rename = "solarradiation")]
    pub sunshine: Option<f32>,
    #[serde(rename = "precip")]
    pub rain_mm: Option<f32>,
    // Binary indicator for rain (1 if rain_mm > 0, else 0). Set manually after deserialization.
    #[serde(skip_deserializing)] 
    pub rain: u8, 
}

impl WeatherRow {
    // Extracts model features (temp, humidity, wind, sunshine, rain_mm) into a Vec<f32>.
    // Missing values are replaced with 0.0 for simplicity. 
    pub fn features(&self) -> Vec<f32> {
        vec![
            self.temp.unwrap_or(0.0),    
            self.humidity.unwrap_or(0.0), 
            self.wind.unwrap_or(0.0),     
            self.sunshine.unwrap_or(0.0)
        ] 
    }

    // Sets the binary `rain` field (0 or 1) based on the `rain_mm` value -> target variable
    pub fn set_binary_rain(&mut self) {
        self.rain = if self.rain_mm.unwrap_or(0.0) > 0.0 { 1 } else { 0 };
    }
}

// Reads a CSV file, parses rows into WeatherRow structs, and sets the binary rain target.
// Skips rows with parsing errors.
pub fn load_weather_data(file_path: &str) -> Result<Vec<WeatherRow>, Box<dyn Error>> {
    // Open file and create CSV reader with headers
    let file = File::open(file_path)?;
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_reader(file);
    let mut data = Vec::new();
    
    // Process each row, skipping any that fail to parse
    for result in rdr.deserialize::<WeatherRow>() {
        if let Ok(mut rec) = result {
            rec.set_binary_rain(); // Set target variable
            data.push(rec);
        } else {
            // Skip rows that fail to parse and report error
            eprintln!("Skipping row due to parsing error: {}", result.unwrap_err());
        }
    }
    Ok(data)
}
