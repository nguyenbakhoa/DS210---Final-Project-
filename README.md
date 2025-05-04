Used ChatGPT to make this markdown format


# DS210 Project Report

**Name**: Khoa Nguyen  
**Course**: DS210 Project  
**Dataset**: `hanoiweather.csv`  
**Method**: Logistic Regression (built from scratch)

---

## A. Project Overview

- **Goal**: Predict daily rainfall in Hanoi (binary: rain > 0mm or not) using same-day weather features.  
- **Dataset**: `hanoiweather.csv` (~12,500 observations, 32 variables)  
- **Features Used**:  
  - Temperature (`temp`)  
  - Humidity (`humidity`)  
  - Wind Speed (`windspeed`, mapped to `wind`)  
  - Solar Radiation (`solarradiation`, mapped to `sunshine`)  
- **Target Variable**: `rain` (binary `u8`, derived from `precip` in mm)

---

## B. Data Processing

- **Loading**: Used `csv` and `serde` crates to read data  
- **Cleaning & Transformations**:
  - Skipped rows with parsing errors
  - Treated empty numeric fields as `0.0`
  - Derived binary target `rain` from `precip`
  - Extracted selected features only
  - Shuffled data using `rand` crate
  - Applied feature scaling to [0, 1] using a custom `MinMaxScaler`

---

## C. Code Structure

### Modules

- `main.rs`: Orchestrates the overall pipeline  
- `data.rs`: Handles data loading and parsing  
- `model.rs`: Contains Logistic Regression logic and scaler  
- `tests.rs`: Unit tests

### Key Functions & Types

| Function/Type | Description |
|---------------|-------------|
| `data::WeatherRow` | Struct representing a CSV row |
| `data::load_weather_data` | Reads and parses CSV file |
| `model::MinMaxScaler` | Scales feature values |
| `model::sigmoid` | Logistic sigmoid function |
| `model::predict_probability` | Outputs prediction probability |
| `model::predict_class` | Returns predicted binary class |
| `model::train_logistic_regression` | Trains model using Batch Gradient Descent |

### Main Workflow

`Load → Shuffle → Split → Scale → Train → Predict → Evaluate → Print Accuracy`

---

## D. Tests

- **Command**: `cargo test`  
- **Test Coverage**:
  - `test_sigmoid`: Validates sigmoid output
  - `test_predict_class`: Checks prediction thresholding
  - `test_min_max_scaler`: Validates scaling logic  

---

## E. Results

- **Command**: `cargo run`  
- **Test Accuracy**: **55.82%**  
- Interpretation: Slightly better than random guessing (50%)

---

## F. Usage Instructions

- **Build**: `cargo build` or `cargo build --release`  
- **Run**: `cargo run` or `cargo run --release`  
- **Test**: `cargo test`  
- **Arguments**: None  
- **Runtime**: A few seconds

### Jupyter Notebook

- File: `data_facts.ipynb`  
- Usage: Basic data analysis and visualization (e.g., histograms)  
- Run: Execute cells with `Ctrl+Enter` or `Shift+Enter`

---

## G. AI-Assistance Disclosure and Citations

Used **Google Gemini 2.5 Pro** to assist with:

- Understanding Logistic Regression theory  
- Code structure and logic feedback  
- Rust crate usage (`rand`, `csv`, `serde`)  
- Debugging issues  
- Structuring and writing this report

