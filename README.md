This is a Flask-based web application that predicts Consumer Price Index (CPI) using machine learning models.

## Project Structure

```
Consumer_Price_Index_Prediction/
├── Data/
│   └── All_India_Index_july2019_20Aug2020_dec20_2.csv
├── forms/
├── static/
│   └── style.css
├── templates/
│   ├── home.html
│   ├── predict.html
│   └── submit.html
├── Training/
│   ├── CPI_Training.ipynb
│   └── cpi_model.h5
├── app.py
└── requirements.txt
```

## Features

- User-friendly web interface for CPI prediction
- Multiple machine learning models for prediction
- Data visualization and analysis
- Model training and evaluation
- Hyperparameter tuning
- Flask-based deployment

## Setup Instructions

1. Install Python 3.8 or higher
2. Install required packages:
   ```
   pip install -r requirements.txt
   ```
3. Train the model using the notebook:
   ```
   jupyter notebook Training/CPI_Training.ipynb
   ```
4. Run the Flask application:
   ```
   python app.py
   ```

## Usage

1. Open your web browser and navigate to `http://localhost:5000`
2. Click on "Predict CPI" to access the prediction form
3. Enter the required economic parameters
4. Click "Predict CPI" to get the predicted value

## Technologies Used

- Backend: Python, Flask
- Frontend: HTML, CSS
- Machine Learning: TensorFlow, Keras, scikit-learn
- Data Visualization: Matplotlib, Seaborn
