# Railway Lifecycle Monitoring ML API

This project provides a FastAPI-based machine learning API for monitoring the lifecycle and predicting the failure risk of railway products. The API fetches product data, preprocesses it, trains a Random Forest model, and generates actionable alerts and recommendations.

## Features

- Fetches and preprocesses railway product data from a remote API
- Trains a machine learning model to predict product failure risk
- Provides AI-driven alerts for failure risk, vendor risk, and location risk
- Offers parameter recommendations to reduce failures

## Requirements

- Python 3.8+
- See [requirements.txt](requirements.txt) for dependencies

## Usage

1. Install dependencies:
    ```sh
    pip install -r requirements.txt
    ```

2. Start the API server:
    ```sh
    python test.py
    ```

3. Access the API:
    - Home: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
    - Predict: Send a POST request to `/predict` with optional `productId` or `lotId` in the JSON body.

## Example Predict Request

```json
POST /predict
{
  "productId": "your_product_id"
}
```

## Project Structure

- `test.py`: Main FastAPI application and ML logic
- `requirements.txt`: Python dependencies

## License

MIT License