import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def main():
    print("\n--- E-Commerce Delivery Time Prediction (Linear Regression) ---\n")

    # ---------------------------------
    # PART 1: Create Historical Dataset
    # ---------------------------------
    data = {
        "Distance_km": [2, 5, 8, 10, 12, 15, 18, 20],
        "Packages": [1, 2, 3, 4, 5, 6, 7, 8],
        "DeliveryTime_hr": [1.0, 2.0, 3.2, 4.0, 4.8, 6.0, 7.2, 8.0]
    }

    df = pd.DataFrame(data)
    print("Delivery Dataset:")
    print(df)

    # ---------------------------------
    # PART 2: Separate Input and Output
    # ---------------------------------
    X = df[["Distance_km", "Packages"]]   # input features
    y = df["DeliveryTime_hr"]             # target variable

    # ---------------------------------
    # PART 3: Train-Test Split
    # ---------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # ---------------------------------
    # PART 4: Train Linear Regression Model
    # ---------------------------------
    model = LinearRegression()
    model.fit(X_train, y_train)

    # ---------------------------------
    # PART 5: Predictions
    # ---------------------------------
    predictions = model.predict(X_test)

    # ---------------------------------
    # PART 6: Evaluation Metrics
    # ---------------------------------
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    print("\n2) Evaluation Metrics:")
    print("MAE  (Mean Absolute Error):", round(mae, 2))
    print("MSE  (Mean Squared Error):", round(mse, 2))
    print("RMSE (Root Mean Squared Error):", round(rmse, 2))
    print("R²   (R Square Score):", round(r2, 2))

    # ---------------------------------
    # PART 7: Real-Time Prediction
    # ---------------------------------
    new_order = pd.DataFrame([[5, 2]], columns=["Distance_km", "Packages"])
    predicted_time = model.predict(new_order)

    print("\n3) New Order Prediction:")
    print("Predicted Delivery Time (hours):",
          round(predicted_time[0], 2))

    print("\n--- End of Prediction of estimated order delivery time prediction ---\n")


if __name__ == "__main__":
    main()