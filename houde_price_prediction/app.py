
import pandas as pd
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

app = Flask(__name__)

# 1. Load the dataset
df = pd.read_csv("house_data.csv")

# Check that the required columns exist
required_columns = ["Area", "Bedrooms", "Bathrooms", "Price"]

if not all(column in df.columns for column in required_columns):
    raise ValueError(
        "house_data.csv must contain: Area, Bedrooms, Bathrooms, Price"
    )

# Remove rows with missing values in required columns
df = df[required_columns].dropna()

if len(df) < 5:
    raise ValueError(
        "At least 5 complete rows are needed for this example."
    )

# 2. Select input features and target
X = df[["Area", "Bedrooms", "Bathrooms"]]
y = df["Price"]

# 3. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# 4. Train a model using only the training data
evaluation_model = LinearRegression()
evaluation_model.fit(X_train, y_train)

# 5. Predict prices for the unseen testing data
y_pred = evaluation_model.predict(X_test)

# 6. Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

# 7. Display evaluation results in the terminal
print("\nHOUSE PRICE MODEL EVALUATION")
print("-" * 35)
print(f"Total dataset rows: {len(df)}")
print(f"Training rows: {len(X_train)}")
print(f"Testing rows: {len(X_test)}")
print(f"Mean Absolute Error (MAE): {mae:,.2f}")
print(f"Mean Squared Error (MSE): {mse:,.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:,.2f}")
print(f"R-squared (R2): {r2:.4f}")
print("-" * 35)

# Show actual and predicted prices for the test set
results = pd.DataFrame({
    "Actual Price": y_test,
    "Predicted Price": y_pred
})

print("\nACTUAL VS PREDICTED PRICES")
print(results.round(2).to_string(index=False))

# 8. Train the final website model using all available data
# The testing metrics above still come from data held out during evaluation.
model = LinearRegression()
model.fit(X, y)

print("\nFinal prediction model trained successfully.")


# 9. Website route
@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error = None

    form_data = {
        "area": "",
        "bedrooms": "",
        "bathrooms": ""
    }

    if request.method == "POST":
        form_data["area"] = request.form.get("area", "").strip()
        form_data["bedrooms"] = request.form.get(
            "bedrooms", ""
        ).strip()
        form_data["bathrooms"] = request.form.get(
            "bathrooms", ""
        ).strip()

        try:
            area = float(form_data["area"])
            bedrooms = int(form_data["bedrooms"])
            bathrooms = int(form_data["bathrooms"])

            if area <= 0 or bedrooms <= 0 or bathrooms <= 0:
                raise ValueError(
                    "All values must be greater than zero."
                )

            input_data = pd.DataFrame(
                [[area, bedrooms, bathrooms]],
                columns=["Area", "Bedrooms", "Bathrooms"]
            )

            prediction = float(model.predict(input_data)[0])

        except ValueError:
            error = (
                "Enter a valid positive area, number of bedrooms, "
                "and number of bathrooms."
            )

    return render_template(
        "index.html",
        prediction=prediction,
        error=error,
        form_data=form_data
    )


if __name__ == "__main__":
    app.run(debug=True)
