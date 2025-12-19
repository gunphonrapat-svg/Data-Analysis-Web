# import streamlit as st
# import pandas as pd

# st.write("hello world")
# name = st.text_input("what your name : ")

# st.write(f"Hello {name}")

# if st.button("click mi"):
#     st.write("you click me")

# df = pd.read_csv("sustainable_waste_management_dataset_2024.csv")

# st.write(df)

# from numpy.random import default_rng as rng

# df = pd.DataFrame(rng(0).standard_normal((20,3)), columns=["a","b","c"])
# st.bar_chart(df)

# st.line_chart(df)



# option = st.selectbox("which major do you like best?",["CO","CI"])

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------
# Title
# -----------------------------
st.title("Waste Management Prediction App")

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv("sustainable_waste_management_dataset_2024.csv")
st.subheader("Raw Dataset")
st.dataframe(df)

# -----------------------------
# Feature selection
# -----------------------------
features = [
    'population',
    'recyclable_kg',
    'organic_kg',
    'collection_capacity_kg',
    'is_weekend',
    'is_holiday',
    'recycling_campaign',
    'temp_c',
    'rain_mm'
]

X = df[features]
y = df['waste_kg']

data = pd.concat([X, y], axis=1)
data = data.dropna()

X = data[features]
y = data['waste_kg']

# -----------------------------
# Train model
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# -----------------------------
# Metrics
# -----------------------------
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.write(f"**MSE:** {mse:,.2f}")
st.write(f"**R²:** {r2:.4f}")

# -----------------------------
# Graph 1: Predicted vs Actual
# -----------------------------
st.subheader("Predicted vs Actual Waste")

fig1, ax1 = plt.subplots()
ax1.scatter(y_test, y_pred, alpha=0.6)
ax1.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--'
)
ax1.set_xlabel("Actual Waste (kg)")
ax1.set_ylabel("Predicted Waste (kg)")

st.pyplot(fig1)




st.subheader("Actual vs Predicted Waste (Line Chart)")

# สร้าง DataFrame สำหรับ plot
result_df = pd.DataFrame({
    "Actual Waste (kg)": y_test.values,
    "Predicted Waste (kg)": y_pred
}).reset_index(drop=True)

st.line_chart(result_df)


st.subheader("Feature Importance (Matplotlib Line Chart)")

importance = pd.Series(
    model.coef_,
    index=features
).sort_values(ascending=False)

fig2, ax2 = plt.subplots()
ax2.plot(importance.values, marker='o')
ax2.set_xticks(range(len(importance)))
ax2.set_xticklabels(importance.index, rotation=45)
ax2.set_ylabel("Coefficient Value")
ax2.set_xlabel("Feature")

st.pyplot(fig2)
