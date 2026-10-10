
import joblib
import pandas as pd

model = joblib.load("model.joblib")

machine_type = input("Enter machine type (L, M, H): ").upper()
air_temperature = float(input("Enter air temperature (K): "))
process_temperature = float(input("Enter process temperature (K): "))
rotational_speed = int(input("Enter rotational speed (rpm): "))
torque = float(input("Enter torque (Nm): "))
tool_wear = int(input("Enter tool wear (min): "))

machine = {
    "type": machine_type,
    "air_temperature": air_temperature,
    "process_temperature": process_temperature,
    "rotational_speed": rotational_speed,
    "torque": torque,
    "tool_wear": tool_wear
}

input_data = pd.DataFrame([machine])

prediction = model.predict(input_data)[0]
probability = model.predict_proba(input_data)[0][1]

print("Model:", type(model.named_steps["classifier"]).__name__)

print("\nPredicted result:",
      "Failure" if prediction == 1 else "No failure")
print(f"Model-estimated failure score: {probability:.2%}")
