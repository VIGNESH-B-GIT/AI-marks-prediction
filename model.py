import numpy as np
from sklearn.linear_model import LinearRegression
import pickle

# Data - only hours (1 feature)
hours = np.array([1,2,3,4,5,6,7,8]).reshape(-1,1)
marks = np.array([35,40,50,55,65,70,75,85])

# Train model
model = LinearRegression()
model.fit(hours, marks)

# Save model
pickle.dump(model, open("model.pkl", "wb"))
print("Model trained and saved!")