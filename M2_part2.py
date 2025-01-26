import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the dataset
file_path = "data/iris_data.csv"
df = pd.read_csv(file_path)

# Assuming 'species' column needs encoding
label_encoder = LabelEncoder()
df.iloc[:, -1] = label_encoder.fit_transform(df.iloc[:, -1])

# One-hot encode the last column
df = pd.get_dummies(df, columns=[df.columns[-1]], prefix="Species")

# Save without header
df.to_csv(file_path, index=False, header=False)

