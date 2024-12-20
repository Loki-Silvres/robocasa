import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load CSV data
file_path = "/home/loki/robocasa/robocasa/temp/log.csv"  # Replace with your file path
data = pd.read_csv(file_path).iloc[:90]

# Print the first few rows of the data to confirm structure
print("Data Preview:")
print(data.head())

if 'Unnamed: 0' in data.columns:  # Common when index is saved as a column
    data = data.drop(columns=['Unnamed: 0'])
# Calculate the correlation matrix
correlation_matrix = data.corr()

# Display the correlation matrix
print("\nCorrelation Matrix:")
print(correlation_matrix)

# Plot the correlation matrix as a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)

# Title and adjustments
plt.title("Correlation Matrix Heatmap")
plt.tight_layout()

# Show the plot
plt.show()
