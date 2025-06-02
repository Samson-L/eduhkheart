import pandas as pd

# Load the original and random CSV files
original_df = pd.read_csv('heart_attack_china.csv')
random_df = pd.read_csv('heart_attack_china-80.csv')

# Filter the original DataFrame to only include PassengerId that do not exist in the random DataFrame
filtered_original_df = original_df[~original_df['Patient_ID'].isin(random_df['Patient_ID'])]

# Save the filtered DataFrame to a new CSV file
filtered_original_df.to_csv('heart_attack_china-filtered.csv', index=False)

print("Filtered CSV file has been created: filtered_original.csv")
