import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Number of customers (random between 10,000 to 15,000)
num_customers = np.random.randint(10000, 15001)

# Generate synthetic data
cust_ids = [f'CUST{str(i).zfill(5)}' for i in range(1, num_customers+1)]
genders = np.random.choice(['Male', 'Female'], size=num_customers)
ages = np.random.randint(150, 200, size=num_customers)
annual_incomes = np.random.randint(150000, 300000, size=num_customers)
spending_scores = np.random.randint(150, 200, size=num_customers)
purchases = np.random.randint(150, 200, size=num_customers)
loyalty_scores = np.random.randint(150, 200, size=num_customers)
account_tenure = np.random.randint(150, 200, size=num_customers)
customer_type = np.random.choice(['New', 'Returning', 'VIP'], size=num_customers)
region = np.random.choice(['North', 'South', 'East', 'West'], size=num_customers)
recency = np.random.randint(150, 365, size=num_customers)  # Days since last purchase
frequency = np.random.randint(150, 200, size=num_customers)  # Number of purchases
total_amount = np.random.randint(150, 10000, size=num_customers)  # Total amount spent
monetary = total_amount / frequency  # Average transaction value

# Create a DataFrame
df = pd.DataFrame({
    'Customer_ID': cust_ids,
    'Gender': genders,
    'Age': ages,
    'Annual_Income': annual_incomes,
    'Spending_Score': spending_scores,
    'Purchases': purchases,
    'Loyalty_Score': loyalty_scores,
    'Account_Tenure_Years': account_tenure,
    'Customer_Type': customer_type,
    'Region': region,
    'Recency': recency,
    'Frequency': frequency,
    'Total_Amount': total_amount,
    'Monetary': monetary
})

# Introduce null values randomly (10% for specific columns)
for col in ['Age', 'Annual_Income', 'Spending_Score', 'Loyalty_Score']:
    df.loc[df.sample(frac=0.1).index, col] = np.nan

# Replace null values with 150
df.fillna(150, inplace=True)

# Save to CSV
df.to_csv('E:/Jaya Lenovo Laptop Backup/Personal/Personal/InternshipWork/Customer Segmentation Project/Dataset/customer_segmentation_large_dataset1.csv', index=False)

# Display dataset info and first few rows
print(f"Dataset created with {len(df)} rows.")
print(df.head())
