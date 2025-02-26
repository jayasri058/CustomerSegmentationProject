import pandas as pd
from scipy.stats import skew, kurtosis
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import os
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as sch

import pickle


# Load the file into a DataFrame (use pd.read_csv if it's a CSV file)
try:
    # Replace with your file path
    df = pd.read_csv(r"E:/Jaya Lenovo Laptop Backup/Personal/Personal/InternshipWork/Customer Segmentation Project/Dataset/customer_segmentation_large_dataset.csv")
    
    # Check if DataFrame is loaded correctly by printing the first few rows
    print(df.head())  # This will display the first 5 rows of the DataFrame

except Exception as e:
    print(f"Error loading file: {e}")
    df = None  # Ensure that df is defined even if loading fails

# Proceed only if DataFrame is loaded correctly
if df is not None:
    # Select numerical columns
    numerical_columns = df.select_dtypes(include=['number']).columns  # This selects all numerical columns
    
    # Loop through each numerical column and calculate EDA metrics
    for column_name in numerical_columns:
        numerical_column = df[column_name]

        # Calculate mean and median
        mean_value = numerical_column.mean()
        median_value = numerical_column.median()

        # Calculate skewness and kurtosis
        skew_value = skew(numerical_column, nan_policy='omit')  # nan_policy='omit' to handle NaNs
        kurtosis_value = kurtosis(numerical_column, nan_policy='omit')

        # Print results for the current column
        print(f"\n--- EDA for Column: {column_name} ---")
        print(f"Mean: {mean_value}")
        print(f"Median: {median_value}")
        print(f"Skewness: {skew_value}")
        print(f"Kurtosis: {kurtosis_value}")

        # Generate and display summary statistics
        summary = numerical_column.describe()
        print(f"Summary Statistics:\n{summary}")
else:
    print("DataFrame not loaded successfully.")
    
    
df.describe()
df.info()

df.isnull().sum()



# Line Graph: Age vs. Spending Score
plt.figure(figsize=(10, 5))
plt.plot(df.groupby("Age")["Spending_Score"].mean(), marker='o', linestyle='-')
plt.xlabel("Age")
plt.ylabel("Average Spending Score")
plt.title("Spending Score vs. Age")
plt.grid(True)
plt.show()

# Pie Chart: Customer Type Distribution
customer_type_counts = df["Customer_Type"].value_counts()

plt.figure(figsize=(8, 8))
plt.pie(customer_type_counts, labels=customer_type_counts.index, autopct='%1.1f%%', 
        startangle=140, colors=["#ff9999", "#66b3ff", "#99ff99", "#ffcc99"])
plt.title("Customer Type Distribution")
plt.show()



    
plt.figure(figsize=(10, 6))
plt.boxplot([df['Purchases'], df['Total_Amount'], df['Recency'], df['Account_Tenure_Years'], 
             df['Spending_Score'], df['Loyalty_Score']], 
            labels=['Purchases', 'Total_Amount', 'Recency', 'Account_Tenure_Years', 
                    'Spending_Score', 'Loyalty_Score'], patch_artist=True)

# Customizing the plot
plt.title('Box Plot for Multiple Columns')
plt.xlabel('Columns')
plt.ylabel('Values')
plt.grid(axis='y')

# Show the plot
plt.show()



# Drop rows with missing values in 'Frequency' or 'Monetary'
scatter_data = df.dropna(subset=['Frequency', 'Monetary'])

# Create the scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(scatter_data['Frequency'], scatter_data['Monetary'])
plt.title('Scatter Plot of Frequency vs. Monetary')
plt.xlabel('Frequency (number of visits)')
plt.ylabel('Monetary (transaction amount)')
plt.grid(True)
plt.show()

#Outliers Tretment

def treat_outliers_quantile(series, lower_quantile=0.05, upper_quantile=0.95):
    lower_bound = series.quantile(lower_quantile)
    upper_bound = series.quantile(upper_quantile)
    return series.clip(lower=lower_bound, upper=upper_bound)

# Apply the function to all numeric columns
columns_to_treat = df.select_dtypes(include='number').columns  # Select numeric columns
for col in columns_to_treat:
    df[col] = treat_outliers_quantile(df[col])

# Display the modified DataFrame
print("DataFrame after quantile outlier treatment:")
print(df)




# Convert multiple categorical columns using Label Encoding
def encode_categorical_columns(df, columns):
    encoder = LabelEncoder()
    for col in columns:
        df[col] = encoder.fit_transform(df[col])
    return df

# Select categorical columns to encode
categorical_columns = df.select_dtypes(include='object').columns

# Apply encoding
df_encoded = encode_categorical_columns(df.copy(), categorical_columns)

print("DataFrame after Label Encoding:")
print(df_encoded)

def save_encoded_data_to_csv(df_encoded, filepath):
    # Save the DataFrame to a CSV file
    df_encoded.to_csv(filepath, index=False)
    print(f"Encoded data saved to {filepath}")

filepath = "E:/Jaya Lenovo Laptop Backup/Personal/Personal/InternshipWork/Customer Segmentation Project/Dataset/encoded_dataset1.csv"
save_encoded_data_to_csv(df_encoded, filepath)

# Clustering methods and silhouette scores storage
clustering_results = {}

# K-Means Clustering
range_n_clusters = range(2, 11)
kmeans_silhouette_scores = []

for n_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(df_encoded)
    silhouette_avg = silhouette_score(df_encoded, cluster_labels)
    kmeans_silhouette_scores.append(silhouette_avg)

best_kmeans_clusters = range_n_clusters[kmeans_silhouette_scores.index(max(kmeans_silhouette_scores))]
clustering_results['K-Means'] = max(kmeans_silhouette_scores)



# Train the K-Means model with the best number of clusters found
kmeans = KMeans(n_clusters=best_kmeans_clusters, random_state=42)
kmeans.fit(df_encoded)

# Save the K-Means model
with open('E:/Jaya Lenovo Laptop Backup/Personal/Personal/InternshipWork/Customer Segmentation Project/kmeans_model1.pkl', 'wb') as file:
    pickle.dump(kmeans, file)


# Hierarchical Clustering
hierarchical_silhouette_scores = []

for n_clusters in range_n_clusters:
    hc = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    cluster_labels = hc.fit_predict(df_encoded)
    silhouette_avg = silhouette_score(df_encoded, cluster_labels)
    hierarchical_silhouette_scores.append(silhouette_avg)

best_hierarchical_clusters = range_n_clusters[hierarchical_silhouette_scores.index(max(hierarchical_silhouette_scores))]
clustering_results['Hierarchical'] = max(hierarchical_silhouette_scores)

# Fit the Agglomerative Clustering model with the best number of clusters found
hc = AgglomerativeClustering(n_clusters=best_hierarchical_clusters, linkage='ward')
hc_labels = hc.fit_predict(df_encoded)

# Save the Agglomerative Clustering labels (optional)
with open('E:/Jaya Lenovo Laptop Backup/Personal/Personal/InternshipWork/Customer Segmentation Project/hierarchical_clustering_labels1.pkl', 'wb') as file:
    pickle.dump(hc_labels, file)


# DBSCAN Clustering
eps_values = [0.5, 1.0, 1.5, 2.0]
min_samples_values = [5, 10, 15]
best_dbscan_score = -1
best_dbscan_params = (0.5, 5) 

for eps in eps_values:
    for min_samples in min_samples_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(df_encoded)
        
        # Only calculate silhouette score if more than one cluster is formed
        if len(set(cluster_labels)) > 1:
            silhouette_avg = silhouette_score(df_encoded, cluster_labels)
            if silhouette_avg > best_dbscan_score:
                best_dbscan_score = silhouette_avg
                best_dbscan_params = (eps, min_samples)
    

clustering_results['DBSCAN'] = best_dbscan_score

# Fit the DBSCAN model with the best parameters found
dbscan = DBSCAN(eps=best_dbscan_params[0], min_samples=best_dbscan_params[1])
dbscan.fit(df_encoded)

# Save the DBSCAN model
with open('E:/Jaya Lenovo Laptop Backup/Personal/Personal/InternshipWork/Customer Segmentation Project/dbscan_model1.pkl', 'wb') as file:
    pickle.dump(dbscan, file)


# Printing results
print("Clustering Method Comparison:")
for method, score in clustering_results.items():
    print(f"{method}: Silhouette Score = {score:.4f}")

best_method = max(clustering_results, key=clustering_results.get)
print(f"\nBest Clustering Method: {best_method} with Silhouette Score = {clustering_results[best_method]:.4f}")

    

