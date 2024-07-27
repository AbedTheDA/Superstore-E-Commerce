#Import the libraries
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import plotly as py
import plotly as py
import plotly.express as px
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

#file_path = '/content/Clean df.csv'
df = pd.read_csv(file_path)

# Display the first few rows of the dataframe
print(df.head())

print(df.describe())

print(df.info())

print(df.isnull().sum())


# Method 1: Number of missing values in each column
print("Missing values in each column:\n", df.isna().sum())

# Method 2: Check if any missing values exist in each column
print("\nAny missing values in each column:\n", df.isna().any())

# Method 3: Total number of missing values in the entire DataFrame
print("\nTotal number of missing values in the DataFrame:", df.isna().sum().sum())

# Method 4: Displaying rows with any missing values
print("\nRows with any missing values:\n", df[df.isna().any(axis=1)])


# Drop rows with NaN values
df = df.dropna()

# Display the modified DataFrame
print("\nDataFrame after dropping NaN rows:")
print(df)


print(df.isnull().sum())



# Encode categorical 'Status' column
df['Status_encoded'] = df['Status'].astype('category').cat.codes

# Calculate the correlation matrix
corr_matrix = df[['Status_encoded', 'Price']].corr()

# Plot the heatmap
plt.figure(figsize=(8, 6))
heatmap = sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, cbar=True)
plt.title('Correlation Matrix Heatmap', fontsize=16)
plt.xlabel('Variables', fontsize=12)
plt.ylabel('Variables', fontsize=12)

# Customize the color bar legend
colorbar = heatmap.collections[0].colorbar
colorbar.set_label('Correlation Coefficient', fontsize=12)

plt.show()



# Encode categorical 'Status' column
df['Status_encoded'] = df['Status'].astype('category').cat.codes

# Calculate the correlation matrix
corr_matrix = df[['Status_encoded', 'grand_total']].corr()

# Plot the heatmap
plt.figure(figsize=(8, 6))
heatmap = sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, cbar=True)
plt.title('Correlation Matrix Heatmap for Status and Grand Total', fontsize=16)
plt.xlabel('Variables', fontsize=12)
plt.ylabel('Variables', fontsize=12)

# Customize the color bar legend
colorbar = heatmap.collections[0].colorbar
colorbar.set_label('Correlation Coefficient', fontsize=12)

plt.show()



# Encode categorical 'Status' column
df['Status_encoded'] = df['Status'].astype('category').cat.codes

# Calculate the correlation matrix
corr_matrix = df[['Status_encoded', 'Quantity']].corr()

# Plot the heatmap
plt.figure(figsize=(8, 6))
heatmap = sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, cbar=True)
plt.title('Correlation Matrix Heatmap for Status and Quantity', fontsize=16)
plt.xlabel('Variables', fontsize=12)
plt.ylabel('Variables', fontsize=12)

# Customize the color bar legend
colorbar = heatmap.collections[0].colorbar
colorbar.set_label('Correlation Coefficient', fontsize=12)

plt.show()



# Encode categorical 'Status' and 'Category' columns
df['Status_encoded'] = df['Status'].astype('category').cat.codes
df['Category_encoded'] = df['Category'].astype('category').cat.codes

# Calculate the correlation matrix
corr_matrix = df[['Status_encoded', 'Category_encoded']].corr()

# Plot the heatmap
plt.figure(figsize=(8, 6))
heatmap = sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, cbar=True)
plt.title('Correlation Matrix Heatmap for Status and Category', fontsize=16)
plt.xlabel('Variables', fontsize=12)
plt.ylabel('Variables', fontsize=12)

# Customize the color bar legend
colorbar = heatmap.collections[0].colorbar
colorbar.set_label('Correlation Coefficient', fontsize=12)

plt.show()



# Encode categorical 'Status' and 'Payment_Method' columns
df['Status_encoded'] = df['Status'].astype('category').cat.codes
df['Payment_Method_encoded'] = df['Payment_Method'].astype('category').cat.codes

# Calculate the correlation matrix
corr_matrix = df[['Status_encoded', 'Payment_Method_encoded']].corr()

# Plot the heatmap
plt.figure(figsize=(8, 6))
heatmap = sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, cbar=True)
plt.title('Correlation Matrix Heatmap for Status and Payment Method', fontsize=16)
plt.xlabel('Variables', fontsize=12)
plt.ylabel('Variables', fontsize=12)

# Customize the color bar legend
colorbar = heatmap.collections[0].colorbar
colorbar.set_label('Correlation Coefficient', fontsize=12)

plt.show()



#Analysis Part":


# Group by 'Category' and sum the 'qty_ordered' for each category
best_selling_category = df.groupby('Category')['Quantity'].sum().idxmax()

# Print the best-selling category
print(f"The best-selling category is: {best_selling_category}")

# Create a scatter plot with Matplotlib
plt.figure(figsize=(15, 10))  # Use reasonable size for display

# Iterate through each unique category and plot
categories = df['Category'].unique()
for category in categories:
    subset = df[df['Category'] == category]
    plt.scatter(subset['Price'], subset['Quantity'], label=category, s=100)  # Added size for better visibility

plt.xlabel('Price')
plt.ylabel('Quantity Ordered')
plt.title('Scatter plot of Price vs Quantity Ordered by Category')
plt.legend(title='Category')
plt.grid(True)  # Added grid for better readability
plt.show()




# Count the number of occurrences for each unique 'Status'
status_counts = df['Status'].value_counts()

print(status_counts)



df['Payment_Method'].value_counts()




# Create a contingency table
contingency_table = pd.crosstab(df['Payment_Method'], df['Status'])

# Plot the heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(contingency_table, annot=True, cmap="YlGnBu", fmt="d")
plt.title('Contingency Table: Payment Method vs Order Status')
plt.xlabel('Status')
plt.ylabel('Payment Method')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.show()




# Create a contingency table
contingency_table = pd.crosstab(df['Month'], df['Category'])

# Plot the heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(contingency_table, annot=True, cmap="YlGnBu", fmt="d")
plt.title('Contingency Table: Order Month vs Item Category')
plt.xlabel('Item Category')
plt.ylabel('Order Month')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.show()




Payment_counts = df['Payment_Method'].value_counts()

# Sort the payments based on their counts in descending order
Payment_counts_sorted = Payment_counts.sort_values(ascending=False)

# Plot a countplot showing payments sold
plt.figure(figsize=(10, 5))  # Set the figure size
sns.countplot(data=df, x='Payment_Method', order=Payment_counts_sorted.index, palette='viridis')  # Create the countplot with sorted order
plt.title('Payment Method Used')  # Add title to the plot
plt.xlabel('Payment Method')  # Add label to x-axis
plt.ylabel('No of Sales')  # Add label to y-axis
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
plt.tight_layout()  # Adjust layout to prevent clipping of labels
plt.show()  # Show the plot





df['Category'].value_counts()



set_size_style(9,6,'darkgrid')
ax = sns.countplot(df[df['Status'] != 'others'], x='Category', hue='Status')
ax.tick_params('x', rotation=90)

customize_plot(ax, 'Product category\'s order count with status', 'Category', 'Orders count',12,10)




df['Month'].value_counts()

df['Category'].value_counts()


# Groupby month and count orders
monthly_counts = df.groupby('Month')['Item_ID'].count()

# Plot pie chart
plt.pie(monthly_counts, labels=monthly_counts.index, autopct='%1.1f%%')

# Add title and labels
plt.title('Orders by Month')
plt.ylabel('Percentage of Total Orders')

plt.show()



# Groupby month and count orders
monthly_counts = df.groupby('Year')['Item_ID'].count()

# Plot pie chart
plt.pie(monthly_counts, labels=monthly_counts.index, autopct='%1.1f%%')

# Add title and labels
plt.title('Orders by Year')
plt.ylabel('Percentage of Total Orders')

plt.show()



df['Customer ID'].value_counts()



# Count the number of purchases made by each customer
customer_purchases = df['Customer ID'].value_counts()

# Define criteria for categorizing customers
new_customers = customer_purchases[customer_purchases == 1].count()
repeated_customers = customer_purchases[customer_purchases > 1].count()
old_customers = len(df['Customer ID'].unique()) - new_customers - repeated_customers

# Plot the bar plot with red and blue colors
plt.figure(figsize=(8, 6))
plt.bar(['New Customers', 'Old Customers', 'Repeated Customers'], [new_customers, old_customers, repeated_customers], color=['red', 'blue', 'blue'])
plt.title('Distribution of New, Old, and Repeated Customers')
plt.xlabel('Customer Type')
plt.ylabel('Count')
plt.show()




# Group by month and sum the sales for each month
monthly_sales = df.groupby('Month')['Quantity'].sum()

# Plot the line graph of sales for each month
plt.figure(figsize=(10, 6))  # Set the figure size
monthly_sales.plot(marker='o', linestyle='-')  # Plot the line graph
plt.title('Sales Trend over Months (Discrete Form)')  # Add title to the plot
plt.xlabel('Month')  # Add label to x-axis
plt.ylabel('Sales')  # Add label to y-axis
plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])  # Set x-axis ticks
plt.tight_layout()  # Adjust layout to prevent clipping of labels
plt.grid(True)  # Add grid
plt.show()  # Show the
