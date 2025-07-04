import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('seaborn')
sns.set_palette('husl')

# Load the data
df = pd.read_csv('Training/CPI_data.csv')

# 1. Univariate Analysis
print("\n=== Univariate Analysis ===")

# Distribution of numerical features
numerical_features = ['Cereals and products', 'Meat and fish', 'Egg', 'Milk and products',
                     'Oils and fats', 'Fruits', 'Vegetables', 'Pulses and products',
                     'Sugar and Confectionery', 'Spices', 'Non-alcoholic beverages',
                     'Prepared meals, snacks, sweets etc.', 'Food and beverages',
                     'Pan, tobacco and intoxicants', 'Clothing', 'Footwear',
                     'Clothing and footwear', 'Fuel and light',
                     'Household goods and services', 'Health',
                     'Transport and communication', 'Recreation and amusement',
                     'Education', 'Personal care and effects', 'Miscellaneous']

# Create histograms for numerical features
plt.figure(figsize=(15, 10))
for i, feature in enumerate(numerical_features, 1):
    plt.subplot(4, 6, i)
    sns.histplot(df[feature], kde=True)
    plt.title(f'Distribution of {feature}')
plt.tight_layout()
plt.show()

# 2. Bivariate Analysis
print("\n=== Bivariate Analysis ===")

# Calculate correlations with target
corr_with_target = df.corr()['General index'].sort_values(ascending=False)
print("\nCorrelations with General index:")
print(corr_with_target)

# Get top 15 features with highest absolute correlation (excluding the target itself)
top_n = 15
features = corr_with_target.drop('General index').abs().sort_values(ascending=False).index[:top_n]
corr_matrix = df[features].corr()

# Create correlation heatmap with better formatting
plt.figure(figsize=(14, 12))
plt.rcParams.update({'font.size': 10})  # Increase font size

# Create a mask to display only the lower triangle
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

# Create the heatmap
heatmap = sns.heatmap(
    corr_matrix, 
    mask=mask,
    cmap='coolwarm', 
    center=0,
    annot=True,  # Show correlation values
    fmt=".2f",    # Format to 2 decimal places
    linewidths=0.5,
    annot_kws={"size": 9}  # Annotation font size
)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.title(f'Top {top_n} Feature Correlations', pad=20, fontsize=14)
plt.tight_layout()

# Save the figure
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# Create a bar plot of top N correlations with the target
plt.figure(figsize=(12, 6))
corr_with_target.drop('General index').head(10).plot(kind='barh', color='skyblue')
plt.title('Top 10 Features Correlated with General Index')
plt.xlabel('Correlation Coefficient')
plt.ylabel('Features')
plt.tight_layout()
plt.savefig('top_correlations.png', dpi=300, bbox_inches='tight')
plt.close()
plt.title('Correlation Heatmap')
plt.show()

# Plot relationships between top correlated features and target
top_features = correlations.abs().sort_values(ascending=False).index[1:6]  # Exclude 'General index' itself
for feature in top_features:
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=feature, y='General index', data=df)
    plt.title(f'{feature} vs General index')
    plt.show()

# 3. Multivariate Analysis
print("\n=== Multivariate Analysis ===")

# Create a pairplot for top correlated features
sns.pairplot(df[top_features.tolist() + ['General index']])
plt.show()

# Create a parallel coordinates plot for different sectors
from pandas.plotting import parallel_coordinates

# Create sector categories
df['Sector'] = pd.Categorical(df['Sector'])

plt.figure(figsize=(12, 8))
parallel_coordinates(df[top_features.tolist() + ['Sector', 'General index']], 'Sector')
plt.title('Parallel Coordinates by Sector')
plt.show()

# 4. Feature Engineering Analysis
print("\n=== Feature Engineering Analysis ===")

# Create interaction features
interaction_features = []
for i in range(len(top_features)):
    for j in range(i+1, len(top_features)):
        new_feature = f'{top_features[i]}_x_{top_features[j]}'
        df[new_feature] = df[top_features[i]] * df[top_features[j]]
        interaction_features.append(new_feature)

# Analyze interaction features
interaction_corrs = df[interaction_features + ['General index']].corr()['General index'].sort_values(ascending=False)
print("\nCorrelations of interaction features with General index:")
print(interaction_corrs)

# 5. Feature Scaling Analysis
print("\n=== Feature Scaling Analysis ===")

# Scale features using StandardScaler
scaler = StandardScaler()
numerical_data = df[numerical_features]
scaled_data = scaler.fit_transform(numerical_data)

# Create boxplots to compare before and after scaling
fig, axes = plt.subplots(1, 2, figsize=(15, 8))

# Before scaling
numerical_data.boxplot(ax=axes[0])
axes[0].set_title('Before Scaling')

# After scaling
pd.DataFrame(scaled_data, columns=numerical_features).boxplot(ax=axes[1])
axes[1].set_title('After Scaling')

plt.tight_layout()
plt.show()

# 6. Feature Importance Analysis
print("\n=== Feature Importance Analysis ===")

# Train a simple model to analyze feature importance
X = df[numerical_features]
y = df['General index']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Get feature importances
importances = pd.DataFrame({
    'Feature': numerical_features,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importances)
plt.title('Feature Importances from Random Forest')
plt.show()

# 7. Model Evaluation Metrics
print("\n=== Model Evaluation Metrics ===")

# Make predictions
y_pred = model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Performance:")
print(f"Mean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")

# Plot predictions vs actual
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted Values')
plt.show()
