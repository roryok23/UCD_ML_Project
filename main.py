import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.stats import randint

# Importing the dataset
nba_data = pd.read_csv('Nba_stats/Nba_21_24_stats.csv', encoding='ISO-8859-1')

# Checking the initial data
print(nba_data.head())
print(nba_data.columns)


# Remove rows where the Team column contains '2TM' - this signifies that they have played for more
#than one team in the course of a season and should not be included for averaging purposes

nba_data = nba_data[nba_data['Team'] != '2TM']  # Overwrite the original dataset


# Checking for missing values and handling them
nba_data['PTS'] = nba_data['PTS'].fillna(nba_data['PTS'].mean())

# Fill missing values for numeric columns only
nba_data.fillna(nba_data.select_dtypes(include='number').mean(), inplace=True)
#TESTING to include more features including defensive stats such as steals and defensive rebounds
# Calculating averages for players with duplicate entries
#average_columns = ['PTS', 'MP', 'TRB', 'AST', 'FGA', 'STL', 'G', '3P', '3PA', '3P%', 'DRB']  # Columns to average
#nba_data = nba_data.groupby('Player', as_index=False)[average_columns].mean()

# Calculating averages for players with duplicate entries
average_columns = ['PTS', 'MP', 'TRB', 'AST', 'FGA', 'G', '3P', '3PA', '3P%']  # Columns to average
nba_data = nba_data.groupby('Player', as_index=False)[average_columns].mean()

# Filtering players with less than 10 games played.

nba_data = nba_data[nba_data['G'] >= 10]








# Adding Efficiency Metrics
nba_data['PTS_per_MP'] = nba_data['PTS'] / nba_data['MP']  # Points per minute played
nba_data['AST_to_PTS_ratio'] = nba_data['AST'] / nba_data['PTS']  # Assists-to-points ratio
nba_data['FGA_efficiency'] = nba_data['PTS'] / nba_data['FGA']  # Points per field goal attempt


# Unsupervised Learning Section
# Features for unsupervised learning
unsupervised_features = ['PTS', 'MP', 'TRB', 'AST', 'FGA', 'PTS_per_MP', 'FGA_efficiency', '3P', '3PA', '3P%']

# Applying PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(nba_data[unsupervised_features])

# Adding PCA components back to the DataFrame
nba_data['PCA1'] = pca_result[:, 0]
nba_data['PCA2'] = pca_result[:, 1]

# K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
nba_data['Cluster'] = kmeans.fit_predict(nba_data[unsupervised_features])

# Visualizing Clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x='PCA1',
    y='PCA2',
    hue='Cluster',
    data=nba_data,
    palette='Set1',
    legend='full'
)
plt.title('Clusters Visualized using PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')




# Defining the feature set and target variable
# I chose these features from my domain knowledge of basketball
# Feature Legend:
# MP: Minutes Played
# TRB: Total Rebounds
# AST: Assists
# FGA: Field Goals Attempted
# STL: Steals
# PTS_per_MP: Points Scored Per Minute Played (Scoring Efficiency)
# FGA_efficiency: Points Scored Per Field Goal Attempt (Shooting Efficiency)
#features = ['MP', 'TRB', 'AST', 'FGA', 'STL', 'PTS_per_MP', 'FGA_efficiency', '3P', '3PA', '3P%', 'DRB']
#X = nba_data[features]
#y = nba_data['PTS']


# FGA_efficiency: Points Scored Per Field Goal Attempt (Shooting Efficiency)
features = ['MP', 'TRB', 'AST', 'FGA', 'PTS_per_MP', 'FGA_efficiency', '3P', '3PA', '3P%']
X = nba_data[features]
y = nba_data['PTS']


# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression Model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Making predictions with Linear Regression
y_pred_linear = linear_model.predict(X_test)

# Evaluating the Linear Regression model
mae_linear = mean_absolute_error(y_test, y_pred_linear)
r2_linear = linear_model.score(X_test, y_test)

# Random Forest Regressor Model with Hyperparameter Tuning

# Set up the hyperparameters for RandomizedSearchCV
param_dist = {
    # Number of trees in the forest. Testing a range of 50 to 200 trees
    # to balance model performance with training time.
    'n_estimators': randint(50, 200),
    # Maximum depth of each decision tree. Testing None (no limit) and specific depths (10, 20, 30)
    # to control the complexity of the model and avoid overfitting.
    'max_depth': [None, 10, 20, 30],
    # Minimum number of samples required to split an internal node. Testing values of 2, 5, and 10
    # to determine the minimum data size needed to create a split, impacting model granularity.
    'min_samples_split': [2, 5, 10],
    # Minimum number of samples required to be at a leaf node. Testing values of 1, 2, and 4
    # to control the size of leaf nodes and avoid creating overly small trees.

    'min_samples_leaf': [1, 2, 4],

    # Whether bootstrap samples are used when building trees. Testing both True and False
    # to see if sampling with replacement improves model performance.
    'bootstrap': [True, False]
}

# Create a RandomForestRegressor object
rf_model = RandomForestRegressor(random_state=42)

# Use RandomizedSearchCV for hyperparameter tuning
# I limited the search to 10 random combinations to enhance code performance
random_search = RandomizedSearchCV(estimator=rf_model, param_distributions=param_dist,
                                   n_iter=10, cv=3, n_jobs=-1, verbose=2, random_state=42)
random_search.fit(X_train, y_train)

# Best parameters found by RandomizedSearchCV
best_params = random_search.best_params_
print("Best Hyperparameters:", best_params)

# Using the best model from RandomizedSearchCV
best_rf_model = random_search.best_estimator_

# Making predictions with the tuned Random Forest
y_pred_rf = best_rf_model.predict(X_test)

# Evaluating the tuned Random Forest model
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = best_rf_model.score(X_test, y_test)

# Feature Importance from the tuned Random Forest
feature_importances = best_rf_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

# Advanced Visualization Section

# 1. Efficiency Insights
plt.figure(figsize=(10, 6))
sns.scatterplot(data=nba_data, x='PTS_per_MP', y='FGA_efficiency', hue='AST', palette='coolwarm')
plt.title('Scoring Efficiency vs Shooting Efficiency (Color-coded by Assists)')
plt.xlabel('Points per Minute Played')
plt.ylabel('Points per Field Goal Attempt')

# 2. Assists-to-Points Ratio Distribution
"""  Distribution of Assists-to-Points Ratio (Seaborn)
A histogram showing that most players rely less on assists for scoring, with an AST-to-PTS ratio below 0.3. (Graph 1)
"""
plt.figure(figsize=(10, 6))
sns.histplot(nba_data['AST_to_PTS_ratio'], kde=True, bins=30)
plt.title('Distribution of Assists-to-Points Ratio')
plt.xlabel('Assists-to-Points Ratio')
plt.ylabel('Frequency')

# 3. Feature Importance Bar Plot
plt.figure(figsize=(10, 6))
sns.barplot(
    x='Importance',
    y='Feature',
    data=feature_importance_df,
    hue='Feature',
    palette='viridis',
    dodge=False  # Prevents separation of bars
)

plt.title('Feature Importance (Random Forest)')
plt.xlabel('Importance')
plt.ylabel('Feature')

# 4. Identifying High-Impact Players (Scatter Plot)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=nba_data, x='PTS', y='PTS_per_MP', hue='FGA_efficiency', palette='coolwarm')
plt.title('High Scorers vs Efficiency')
plt.xlabel('Average Points Scored')
plt.ylabel('Points per Minute Played')

# Matplotlib Visualizations
# 5. Horizontal Bar Plot for Assists 
# Sort the dataset by Assists (AST) in descending order
top_assist_players = nba_data.sort_values(by='AST', ascending=False).head(10)

# Horizontal Bar Plot for Top 10 Players by Assists
plt.figure(figsize=(10, 6))
plt.barh(top_assist_players['Player'], top_assist_players['AST'], color='skyblue')
plt.title('Top 10 Players by Average Assists')
plt.xlabel('Average Assists')
plt.ylabel('Player')
plt.gca().invert_yaxis()
# plt.show()


# 6. Box Plot for Minutes Played
plt.figure(figsize=(10, 6))
sns.boxplot(x=nba_data['MP'])
plt.title('Distribution of Minutes Played')
plt.xlabel('Minutes Played')


# 7. Correlation Heatmap

"""Correlation Heatmap (Seaborn)
Highlights strong correlations between minutes played (MP) and points scored (PTS), and between field goals attempted (FGA) and PTS. (Graph 2)
"""
plt.figure(figsize=(10, 6))
correlation_matrix = nba_data[['PTS', 'MP', 'TRB', 'AST', 'FGA', 'PTS_per_MP', 'FGA_efficiency', 'AST_to_PTS_ratio']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix for NBA Metrics')


# 8. Line Chart for Points per Game Trend (Top 10 Players by Points)
top_players_by_points = nba_data.sort_values(by='PTS', ascending=False).head(10)
plt.figure(figsize=(10, 6))
for player in top_players_by_points['Player']:
    player_data = nba_data[nba_data['Player'] == player]
    plt.plot(player_data['G'], player_data['PTS'], marker='o', label=player)

plt.title('Scoring Trends for Top 10 Players by Points')
plt.xlabel('Games Played')
plt.ylabel('Points Scored')
plt.legend(title="Player", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

"""# Scatter plot showing the relationship between 3P% and Points Scored (PTS)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=nba_data, x='3P%', y='PTS', hue='3PA', size='3PA', palette='coolwarm', sizes=(20, 200))
plt.title('Impact of 3P% on Points Scored (Color-coded by 3-Point Attempts)', fontsize=14)
plt.xlabel('Three-Point Percentage (3P%)', fontsize=12)
plt.ylabel('Points Scored (PTS)', fontsize=12)
plt.legend(title='3PA (3-Point Attempts)', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()"""

"""
# Box plot showing the distribution of PTS for different ranges of DRB
nba_data['DRB_range'] = pd.cut(nba_data['DRB'], bins=[0, 2, 5, 10, 15, 20], labels=['0-2', '3-5', '6-10', '11-15', '16+'])

plt.figure(figsize=(10, 6))
sns.boxplot(data=nba_data, x='DRB_range', y='PTS', palette='Blues')
plt.title('Defensive Rebounds vs Points Scored', fontsize=14)
plt.xlabel('Defensive Rebounds (DRB Range)', fontsize=12)
plt.ylabel('Points Scored (PTS)', fontsize=12)
plt.tight_layout()
"""








# Evaluation Results
print(f'Linear Regression Mean Absolute Error: {mae_linear}')
print(f'Linear Regression R²: {r2_linear}')
print(f'Tuned Random Forest Mean Absolute Error: {mae_rf}')
print(f'Tuned Random Forest R²: {r2_rf}')


# Show all plots
plt.show()


