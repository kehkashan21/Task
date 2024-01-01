import pandas as pd
from sklearn.datasets import load_iris

# Load Iris dataset
iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
target = pd.DataFrame(data=iris.target, columns=['species'])

# Combine features and target into a single DataFrame
df = pd.concat([data, target], axis=1)
# Summarize key statistics
summary_stats = df.describe()

# Visualize data distributions
import matplotlib.pyplot as plt

for feature in df.columns:
    if feature != 'species':
        df[feature].plot(kind='hist', title=feature)
        plt.show()

# Identify missing values
missing_values = df.isnull().sum()

# Explore relationships between variables
correlation_matrix = df.corr()
# No missing values in the Iris dataset, so no handling needed here

# Encoding categorical variables (if any)
# No categorical variables in Iris dataset, so skipping this step

# Splitting the data into training and testing sets
from sklearn.model_selection import train_test_split

X = df.drop('species', axis=1)
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Building a classification model using RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import seaborn as sns

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

# Evaluating model performance
accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)
roc_auc = roc_auc_score(y_test, clf.predict_proba(X_test), multi_class='ovr')

# Visualizing key evaluation metrics and ROC curve
print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{report}")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"ROC AUC: {roc_auc}")

# Feature importance
feature_importance = clf.feature_importances_
plt.figure(figsize=(8, 6))
sns.barplot(x=feature_importance, y=X.columns)
plt.title('Feature Importance')
plt.show()
# Building a regression model using LinearRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

reg = LinearRegression()
reg.fit(X_train, y_train)
reg_predictions = reg.predict(X_test)

# Evaluating regression model performance
rmse = mean_squared_error(y_test, reg_predictions, squared=False)
r2 = r2_score(y_test, reg_predictions)

# Visualizing predictions and actual values
plt.scatter(y_test, reg_predictions)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Regression Model: Actual vs Predicted')
plt.show()

print(f"RMSE: {rmse}")
print(f"R-squared: {r2}")
