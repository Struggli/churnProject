import kaggle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# Using Kaggle's API to get the data file we require
# kaggle.api.authenticate()
# kaggle.api.dataset_download_files('blastchar/telco-customer-churn', path=".", unzip=True)

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Data inspection and cleaning

# Descriptives
# print(df.shape)
# print(df.isnull().sum())

# Considering there aren't any missing values, we drop Customer ID as we're using aggregate data
df.drop(['customerID'], axis=1, inplace=True)

# Since there are no null values, we only need to check for values that aren't the data type mentioned
df['Total Charges'] = pd.to_numeric(df.TotalCharges, errors="coerce")

# This gives us 11 rows that have NaN as their total charges
# print(df[np.isnan(df['Total Charges'])])

# Considering the size of the dataset dropping these rows will have no impact on the outcome
df.dropna(subset=['Total Charges'], inplace=True)

# Analysis
value_counts = df['InternetService'].value_counts(normalize=True) * 100
print("Internet use insights: \n")
print(f"Most of our clients use: {value_counts.idxmax()}")
print(f"{round(value_counts.min())} % of our customers do not use our internet service")

# Pie chart to visualize internet use trends in our customer base
plt.figure(figsize=(6, 6))
plt.pie(
    value_counts,
    labels=value_counts.index,
    autopct="%.2f%%",
    startangle=90
)
plt.title("Internet use distribution")
#plt.show()

# We pivot into looking at churn
churn_value_counts = df['Churn'].value_counts(normalize=True) * 100
plt.figure(figsize=(6, 6))
plt.pie(
    churn_value_counts,
    labels=churn_value_counts.index,
    autopct="%.2f%%"
)
plt.title("Churn")
#plt.show()

# Factors that might influence better churn behaviour in consumers, we consider contract types, payment methods, and tenure

churn_factor_1 = px.histogram(data_frame=df, x="Churn", color="Contract", barmode="group",
                              title="Distribution of contracts for Churn")
# churn_factor_1.show()
churn_factor_2 = px.histogram(data_frame=df, x="Churn", color="PaymentMethod", barmode="group", title="Distribution "
                                                                                                      "of payment "
                                                                                                      "types for "
                                                                                                      "Churn")
# churn_factor_2.show()
# While we're able to provide descriptive insights into our data, we can leverage this data to get predictive insights
# as well, for example can we predict if a customer will be lost to churn based off the factors we've measured?
# Considering the binary nature of churn (Yes/No) Logistic regression is being selected to help predict
# future consumer behaviour

data_frame_for_prediction = df

# We encode the dependent variable
data_frame_for_prediction['Churn'] = LabelEncoder().fit_transform(data_frame_for_prediction['Churn'])
data_frame_for_prediction = pd.get_dummies(data_frame_for_prediction, drop_first=True)

# Dependent variable
Y = data_frame_for_prediction['Churn']

# Independent variables
X = data_frame_for_prediction.drop('Churn', axis=1)

# Splitting data into training and test halves
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=7)

# Considering the wide range of values we scale the data
data_scaler = StandardScaler()
X_train = data_scaler.fit_transform(X_train)
X_test = data_scaler.transform(X_test)

predictive_model = LogisticRegression()
predictive_model.fit(X_train, Y_train)

churn_prediction = predictive_model.predict(X_test)
churn_probability_prediction = predictive_model.predict_proba(X_test)[:, 1]

print(f'Accuracy of the predictive model: {classification_report(y_true=Y_test, y_pred=churn_prediction)}')
