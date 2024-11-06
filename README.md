A data analysis projec involving telecom customer data. 

Toolkit used: Python for data visualization, and predictive analysis
Packages used: pandas, numPy, pyplot, matplotlib, scikit-learn

Written Analysis:
Customer retention needs to be a priority for any organization whose bottomline is funded by repeat customers. 
With customer retention, we have a higher probability rate (60-70%) of selling any additional services they may require and we may have as opposed to enticing new customers (2-10%). 
The premise of this project is I am a data analyst for the internet services segment of the telecom provider and I am offering insights on what factors can be influenced in order to better retain a customer.
We begin by looking at usage statistics for internet:
![Internet use statistics](https://github.com/user-attachments/assets/80041256-c4a9-423f-92a2-c188ab290038)

We then begin by pivoting into looking at churn with two factors I've selected as factors that the company can influence by incentivizing alternatives that have shown better retention statistics.
For example, by providing a discount or other financial incentives for customers to switch from a month-to-month contract to a one-year contract decreases churn by almost 80%. 
![Churn factor](https://github.com/user-attachments/assets/59d52d58-b752-410d-97be-d3eb162ba291)

Similarly, if clients were offered a financial incentive to pay using their credit cards, this could be a concerted effort with a selected financial partner to drive traffic towards us and reduce churn.
![Churn factor 2](https://github.com/user-attachments/assets/079e2956-5d00-49f3-9292-5062276ead34)

Finally, considering the binary nature of churn (either it's a yes or no), we use a logistic regression model to predict future customer churn. The model's fit and accuracy statistics are provided below:
Accuracy of the predictive model:               
                precision    recall  f1-score   support
           0       0.82      0.90      0.86      2076
           1       0.62      0.45      0.53       737

    accuracy                           0.78      2813
   macro avg       0.72      0.68      0.69      2813
weighted avg       0.77      0.78      0.77      2813
