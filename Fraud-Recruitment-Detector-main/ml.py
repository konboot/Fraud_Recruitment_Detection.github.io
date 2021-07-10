import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.datasets import make_classification
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import confusion_matrix,classification_report
from time import time
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


train_df = pd.read_csv("fake_job_postings.csv")
duplicates = train_df.drop_duplicates(keep="first")
diff = train_df.shape[0] - duplicates.shape[0]
print(f"There were {diff} duplicate records identified and removed in the data set")

print ("Null values detected in entire data set: \n") 
round(100*(train_df.isnull().sum()/len(train_df.index)),2)

train_df = train_df.drop('salary_range', axis=1)

train_df['location'] = train_df['location'].fillna(value='Other')
train_df['department'] = train_df['department'].fillna(value='Other')
train_df['company_profile'] = train_df['company_profile'].fillna(value='')
train_df['description'] = train_df['description'].fillna(value='')
train_df['requirements'] = train_df['requirements'].fillna(value='')
train_df['benefits'] = train_df['benefits'].fillna(value='')
train_df['employment_type'] = train_df['employment_type'].fillna(value='Other')
train_df['required_experience'] = train_df['required_experience'].fillna(value='Not Applicable')
train_df['required_education'] = train_df['required_education'].fillna(value='Unspecified')
train_df['industry'] = train_df['industry'].fillna(value='Other')
train_df['function'] = train_df['function'].fillna(value='Other')

train_df.isnull().sum().sum()

cat_df = train_df[["telecommuting", "has_company_logo", "has_questions", "employment_type", "required_experience", "required_education", "industry", "function","labeled"]]
cat_df = cat_df.fillna("None")

fig, axes = plt.subplots(ncols=2, figsize=(17, 5), dpi=100)
plt.tight_layout()

train_df["labeled"].value_counts().plot(kind='pie', ax=axes[0], labels=['Real Post (95%)', 'Fake Post (5%)'])
temp = train_df["labeled"].value_counts()
sns.barplot(temp.index, temp, ax=axes[1])

axes[0].set_ylabel(' ')
axes[1].set_ylabel(' ')
axes[1].set_xticklabels(["Real Post (17014) [0's]", "Fake Post (866) [1's]"])

axes[0].set_title('Target Distribution in Dataset', fontsize=13)
axes[1].set_title('Target Count in Dataset', fontsize=13)

plt.show()

cat_cols = ["telecommuting", "has_company_logo", "has_questions", "employment_type", "required_experience", "required_education",]



chart = train_df[['required_education','labeled']]

plt.figure(figsize=(30,10))
sns.countplot(palette='Oranges', hue='labeled', y='required_education',data=chart)
plt.show()

chart = train_df[['function','labeled']]

plt.figure(figsize=(30,20))
sns.countplot(palette='Oranges', hue='labeled', y='function',data=chart)
plt.show()

chart = train_df[['employment_type','labeled']]

plt.figure(figsize=(10,5))
sns.countplot(palette='Oranges', hue='labeled', y='employment_type',data=chart)
plt.show()

chart = train_df[['required_experience','labeled']]

plt.figure(figsize=(10,5))
sns.countplot(palette='Oranges', hue='labeled', y='required_experience',data=chart)
plt.show()

chart = train_df[['has_company_logo','labeled']]

plt.figure(figsize=(5,5))
sns.countplot(palette='Oranges', hue='labeled', y='has_company_logo',data=chart)
plt.show()

chart = train_df[['has_questions','labeled']]

plt.figure(figsize=(5,5))
sns.countplot(palette='Oranges', hue='labeled', y='has_questions',data=chart)
plt.show()

train_df = train_df.drop(['job_id','company_profile','description','benefits','requirements'],axis=1)

train_df = pd.get_dummies(train_df, prefix_sep = '_', drop_first=True)
train_df.shape

X = train_df.drop('labeled',axis=1)


y = train_df['labeled']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)

rfc = RandomForestClassifier() 

start = time() 
rfc.fit(X_train, y_train)

print("RandomForestClassifier took %.2f seconds"
      % (time() - start,))

print(f"Accuracy: {round(rfc.score(X_test, y_test)*100)}%")

print("Parameters: ", rfc.get_params())

y_pred = rfc.predict(X_test)
confusion_matrix(y_test, y_pred)

print(classification_report(y_test, y_pred))

ns_probs = [0 for _ in range(len(y_test))]
lr_probs = rfc.predict_proba(X_test)

lr_probs = lr_probs[:, 1]

ns_auc = roc_auc_score(y_test, ns_probs)
lr_auc = roc_auc_score(y_test, lr_probs)

print('0 Classifier: ROC AUC=%.3f' % (ns_auc))
print('RFC Classifier: ROC AUC=%.3f' % (lr_auc))

ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)

plt.plot(ns_fpr, ns_tpr, linestyle='--', label='0 Classifier')
plt.plot(lr_fpr, lr_tpr, marker='.', label='RFC Classifier')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.legend()

plt.show()

cv = ShuffleSplit(n_splits=100, test_size=0.3, random_state=16)
cv = KFold(n_splits=10, shuffle=True, random_state=16)

start = time()
results = cross_val_score(rfc, X_train, y_train, cv=cv)
print("Cross validation took %.2f seconds"
      % (time() - start,))
print(f"Accuracy: {round(results.mean()*100, 2)}%")

feature_imp = pd.DataFrame(rfc.feature_importances_, index=X_train.columns,
columns=['importance']).sort_values('importance', ascending=False)
feature_imp

knn = KNeighborsClassifier()

start = time()

knn.fit(X_train, y_train)

print("KNeighborsClassifier took %.2f seconds"
      % (time() - start,))

print(f"Accuracy: {round(knn.score(X_test, y_test)*100)}%")
print("Parameters: ", knn.get_params())
y_pred = knn.predict(X_test)
confusion_matrix(y_test, y_pred)

print(classification_report(y_test, y_pred))

ns_probs = [0 for _ in range(len(y_test))]
lr_probs = knn.predict_proba(X_test)

lr_probs = lr_probs[:, 1]

ns_auc = roc_auc_score(y_test, ns_probs)
lr_auc = roc_auc_score(y_test, lr_probs)

print('0 Classifier: ROC AUC=%.3f' % (ns_auc))
print('KNN Classifier: ROC AUC=%.3f' % (lr_auc))

ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)

plt.plot(ns_fpr, ns_tpr, linestyle='--', label='0 Classifier')
plt.plot(lr_fpr, lr_tpr, marker='.', label='KNN Classifier')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.legend()

plt.show()

dt = DecisionTreeClassifier()

start = time()

dt.fit(X_train, y_train)

print("Decision Tree Classifier took %.2f seconds"
      % (time() - start,))

print(f"Accuracy: {round(dt.score(X_test, y_test)*100)}%")
print("Parameters: ", dt.get_params())
y_pred = dt.predict(X_test)
confusion_matrix(y_test, y_pred)

print(classification_report(y_test, y_pred))

ns_probs = [0 for _ in range(len(y_test))]
lr_probs = dt.predict_proba(X_test)

lr_probs = lr_probs[:, 1]

ns_auc = roc_auc_score(y_test, ns_probs)
lr_auc = roc_auc_score(y_test, lr_probs)

print('0 Classifier: ROC AUC=%.3f' % (ns_auc))
print('DT Classifier: ROC AUC=%.3f' % (lr_auc))

ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)

plt.plot(ns_fpr, ns_tpr, linestyle='--', label='0 Classifier')
plt.plot(lr_fpr, lr_tpr, marker='.', label='Decision Tree Classifier')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.legend()

plt.show()