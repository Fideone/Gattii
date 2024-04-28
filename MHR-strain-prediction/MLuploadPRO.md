```
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn import metrics
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("./0_testData14features.csv")
print(data.head())
print(data.shape)
print(data['class'])

selected_columns = [1,2,3,4,5,6,7,8,9,10,11,12,13,14] # 14 features
#selected_columns = [3,6,7,8,9,10,11,12,13,14] # 10 features
#selected_columns = [3,6,7,8,9,10] # 6 features
#selected_columns = [3,6,7,9,10] # 5 features
#selected_columns = [3,7,9,10] # 4 features
#selected_columns = [3,7,10] # 3 features
print(data.iloc[:, selected_columns])

x_train, x_test, y_train, y_test = train_test_split(data.iloc[:, selected_columns], data.iloc[:, 0], test_size=0.2, random_state=5)

class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models

    def fit(self, X, y):
        self.models = [clone(x.best_estimator_) for x in self.models]

        for model in self.models:
            model.fit(X, y)

        return self

    def predict(self, X):
        proba = self.predict_proba(X)
        labels = np.argmax(proba, axis=1)
        return labels

    def predict_proba(self, X):
        predictions = np.column_stack([
            model.predict_proba(X)[:, 1] for model in self.models
        ])

        means = np.mean(predictions, axis=1)

        return np.column_stack([1 - means, means])

def optimise_logres_featsel(X, y, cv, label='Response', metric='roc_auc'):
    scaler = StandardScaler()
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=5)
    logres = LogisticRegression(random_state=5, penalty='elasticnet', solver='saga', max_iter=10000, n_jobs=-1,
                                class_weight='balanced')
    pipe = Pipeline(steps=[('scaler', scaler), ('logres', logres)])

    param_grid = {'logres__C': np.logspace(-3, 3, 30),
                  'logres__l1_ratio': np.arange(0.1, 1.1,
                                                0.1)}

    search = GridSearchCV(pipe, param_grid, cv=kf, n_jobs=-1, verbose=0)
    search.fit(X, y)
    logres_best = search.best_estimator_.named_steps['logres']
    return search, logres_best

def optimise_xgboost_featsel(X, y, cv, metric='roc_auc'):
    scaler = StandardScaler()
    xgboost = xgb.XGBClassifier(random_state=5, n_jobs=-1)
    pipe = Pipeline(steps=[('scaler', scaler), ('xgboost', xgboost)])

    param_grid = {
        'xgboost__max_depth': [3, 6, 9],
        'xgboost__n_estimators': [100, 200, 300],
        'xgboost__learning_rate': [0.01, 0.1, 0.2]
    }

    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=5)
    search = GridSearchCV(pipe, param_grid, cv=kf, scoring=metric, n_jobs=-1, verbose=0)
    search.fit(X, y)

    xgboost_best = search.best_estimator_.named_steps['xgboost']

    feature_importances = xgboost_best.feature_importances_

    return search, feature_importances, xgboost_best


def optimise_rf_featsel(X, y, cv, label='Response'):
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=5)
    rf = RandomForestClassifier(random_state=5)
    pipe = Pipeline(steps=[('rf', rf)])
    param_grid = {"rf__max_depth": [3, None],
                  "rf__n_estimators": [5, 10, 25, 50, 100],
                  "rf__max_features": [0.05, 0.1, 0.2, 0.5, 0.7],
                  "rf__min_samples_split": [2, 3, 6, 10, 12, 15]
                  }

    search = RandomizedSearchCV(pipe, param_grid, cv=kf, scoring='roc_auc', return_train_score=True, n_jobs=-1,
                                verbose=0, n_iter=1000, random_state=1)
    search.fit(X, y)
    rf_best = search.best_estimator_.named_steps['rf']
    return search, rf_best

def run_all_models(X, y, splits):
    logres_result_auc, logres_best = optimise_logres_featsel(X, y, cv=splits, metric='roc_auc')

    xgbt_result, feature_importances, xgboost_best = optimise_xgboost_featsel(X, y, cv=splits)

    rf_result, rf_best = optimise_rf_featsel(X, y, cv=splits)

    averaged_models = AveragingModels(models=(logres_result_auc, xgbt_result, rf_result))

    results = {}
    results['lr'] = logres_result_auc
    results['xgbt'] = xgbt_result
    results['rf'] = rf_result
    results['avg'] = averaged_models
    xgbt_best = xgbt_result.best_estimator_.named_steps['xgboost']
    xgbt_feature_importances = xgbt_best.feature_importances_
    rf_feature_importances = rf_result.best_estimator_.named_steps['rf'].feature_importances_
    print("XGBoost Feature Importances:", xgbt_feature_importances)
    print("Random Forest Feature Importances:", rf_feature_importances)

    logreg_coefs = logres_result_auc.best_estimator_.named_steps['logres'].coef_
    print("Logistic Regression Coefficients:", logreg_coefs)
    return results


kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=5)
results = run_all_models(x_train, y_train, kf)

pred_logreg = results['lr'].predict_proba(x_test)[:, 1]
pred_xgbt = results['xgbt'].predict_proba(x_test)[:, 1]
pred_rf = results['rf'].predict_proba(x_test)[:, 1]
avg_pred = results['avg'].predict_proba(x_test)[:, 1]
avg_pred2 = results['avg'].predict_proba(x_test)
avg_pred_simple = (pred_logreg + pred_xgbt + pred_rf) / 3

y_test_preds = np.column_stack((
    pred_logreg,
    pred_xgbt,
    pred_rf,
    avg_pred,
    avg_pred2,
    avg_pred_simple
))
aucs_test = []
for i in range(y_test_preds.shape[1]):
    auc = metrics.roc_auc_score(y_test, y_test_preds[:, i])
    aucs_test.append(auc)

print(f"Fold-2: LR AUC = {aucs_test[0]}")
print(f"Fold-2: XGBT AUC = {aucs_test[1]}")
print(f"Fold-2: RF AUC = {aucs_test[2]}")
print(f"Fold-2: Average Pred AUC = {aucs_test[3]}")
print(f"Fold-2: Average Pred AUC2 = {aucs_test[4]}")
print(f"Fold-2: Average Pred AUC3 = {aucs_test[5]}")

def output_feature_contributions(model, feature_names):
    if isinstance(model, LogisticRegression):
        coefs = model.coef_[0]
        feature_contributions = dict(zip(feature_names, coefs))
    elif isinstance(model, RandomForestClassifier) or isinstance(model, xgb.XGBClassifier):
        feature_importances = model.feature_importances_
        feature_contributions = dict(zip(feature_names, feature_importances))
    else:
        feature_contributions = None
    return feature_contributions

logreg_feature_contributions = output_feature_contributions(results['lr'].best_estimator_.named_steps['logres'], x_train.columns)
print("Logistic Regression Feature Contributions:", logreg_feature_contributions)

rf_feature_contributions = output_feature_contributions(results['rf'].best_estimator_.named_steps['rf'], x_train.columns)
print("Random Forest Feature Contributions:", rf_feature_contributions)

xgbt_result, feature_importances, xgboost_best = optimise_xgboost_featsel(x_train, y_train, kf)

xgbt_feature_contributions = output_feature_contributions(xgboost_best, x_train.columns)
print("XGBoost Feature Contributions:", xgbt_feature_contributions)


feature_contributions = {
    'Feature': list(logreg_feature_contributions.keys()),
    'Logistic_Regression': list(logreg_feature_contributions.values()),
    'Random_Forest': list(rf_feature_contributions.values()),
    'XGBoost': list(xgbt_feature_contributions.values())
}

df = pd.DataFrame(feature_contributions)

csv_file_path = './feature_contributionsFinal14.csv'
df.to_csv(csv_file_path, index=False)

print(f"DataFrame saved as {csv_file_path}")

y_pred = results['avg'].predict_proba(x_test)

y_preds = results['avg'].predict(x_test)
accuracy = accuracy_score(y_test, y_preds)
tn, fp, fn, tp = confusion_matrix(y_test, y_preds).ravel()
sensitivity = tp / (tp + fn)
print("准确率是：", accuracy)
print("敏感度是：", sensitivity)

AUC = roc_auc_score(y_true=y_test, y_score=y_pred[:,1])
print("AUC是：\n", AUC)
y_prob = results['avg'].predict_proba(x_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = sklearn.metrics.auc(fpr, tpr)

y_test = y_test.reset_index(drop=True)

n_bootstraps = 1000
auc_values = []
for i in range(n_bootstraps):
    indices = np.random.choice(np.arange(len(y_prob)), size=len(y_prob), replace=True)
    bootstrap_scores = y_prob[indices]
    bootstrap_true_labels = y_test[indices]

    if len(np.unique(bootstrap_true_labels)) < 2:
        continue

    bootstrap_fpr, bootstrap_tpr, _ = roc_curve(bootstrap_true_labels, bootstrap_scores)

    bootstrap_auc = sklearn.metrics.auc(bootstrap_fpr, bootstrap_tpr)
    auc_values.append(bootstrap_auc)

lower_bound = np.percentile(auc_values, 2.5)
upper_bound = np.percentile(auc_values, 97.5)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.savefig('roc_train14_3.pdf', format='pdf')
plt.show()

print("lower_bound：\n", lower_bound)
print("upper_bound：\n", upper_bound)

trained_models = results
new_data = pd.read_csv("./1_validatedData14features第三个版本.csv")

new_x = new_data.iloc[:, selected_columns]

new_pred_logreg = trained_models['lr'].predict_proba(new_x)[:, 1]
new_pred_xgbt = trained_models['xgbt'].predict_proba(new_x)[:, 1]
new_pred_rf = trained_models['rf'].predict_proba(new_x)[:, 1]
new_avg_pred = trained_models['avg'].predict_proba(new_x)[:, 1]
new_avg_pred_simple = (new_pred_logreg + new_pred_xgbt + new_pred_rf) / 3

print("逻辑回归预测结果：\n", new_pred_logreg)
print("XGBoost预测结果：\n", new_pred_xgbt)
print("随机森林预测结果：\n", new_pred_rf)
print("平均预测结果：\n", new_avg_pred)
print("简单平均的预测结果：\n", new_avg_pred_simple)

new_y_true = new_data.iloc[:, 0]
new_y_pred = new_avg_pred
new_y_preds = trained_models['avg'].predict(new_x)
new_accuracy = accuracy_score(new_y_true, new_y_preds)
tn, fp, fn, tp = confusion_matrix(new_y_true, new_y_preds).ravel()
new_sensitivity = tp / (tp + fn)
print("准确率是：", new_accuracy)
print("敏感度是：", new_sensitivity)

new_y_true = new_data.iloc[:, 0]
new_y_pred = new_avg_pred
new_auc = roc_auc_score(y_true=new_y_true, y_score=new_y_pred)
print("验证集AUC是：", new_auc)

new_y_true = new_y_true.reset_index(drop=True)

n_bootstraps = 1000
auc_values = []
for i in range(n_bootstraps):
    indices = np.random.choice(np.arange(len(new_y_pred)), size=len(new_y_pred), replace=True)
    bootstrap_scores = new_y_pred[indices]
    bootstrap_true_labels = new_y_true[indices]

    if len(np.unique(bootstrap_true_labels)) < 2:
        continue

    bootstrap_fpr, bootstrap_tpr, _ = roc_curve(bootstrap_true_labels, bootstrap_scores)

    bootstrap_auc = sklearn.metrics.auc(bootstrap_fpr, bootstrap_tpr)
    auc_values.append(bootstrap_auc)

lower_bound = np.nanpercentile(auc_values, 2.5)
upper_bound = np.nanpercentile(auc_values, 97.5)

new_fpr, new_tpr, new_thresholds = roc_curve(new_y_true, new_y_pred)
new_roc_auc = sklearn.metrics.auc(new_fpr, new_tpr)

plt.figure()
plt.plot(new_fpr, new_tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % new_roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for New Data')
plt.legend(loc="lower right")
plt.savefig('roc_validation14_3.pdf', format='pdf')
plt.show()

print("validation_lower_bound：\n", lower_bound)
print("validation_upper_bound：\n", upper_bound)
```

