import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn import ensemble
from sklearn import model_selection
import xgboost as xgb
from sklearn import preprocessing
from sklearn import svm
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
########################################feature engineering####################################################
train_data_org = pd.read_csv('train.csv')
test_data_org = pd.read_csv('test.csv')
test_data_org['Survived'] = 0
combined_train_test = train_data_org.append(test_data_org)

#embarked
if combined_train_test['Embarked'].isnull().sum() != 0:
    combined_train_test['Embarked'].fillna(combined_train_test['Embarked'].mode().iloc[0], inplace=True)
emb_dummies_df = pd.get_dummies(combined_train_test['Embarked'],prefix=combined_train_test[['Embarked']].columns[0])
combined_train_test = pd.concat([combined_train_test, emb_dummies_df], axis=1)

#sex
sex_dummies_df = pd.get_dummies(combined_train_test['Sex'], prefix=combined_train_test[['Sex']].columns[0])
combined_train_test = pd.concat([combined_train_test, sex_dummies_df], axis=1)

#name
combined_train_test['Title'] = combined_train_test['Name'].str.extract('.+,(.+)').str.extract( '^(.+?)\.').str.strip()

title_Dict = {}
title_Dict.update(dict.fromkeys(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 0))
title_Dict.update(dict.fromkeys(['Jonkheer', 'Don', 'Sir', 'the Countess', 'Dona', 'Lady'], 1))
title_Dict.update(dict.fromkeys(['Mme', 'Ms', 'Mrs'], 2))
title_Dict.update(dict.fromkeys(['Mlle', 'Miss'], 3))
title_Dict.update(dict.fromkeys(['Mr'], 4))
title_Dict.update(dict.fromkeys(['Master'], 5))

combined_train_test['Title'] = combined_train_test['Title'].map(title_Dict)

title_dummies_df = pd.get_dummies(combined_train_test['Title'], prefix=combined_train_test[['Title']].columns[0])
combined_train_test = pd.concat([combined_train_test, title_dummies_df], axis=1)

#fare
def fare_category(fare):
        if fare <= 4:
            return 0
        elif fare <= 10:
            return 1
        elif fare <= 30:
            return 2
        elif fare <= 45:
            return 3
        else:
            return 4
if combined_train_test['Fare'].isnull().sum() != 0:
    combined_train_test['Fare'] = combined_train_test[['Fare']].fillna(combined_train_test.groupby('Pclass').transform('mean'))
combined_train_test['Group_Ticket'] = combined_train_test['Fare'].groupby(by=combined_train_test['Ticket']).transform('count')
combined_train_test['Fare'] = combined_train_test['Fare'] / combined_train_test['Group_Ticket']
combined_train_test.drop(['Group_Ticket'], axis=1, inplace=True)

combined_train_test['Fare_Category'] = combined_train_test['Fare'].map(fare_category)
fare_cat_dummies_df = pd.get_dummies(combined_train_test['Fare_Category'],prefix=combined_train_test[['Fare_Category']].columns[0])
combined_train_test = pd.concat([combined_train_test, fare_cat_dummies_df], axis=1)

#class
Pclass_1_mean_fare = combined_train_test['Fare'].groupby(by=combined_train_test['Pclass']).mean().get([1]).values[0]
Pclass_2_mean_fare = combined_train_test['Fare'].groupby(by=combined_train_test['Pclass']).mean().get([2]).values[0]
Pclass_3_mean_fare = combined_train_test['Fare'].groupby(by=combined_train_test['Pclass']).mean().get([3]).values[0]
# 建立Pclass_Fare Category
def pclass_fare_category(df, Pclass_1_mean_fare, Pclass_2_mean_fare, Pclass_3_mean_fare):
    if (df['Pclass'] == 1):
        if (df['Fare'] <= Pclass_1_mean_fare):
            return 'Pclass_1_Low_Fare'
        else:
            return 'Pclass_1_High_Fare'
    elif (df['Pclass'] == 2):
        if (df['Fare'] <= Pclass_2_mean_fare):
            return 'Pclass_2_Low_Fare'
        else:
            return 'Pclass_2_High_Fare'
    elif (df['Pclass'] == 3):
        if (df['Fare'] <= Pclass_3_mean_fare):
            return 'Pclass_3_Low_Fare'
        else:
            return 'Pclass_3_High_Fare'

combined_train_test['Pclass_Fare_Category'] = combined_train_test.apply(pclass_fare_category, args=(Pclass_1_mean_fare, Pclass_2_mean_fare, Pclass_3_mean_fare), axis=1)
p_fare = LabelEncoder()
p_fare.fit(np.array(['Pclass_1_Low_Fare', 'Pclass_1_High_Fare', 'Pclass_2_Low_Fare', 'Pclass_2_High_Fare', 'Pclass_3_Low_Fare','Pclass_3_High_Fare']))#给每一项添加标签
combined_train_test['Pclass_Fare_Category'] = p_fare.transform(combined_train_test['Pclass_Fare_Category'])#转换成数值

#family size
def family_size_category(family_size):
    if (family_size <= 1):
        return 'Single'
    elif (family_size <= 3):
        return 'Small_Family'
    else:
        return 'Large_Family'

combined_train_test['Family_Size'] = combined_train_test['Parch'] + combined_train_test['SibSp'] + 1
combined_train_test['Family_Size_Category'] = combined_train_test['Family_Size'].map(family_size_category)
le_family = LabelEncoder()
le_family.fit(np.array(['Single', 'Small_Family', 'Large_Family']))
combined_train_test['Family_Size_Category'] = le_family.transform(combined_train_test['Family_Size_Category'])
fam_size_cat_dummies_df = pd.get_dummies(combined_train_test['Family_Size_Category'],
                                         prefix=combined_train_test[['Family_Size_Category']].columns[0])
combined_train_test = pd.concat([combined_train_test, fam_size_cat_dummies_df], axis=1)

#age
def drop_col_not_req(df, cols):
    df.drop(cols, axis = 1, inplace = True)

def fill_missing_age(missing_age_train, missing_age_test):
    missing_age_X_train = missing_age_train.drop(['Age'], axis=1)
    missing_age_Y_train = missing_age_train['Age']
    missing_age_X_test = missing_age_test.drop(['Age'], axis=1)
    # 模型1
    gbm_reg = ensemble.GradientBoostingRegressor(random_state=42)
    gbm_reg_param_grid = {'n_estimators': [2000], 'max_depth': [3], 'learning_rate': [0.01], 'max_features': [3, 5]}
    gbm_reg_grid = model_selection.GridSearchCV(gbm_reg, gbm_reg_param_grid, cv=10, n_jobs=25, verbose=1,
                                                scoring='neg_mean_squared_error')
    gbm_reg_grid.fit(missing_age_X_train, missing_age_Y_train)
    print('Age feature Best GB Params:' + str(gbm_reg_grid.best_params_))
    print('Age feature Best GB Score:' + str(gbm_reg_grid.best_score_))
    print('GB Train Error for "Age" Feature Regressor:' + str(
        gbm_reg_grid.score(missing_age_X_train, missing_age_Y_train)))
    missing_age_test['Age_GB'] = gbm_reg_grid.predict(missing_age_X_test)
    print(missing_age_test['Age_GB'][:4])
    # 模型2
    lrf_reg = LinearRegression()
    lrf_reg_param_grid = {'fit_intercept': [True], 'normalize': [True]}
    lrf_reg_grid = model_selection.GridSearchCV(lrf_reg, lrf_reg_param_grid, cv=10, n_jobs=25, verbose=1,
                                                scoring='neg_mean_squared_error')
    lrf_reg_grid.fit(missing_age_X_train, missing_age_Y_train)
    print('Age feature Best LR Params:' + str(lrf_reg_grid.best_params_))
    print('Age feature Best LR Score:' + str(lrf_reg_grid.best_score_))
    print('LR Train Error for "Age" Feature Regressor' + str(
        lrf_reg_grid.score(missing_age_X_train, missing_age_Y_train)))
    missing_age_test['Age_LRF'] = lrf_reg_grid.predict(missing_age_X_test)
    print(missing_age_test['Age_LRF'][:4])
    # 将两个模型预测后的均值作为最终预测结果
    print('shape1', missing_age_test['Age'].shape, missing_age_test[['Age_GB', 'Age_LRF']].mode(axis=1).shape)
    # missing_age_test['Age'] = missing_age_test[['Age_GB','Age_LRF']].mode(axis=1)
    missing_age_test['Age'] = np.mean([missing_age_test['Age_GB'], missing_age_test['Age_LRF']])
    print(missing_age_test['Age'][:4])
    drop_col_not_req(missing_age_test, ['Age_GB', 'Age_LRF'])

    return missing_age_test

combined_train_test['Age_Null'] = combined_train_test['Age'].apply(lambda x: 1 if(pd.notnull(x)) else 0)
missing_age_df = pd.DataFrame(combined_train_test[['Age', 'Parch', 'Sex', 'SibSp', 'Family_Size', 'Family_Size_Category',
                             'Title', 'Fare', 'Fare_Category', 'Pclass', 'Embarked']])
missing_age_df = pd.get_dummies(missing_age_df,columns=['Title', 'Family_Size_Category', 'Fare_Category', 'Sex', 'Pclass' ,'Embarked'])
missing_age_train = missing_age_df[missing_age_df['Age'].notnull()]
missing_age_test = missing_age_df[missing_age_df['Age'].isnull()]
combined_train_test.loc[(combined_train_test.Age.isnull()), 'Age'] = fill_missing_age(missing_age_train,missing_age_test)

#ticket
combined_train_test['Ticket_Letter'] = combined_train_test['Ticket'].str.split().str[0]
combined_train_test['Ticket_Letter'] = combined_train_test['Ticket_Letter'].apply(lambda x:np.nan if x.isnumeric() else x)
combined_train_test['Ticket_Number'] = combined_train_test['Ticket'].apply(lambda x: pd.to_numeric(x,errors='coerce'))
combined_train_test['Ticket_Number'].fillna(0,inplace=True)
combined_train_test = pd.get_dummies(combined_train_test,columns=['Ticket','Ticket_Letter'])

#Cabin
combined_train_test['Cabin_Letter'] = combined_train_test['Cabin'].apply(lambda x:str(x)[0] if pd.notnull(x) else x)
combined_train_test = pd.get_dummies(combined_train_test,columns=['Cabin','Cabin_Letter'])

#normalize age and fare
scale_age_fare = preprocessing.StandardScaler().fit(combined_train_test[['Age', 'Fare']])
combined_train_test[['Age', 'Fare']] = scale_age_fare.transform(combined_train_test[['Age', 'Fare']])

#train, test seperation
combined_train_test.drop(['Name', 'PassengerId', 'Ticket_Number', 'Embarked', 'Sex', 'Family_Size_Category'], axis = 1, inplace = True)
train_data = combined_train_test[:891]
test_data = combined_train_test[891:]
titanic_train_data_X = train_data.drop(['Survived'],axis=1)
titanic_train_data_Y = train_data['Survived']
titanic_test_data_X = test_data.drop(['Survived'],axis=1)

##############################################################ensemble##########################################################
#choose top features
def get_top_n_features(titanic_train_data_X, titanic_train_data_Y, top_n_features):
    # 随机森林
    rf_est = RandomForestClassifier(random_state=3)
    rf_param_grid = {'n_estimators': [500], 'max_features':[5, 6, 10], 'min_samples_split': [2, 3], 'max_depth': [20]}
    rf_grid = model_selection.GridSearchCV(rf_est, rf_param_grid, n_jobs=25, cv=10, verbose=1)
    rf_grid.fit(titanic_train_data_X, titanic_train_data_Y)
    # 将feature按Importance排序
    feature_imp_sorted_rf = pd.DataFrame({'feature': list(titanic_train_data_X),
                                          'importance': rf_grid.best_estimator_.feature_importances_}).sort_values(
        'importance', ascending=False)
    features_top_n_rf = feature_imp_sorted_rf.head(top_n_features)['feature']
    print('Sample 25 Features from RF Classifier')
    print(str(features_top_n_rf[:25]))

    # AdaBoost
    ada_est = ensemble.AdaBoostClassifier(random_state=42)
    ada_param_grid = {'n_estimators': [500], 'learning_rate': [0.5, 0.6]}
    ada_grid = model_selection.GridSearchCV(ada_est, ada_param_grid, n_jobs=25, cv=10, verbose=1)
    ada_grid.fit(titanic_train_data_X, titanic_train_data_Y)
    # 排序
    feature_imp_sorted_ada = pd.DataFrame({'feature': list(titanic_train_data_X),
                                           'importance': ada_grid.best_estimator_.feature_importances_}).sort_values(
        'importance', ascending=False)
    features_top_n_ada = feature_imp_sorted_ada.head(top_n_features)['feature']
    print('Sample 25 Features from ADA Classifier:')
    print(str(features_top_n_ada[:25]))

    # ExtraTree
    et_est = ensemble.ExtraTreesClassifier(random_state=42)
    et_param_grid = {'n_estimators': [500], 'min_samples_split': [3, 4], 'max_depth': [15]}
    et_grid = model_selection.GridSearchCV(et_est, et_param_grid, n_jobs=25, cv=10, verbose=1)
    et_grid.fit(titanic_train_data_X, titanic_train_data_Y)
    # 排序
    feature_imp_sorted_et = pd.DataFrame({'feature': list(titanic_train_data_X),
                                          'importance': et_grid.best_estimator_.feature_importances_}).sort_values(
        'importance', ascending=False)
    features_top_n_et = feature_imp_sorted_et.head(top_n_features)['feature']
    print('Sample 25 Features from ET Classifier:')
    print(str(features_top_n_et[:25]))
    # 将三个模型挑选出来的前features_top_n_et合并
    features_top_n = pd.concat([features_top_n_rf, features_top_n_ada, features_top_n_et],
                               ignore_index=True).drop_duplicates()

    return features_top_n

feature_to_pick = 196
titanic_train_data_X.info()
0/0
feature_top_n = get_top_n_features(titanic_train_data_X,titanic_train_data_Y,feature_to_pick)

titanic_train_data_X = titanic_train_data_X[feature_top_n]
titanic_test_data_X = titanic_test_data_X[feature_top_n]


#oversampling
#titanic_train_data_X, titanic_train_data_Y = RandomOverSampler().fit_sample(titanic_train_data_X, titanic_train_data_Y)
titanic_train_data_X, titanic_train_data_Y = SMOTE().fit_sample(titanic_train_data_X, titanic_train_data_Y)
#voting
#xgb_est = xgb.XGBClassifier(learning_rate=0.03, random_state=3, n_estimators=900, subsample=0.8, n_jobs = 50,colsample_bytree = 0.8, max_depth = 10, verbose=1)
#svm_est = svm.SVC(kernel='rbf', gamma = 1e-3, C =100)
ada_est = ensemble.AdaBoostClassifier(n_estimators = 1000, random_state = 3, learning_rate = 0.1)
rf_est = ensemble.RandomForestClassifier(n_estimators = 1000, criterion = 'gini', max_features = 'sqrt',
                                             max_depth = 10, min_samples_split = 4, min_samples_leaf = 20,
                                             n_jobs = 50, random_state = 42, verbose = 1)
gbm_est = ensemble.GradientBoostingClassifier(n_estimators=1000, learning_rate=0.003, loss='exponential',
                                              min_samples_split=3, min_samples_leaf=20, max_features='sqrt',
                                              max_depth=10, random_state=42, verbose=1)
et_est = ensemble.ExtraTreesClassifier(n_estimators=1000, max_features='sqrt', max_depth=50, n_jobs=50,
                                       criterion='entropy', random_state=42, verbose=1)
voting_est = ensemble.VotingClassifier(estimators = [('ada',ada_est), ('rf', rf_est),('gbm', gbm_est),('et', et_est)],
                                   voting = 'soft', weights = [3,4,5,3],
                                   n_jobs = 50)
voting_est.fit(titanic_train_data_X, titanic_train_data_Y)
print('Score', voting_est.score(titanic_train_data_X,titanic_train_data_Y))
########################################################################3output###########################################################################
titanic_test_data_X['Survived'] = voting_est.predict(titanic_test_data_X)
submission = pd.DataFrame({'PassengerId':test_data_org.loc[:,'PassengerId'],
                           'Survived':titanic_test_data_X.loc[:,'Survived']})
submission.to_csv('submission_result.csv',index=False,sep=',')