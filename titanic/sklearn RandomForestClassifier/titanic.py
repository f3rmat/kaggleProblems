import numpy as np
import pandas as pd
import statistics
from sklearn.ensemble import RandomForestClassifier

def fill_in_missing_fares(train, test): 
	pclass_to_avg_fare_map = {}
	for it in range(len(train["Pclass"])):
		if train["Pclass"][it] in pclass_to_avg_fare_map: 
			pclass_to_avg_fare_map[train["Pclass"][it]].append(train["Fare"][it])
		else: 
			pclass_to_avg_fare_map[train["Pclass"][it]] = [train["Fare"][it]]

	for key in pclass_to_avg_fare_map:
		pclass_to_avg_fare_map[key] = round(statistics.mean(pclass_to_avg_fare_map[key]), 4)

	for x in range(len(test["Fare"])):
	    if pd.isnull(test["Fare"][x]):
	    	test["Fare"][x] = pclass_to_avg_fare_map[test["Pclass"][x]]

	return test

def main(): 
	train = pd.read_csv('../input/train.csv')
	test = pd.read_csv('../input/test.csv')

	combine = [train, test]

	# Convert string values 'male' and 'female' to int values
	sex_mapping = {'male': 0, 'female': 1}
	train['Sex'] = train['Sex'].map(sex_mapping)
	test['Sex'] = test['Sex'].map(sex_mapping)

	#for the fares that don't exist in the test data
	#use the test data to get the average fare for that pclass
	test = fill_in_missing_fares(train, test)

	guess_ages = np.zeros((2,3))

	for dataset in combine:

	    for sex in range(0, 2):
	        for pclass in range(0, 3):
	            guess_df = dataset[
	                (dataset['Sex'] == sex) &
	                (dataset['Pclass'] == pclass+1)
	            ]['Age'].dropna()
	            age_guess = guess_df.median()
	            guess_ages[sex, pclass] = int(age_guess/0.5 + 0.5) * 0.5

	    for sex in range(0, 2):
	        for pclass in range(0, 3):
	            dataset.loc[
	                (dataset.Age.isnull()) &
	                (dataset.Sex == sex) &
	                (dataset.Pclass == pclass+1),
	                'Age'
	            ] = guess_ages[sex, pclass]

	#Train DecisionTreeClassifier
	Y_train = train['Survived']
	X_Train = train.drop(['Ticket', 'Cabin', 'Name', 'PassengerId', 'SibSp', 'Parch', 'Embarked', 'Survived'], axis=1)
	random_forest = RandomForestClassifier(criterion='gini', 
                             n_estimators=700,
                             min_samples_split=10,
                             min_samples_leaf=1,
                             max_features='auto',
                             oob_score=True,
                             random_state=1,
                             n_jobs=-1)
	random_forest.fit(X_Train, Y_train)

	#Use DecisionTreeClassifier to predict
	test = test.drop(['Ticket', 'Cabin', 'Name', 'SibSp', 'Parch', 'Embarked'], axis=1)
	X_test  = test.drop("PassengerId", axis=1)
	Y_test = random_forest.predict(X_test)
	submission = pd.DataFrame({
	    'PassengerId': test['PassengerId'],
	    'Survived': Y_test
	})

	submission.to_csv('prediction.csv', index=False)

if __name__ == '__main__':
	main()