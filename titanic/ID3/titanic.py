from collections import Counter
import csv
import id3
import numpy
import pandas
import sklearn

if __name__ == '__main__':
    with open('../input/train.csv') as f:
        reader = csv.DictReader(f)
        train = list(reader)
    with open('../input/test.csv') as f:
        reader = csv.DictReader(f)
        test = list(reader)

    attrs_to_ignore = {'PassengerId', 'Name', 'Ticket', 'Cabin'}
    class_attr = 'Survived'
    attrs = set(train[0].keys())
    significant_attrs = attrs - set([class_attr]) - attrs_to_ignore

    tree = id3.ID3(train, significant_attrs, class_attr)

    csvData = [['PassengerId', 'Survived']]
    for row in test:
        csvData.append([row['PassengerId'], tree.query(row)])

    with open('person.csv', 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(csvData)

    csvFile.close()
