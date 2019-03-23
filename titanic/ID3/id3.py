from sys import float_info
from collections import Counter
from math import log2

#This implements the ID3 decision tree algorithm
#Paper link: https://link.springer.com/content/pdf/10.1007%2FBF00116251.pdf

class ID3:
    def __init__(self, dataset, attrs, class_attr):
        self.dataset = dataset
        self.attrs = attrs
        self.class_attr = class_attr

        self.tree = self._id3(dataset, attrs, class_attr)

    #Given a list of probabilities, this returns 
    #-1 * summation(list_item*log2(list_item))
    @staticmethod
    def get_expected_information_required(probs):
        return -sum(p * log2(p) for p in probs if p > 0)

    #Calculates the probabilities with which the 
    #various attr occur in dataset passed
    #then calls get_expected_information_required method
    @staticmethod
    def get_info_required_for_branching_on_attr(dataset, attr):
        classes = set(row[attr] for row in dataset)
        probs = []
        for c in classes:
            subset = (row for row in dataset if row[attr] == c)
            probs.append(sum(1 for row in subset))
        probs = [p/sum(probs) for p in probs]

        return ID3.get_expected_information_required(probs)

    #Calculates the information gained by branching on attribute attr
    @staticmethod
    def get_information_gained_by_branching_on_attr(dataset, attr, class_attr):
        #get the expected information required for labeling
        #the subset of the dataset at the current node
        #this will be I(p, n)
        total_expected_information_required = ID3.get_info_required_for_branching_on_attr(dataset, class_attr)

        options = set(row[attr] for row in dataset)
        #proportion_list_for_attr_values will contain (pi + ni)/(p + n)
        proportion_list_for_attr_values = []
        #expected_information_required_list_for_attr_values will contain I(pi, ni)
        expected_information_required_list_for_attr_values = []

        for o in options:
            subset = [row for row in dataset if row[attr] == o]
            proportion_list_for_attr_values.append(sum(1 for row in subset))

            expected_information_required_list_for_attr_values.append(ID3.get_info_required_for_branching_on_attr(subset, class_attr))
        proportion_list_for_attr_values = [f/sum(proportion_list_for_attr_values) for f in proportion_list_for_attr_values]

        #return I(p,n) - summation_of(I(pi,ni)*((pi+ni)/(p+n)))
        return total_expected_information_required - sum(f * e for f, e in zip(proportion_list_for_attr_values, expected_information_required_list_for_attr_values))

    def _id3(self, dataset, attrs, class_attr):
        root = {}

        #calculate information gain for each attribute
        information_gain = {}
        for attr in attrs:
            information_gain[attr] = ID3.get_information_gained_by_branching_on_attr(dataset, attr, class_attr)

        #selected_attr is the one which has max information gain
        selected_attr = max(
                (attr for attr in information_gain),
                key=lambda attr: information_gain[attr]
                )
        #If the information required to branch off on class_attr is 0
        #then we have reached a leaf node. the dataset has a single value 
        #for the class_attr, so no information is further required to do branching. 
        if abs(ID3.get_info_required_for_branching_on_attr(dataset, class_attr)) < float_info.epsilon:
            root['class'] = dataset[0][class_attr]
        #If maximum information gain is 0 by branching off any of the left 
        #attributes, then select the most common class_atr value and 
        #set root{'class'} to that class_atr value. 
        elif abs(max(information_gain.values())) < float_info.epsilon:
            classes = set(row[class_attr] for row in dataset)
            occurrences = Counter(row[class_attr] for row in dataset)
            root['class'], _ = occurrences.most_common(1)[0]
        else:
            root['attr'] = selected_attr
            root['possibilities'] = set(row[selected_attr] for row in dataset)
            root['children'] = {}
            occurrences = Counter(row[class_attr] for row in dataset)
            root['guess_class'], _ = occurrences.most_common(1)[0]

            for p in root['possibilities']:
                subset = [row for row in dataset if row[selected_attr] == p]
                root['children'][p] = self._id3(subset, attrs, class_attr)

        return root


    def query(self, query, node=None):
        node = node or self.tree

        if node.get('class', None) is not None:
            return node['class']

        try:
            value = query[node['attr']]
            return self.query(query, node['children'][value])
        except:
            return node['guess_class']
