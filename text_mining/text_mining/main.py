import xml.etree.ElementTree as ET
import os
import json
from nltk import PorterStemmer
import csv
from math import log2
import random
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import recall_score


def formatting_rules(word: str):
    if '$' in word or '%' in word:
        word = ''
    else:
        signs_to_replace = "()?@&,.+!\\/*\":"
        for each_char in signs_to_replace:
            if word.find(each_char) != -1:
                word = word.replace(each_char, '')
        word = word.removesuffix("\'s")
        word = word.removesuffix("\'")
        word = word.removesuffix("--")
        word = word.removesuffix("-")
        word = word.removeprefix("\'")
        word = word.removeprefix("--")
        word = word.removeprefix("-")
        try:
            a = float(word)
            word = ''
        except:
            pass
        numbers = '1234567890'
        nmb_count = 0
        for each_char in numbers:
            if word.find(each_char) != -1:
                nmb_count = nmb_count + 1
        if nmb_count > len(word) / 4:
            word = ''
    return word


def extract_topics(path: str, topics_dict: dict):
    reuters_files = os.listdir(path)
    topics = list()
    for current_file in reuters_files:
        tree = ET.parse(path + current_file)
        root = tree.getroot()
        local_topic = list()
        for codes in root.findall('metadata/codes'):
            if ("topics" in str(codes.attrib.values())):
                for eachcode in codes.findall('code'):
                    local_topic.append(eachcode.attrib.get('code'))
        topics.append(local_topic)
        topics_dict[current_file] = local_topic
    with open("topics.txt", "w") as topics_file:
        ct = 0
        for each_entry in topics:
            topics_file.write(reuters_files[ct] + '\n')
            topics_file.write(json.dumps(each_entry, indent=5) + '\n')
            ct = ct + 1


def prepare_dicts(stopwords: list, path: str, local_dicts: dict, global_dict: dict):
    jsoned_lists = []
    reuters_files = os.listdir(path)
    for current_file in reuters_files:
        tree = ET.parse(path + current_file)
        root = tree.getroot()
        streeng: str
        local_dict = dict()
        for paragraphs in root.findall('title'):
            streeng = paragraphs.text
            word_list = streeng.split()
            paragraph_words = len(word_list)
            words_counter = 0
            for word in word_list:
                word = word.lower()
                word = formatting_rules(word)
                word = stem(word)
                words_counter = words_counter + 1
                if (len(word) != 0):
                    if (words_counter == paragraph_words):
                        if (word[-1] == "."):
                            word = word.replace('.', '')
                    if (word not in stopwords):
                        if (word not in local_dict):
                            local_dict[word] = 1
                        else:
                            local_dict[word] = local_dict[word] + 1
                        if (word not in global_dict):
                            global_dict[word] = 1
                        else:
                            global_dict[word] = global_dict[word] + 1
        for paragraphs in root.findall('text/p'):
            streeng = paragraphs.text
            word_list = streeng.split()
            paragraph_words = len(word_list)
            words_counter = 0
            for word in word_list:
                word = word.lower()
                word = formatting_rules(word)
                word = stem(word)
                words_counter = words_counter + 1
                if (len(word) != 0):
                    if (words_counter == paragraph_words):
                        if (word[-1] == "."):
                            word = word.replace('.', '')
                    if (word not in stopwords):
                        if (word not in local_dict):
                            local_dict[word] = 1
                        else:
                            local_dict[word] = local_dict[word] + 1
                        if (word not in global_dict):
                            global_dict[word] = 1
                        else:
                            global_dict[word] = global_dict[word] + 1
        jsoned_lists.append(json.dumps(local_dict, indent=5))
        local_dicts[current_file] = local_dict
    with open("local_dicts.txt", "w") as output_local, open("global_dict.txt", "w") as output_global:
        ct = 0
        for each_dict in jsoned_lists:
            output_local.write(reuters_files[ct] + '\n')
            output_local.write(each_dict + "\n")
            ct = ct + 1
        output_global.write("GLOBAL_DICT" + "\n")
        output_global.write(json.dumps(global_dict, indent=5))


def get_stopwords(path: str):
    stopword_list = list()
    with open(path, 'r', encoding='UTF8') as stopword_file:
        lines = stopword_file.readlines()
        for each_word in lines:
            stopword_list.append(each_word.replace('\n', ''))
    return stopword_list


def stem(word: str):
    ps = PorterStemmer()
    return ps.stem(word)


def update_dictionaries(locals: dict(), global_dict: dict(), topics_dict: dict(), lower_threshold=5,
                        upper_threshold=95):
    rec_dict = dict()
    for key in topics_dict.keys():
        for eachElement in topics_dict[key]:
            if eachElement in rec_dict:
                rec_dict[eachElement] = rec_dict[eachElement] + 1
            else:
                rec_dict[eachElement] = 1
    total = len(topics_dict)
    keys_to_delete = list()
    for key in rec_dict.keys():
        if rec_dict[key] < lower_threshold / 100 * total or rec_dict[key] > upper_threshold / 100 * total:
            keys_to_delete.append(key)
    for key in topics_dict.keys():
        dict_update = list()
        for value in topics_dict[key]:
            if value not in keys_to_delete:
                dict_update.append(value)
        topics_dict[key] = dict_update
    for key in topics_dict.keys():
        if topics_dict[key] == []:
            for eachKey in locals[key]:
                global_dict[eachKey] = global_dict[eachKey] - locals[key][eachKey]
                if global_dict[eachKey] == 0:
                    del global_dict[eachKey]
            del locals[key]
    topics_to_delete = list()
    for key in topics_dict:
        if topics_dict[key] == []:
            topics_to_delete.append(key)
    for eachTopic in topics_to_delete:
        del topics_dict[eachTopic]
    with open("local_dicts.txt", "w") as output_local:
        for key in locals:
            output_local.write(key + '\n')
            output_local.write(json.dumps(locals[key], indent=5))
            output_local.write("\n")
    with open("global_dict.txt", "w") as output_global:
        output_global.write("GLOBAL_DICT" + "\n")
        output_global.write(json.dumps(global_dict, indent=5))
    with open("topics.txt", "w") as topics_file:
        for key in topics_dict:
            topics_file.write(key + '\n')
            topics_file.write(json.dumps(topics_dict[key], indent=5) + '\n')


def dataset_representation(global_dict: dict, locals: dict, normalized_locals: dict):
    with open('normalized_scores.csv', mode='w', newline='') as csv_file:
        fields = [' ']
        for key in global_dict.keys():
            fields.append(key)
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        sum = 0
        for eachKey in global_dict:
            sum = sum + global_dict[eachKey]
        average_score = sum / len(global_dict)
        for key in locals:
            words_in_locals = dict()
            csv_dict = dict()
            csv_dict[' '] = key
            for currentItem in fields[1:]:
                if currentItem in locals[key]:
                    csv_dict[currentItem] = locals[key][currentItem]
                    if locals[key][currentItem] > average_score:
                        csv_dict[currentItem] = 2
                        words_in_locals[currentItem] = 2  # 2 means above average
                    else:
                        csv_dict[currentItem] = 1
                        words_in_locals[currentItem] = 1  # 1 means below average
                else:
                    csv_dict[currentItem] = 0
                    words_in_locals[currentItem] = 0
            normalized_locals[key] = words_in_locals
            writer.writerow(csv_dict)

def topics_filtering(topics_dict: dict):
    topic_recurrence = dict()
    for eachKey in topics_dict.keys():
        for eachValue in topics_dict[eachKey]:
            if eachValue not in topic_recurrence:
                topic_recurrence[eachValue] = 1
            else:
                topic_recurrence[eachValue] = topic_recurrence[eachValue] + 1
    for eachKey in topics_dict.keys():
        if len(topics_dict[eachKey]) > 1:
            recurrence_list = list()
            for eachItem in topics_dict[eachKey]:
                recurrence_list.append(topic_recurrence[eachItem])
            max_element = max(recurrence_list)
            recurrence_list.clear()
            for eachItem in topics_dict[eachKey]:
                if topic_recurrence[eachItem] == max_element:
                    recurrence_list.append(eachItem)
                    break
            topics_dict[eachKey] = recurrence_list
    with open("topics.txt", "w") as topics_file:
        for key in topics_dict:
            topics_file.write(key + '\n')
            topics_file.write(json.dumps(topics_dict[key], indent=5) + '\n')


def compute_classes_dict(topics_dict: dict, classes_dict: dict):
    for eachItem in topics_dict:
        if topics_dict[eachItem][0] not in classes_dict:
            classes_dict[topics_dict[eachItem][0]] = 1
        else:
            classes_dict[topics_dict[eachItem][0]] = classes_dict[topics_dict[eachItem][0]] + 1


def entropy(classes_dict: dict, number_of_documents: int):
    entropy = 0
    for eachItem in classes_dict.values():
        current_clas = eachItem / number_of_documents
        entropy = entropy + (current_clas * log2(current_clas))
    return -entropy


def information_gain(global_dict: dict, normalized_locals: dict, topics_dict: dict, global_entropy):
    attributes_dict = dict()
    attributes_gain_dict = dict()
    attributes_gain_list = list()
    for eachAttribute in global_dict:
        curr_class_dict = dict()
        values_dict = dict()
        attributes_dict[eachAttribute] = dict()
        for eachKey in normalized_locals:
            curr_class_dict = dict()
            if normalized_locals[eachKey][eachAttribute] not in values_dict:
                values_dict[normalized_locals[eachKey][eachAttribute]] = dict()
                curr_class_dict[topics_dict[eachKey][0]] = 1
                values_dict[normalized_locals[eachKey][eachAttribute]] = curr_class_dict
            else:
                curr_class_dict = values_dict[normalized_locals[eachKey][eachAttribute]]
                if topics_dict[eachKey][0] not in curr_class_dict:
                    curr_class_dict[topics_dict[eachKey][0]] = 1
                    values_dict[normalized_locals[eachKey][eachAttribute]] = curr_class_dict
                else:
                    curr_class_dict[topics_dict[eachKey][0]] = curr_class_dict[topics_dict[eachKey][0]] + 1
                    values_dict[normalized_locals[eachKey][eachAttribute]] = curr_class_dict
        attributes_dict[eachAttribute] = values_dict
        attribute_gain = 0
        for eachKey in attributes_dict[eachAttribute]:
            sum = 0
            for eachValue in attributes_dict[eachAttribute][eachKey].values():
                sum = sum + eachValue
            local_entropy = (sum / len(normalized_locals)) * entropy(attributes_dict[eachAttribute][eachKey], sum)
            attribute_gain = attribute_gain + local_entropy
        attribute_gain = global_entropy - attribute_gain
        attributes_gain_dict[eachAttribute] = attribute_gain
        attributes_gain_list.append(attribute_gain)
    most_significant_attributes = list()

    while len(most_significant_attributes) != round(len(global_dict) / 10):
        highest_gain = max(attributes_gain_list)
        keys_to_pop = list()
        for eachKey in attributes_gain_dict:
            if attributes_gain_dict[eachKey] == highest_gain and (
                    len(most_significant_attributes) != round(len(global_dict) / 10)):
                keys_to_pop.append(eachKey)
                most_significant_attributes.append(eachKey)

        while highest_gain in attributes_gain_list:
            attributes_gain_list.remove(highest_gain)
    with open("most_significant_attributes.txt", "w") as msa:
        json.dump(most_significant_attributes, msa)
    return most_significant_attributes


def local_dictionary_filtering(locals: dict, topics_dict: dict, most_significant_attributes: list):
    pop_from_local = list()
    for eachKey in locals:
        keys_to_pop = list()
        for eachAttribute in locals[eachKey]:
            if eachAttribute not in most_significant_attributes:
                keys_to_pop.append(eachAttribute)
        for eachItem in keys_to_pop:
            locals[eachKey].pop(eachItem)
        if len(locals[eachKey]) == 0:
            pop_from_local.append(eachKey)
    for eachItem in pop_from_local:
        locals.pop(eachItem)
        topics_dict.pop(eachItem)
    with open("local_dicts_final.txt", "w") as output_local:
        """
        for key in locals:
            output_local.write(key + '\n')
            output_local.write(json.dumps(locals[key], indent=5))
            output_local.write("\n")
        """
        json.dump(locals, output_local)
    with open("topics_final.txt", "w") as topics_file:
        """
        for key in topics_dict:
            topics_file.write(key + '\n')
            topics_file.write(json.dumps(topics_dict[key],indent=5) + '\n')
        """
        json.dump(topics_dict, topics_file)


def train_and_test(topics_dict: dict):
    trainig_list = list()
    testing_list = list()
    training_values = list()
    testing__values = list()
    with open("training_dataset.csv") as training_dataset:
        csv_reader = csv.reader(training_dataset, delimiter=',')
        header_line = True
        for row in csv_reader:
            if header_line:
                header_line = False
            else:
                training_values.append(topics_dict[row[0]][0])
                row = row[1:]
                for i in range(0, len(row)):
                    row[i] = int(row[i])
                trainig_list.append(row)

    with open("testing_dataset.csv") as testing_dataset:
        csv_reader = csv.reader(testing_dataset, delimiter=',')
        header_line = True
        for row in csv_reader:
            if header_line:
                header_line = False
            else:
                testing__values.append(topics_dict[row[0]][0])
                row = row[1:]
                for i in range(0, len(row)):
                    row[i] = int(row[i])
                testing_list.append(row)

    training_dataset = np.array(trainig_list)
    testing_dataset = np.array(testing_list)
    classifier = MultinomialNB()
    classifier.fit(training_dataset, training_values)
    return classifier.score(testing_dataset, testing__values)


def prediction_metrics(most_significant_attributes: list, locals: dict, topics_dict: dict):
    mean_accuracy = 0
    # for i in range(0,15):
    mean_accuracy += training_testing_dataset(locals, topics_dict, most_significant_attributes)
    return mean_accuracy


def training_testing_dataset(locals: dict, topics_dict: dict, most_significant_attributes: list):
    training_dataset = list()
    testing_dataset = list()
    training_dataset_attributes = list()
    training_dataset_classes = list()
    testing_dataset_attributes = list()
    testing_dataset_classes = list()
    while len(training_dataset) != round(0.7 * len(locals)):
        random_document = random.choice(list(locals.keys()))
        if random_document not in training_dataset:
            training_dataset.append(random_document)
    for eachKey in locals:
        if eachKey not in training_dataset:
            testing_dataset.append(eachKey)
    attributes = list()
    for eachItem in most_significant_attributes:
        attributes.append(eachItem)
    for eachKey in training_dataset:
        currentTrainingList = list()
        for currentItem in attributes:
            if currentItem in locals[eachKey]:
                currentTrainingList.append(locals[eachKey][currentItem])
            else:
                currentTrainingList.append(0)
        training_dataset_attributes.append(currentTrainingList)
        training_dataset_classes.append(topics_dict[eachKey][0])
    for eachKey in testing_dataset:
        currentTestingList = list()
        for currentItem in attributes:
            if currentItem in locals[eachKey]:
                currentTestingList.append(locals[eachKey][currentItem])
            else:
                currentTestingList.append(0)
        testing_dataset_attributes.append(currentTestingList)
        testing_dataset_classes.append(topics_dict[eachKey][0])
    training_dataset_matrix = np.array(training_dataset_attributes)
    testing_dataset_matrix = np.array(testing_dataset_attributes)
    classifier = MultinomialNB()
    classifier.fit(training_dataset_matrix, training_dataset_classes)
    predicted_values = classifier.predict(testing_dataset_matrix)
    print(predicted_values)
    print(testing_dataset_classes)
    # print(precision_recall_fscore_support(np.array(testing_dataset_classes), predicted_values, average = "weighted", zero_division = 1))
    return classifier.score(testing_dataset_matrix, testing_dataset_classes)



def preprocessing(files_root: str):
    locals = dict()  # local dictionaries
    global_dict = dict()  # global dictionary
    topics_dict = dict()  # topics dictionary
    normalized_locals = dict()  # normalized local dictionaries (score in range(0,3))
    classes_dict = dict()  # classes dictionary in our files

    prepare_dicts(get_stopwords('stop_words_english.txt'), files_root, locals, global_dict)
    extract_topics(files_root, topics_dict)
    update_dictionaries(locals, global_dict, topics_dict, 5, 95)
    topics_filtering(topics_dict)
    dataset_representation(global_dict, locals, normalized_locals)
    compute_classes_dict(topics_dict, classes_dict)
    global_entropy = entropy(classes_dict, len(normalized_locals))
    most_significant_attributes = information_gain(global_dict, normalized_locals, topics_dict, global_entropy)
    local_dictionary_filtering(locals, topics_dict, most_significant_attributes)


def classification_algo():
    with open("local_dicts_final.txt", "r") as local_dicts:
        locals = json.load(local_dicts)
    with open("topics_final.txt", "r") as topics:
        topics_dict = json.load(topics)
    with open("most_significant_attributes.txt") as msa:
        most_significant_attributes = json.load(msa)
    print(prediction_metrics(most_significant_attributes, locals, topics_dict))

#main
preprocessing('Reuters_34/Training/')
for i in range(0, 5):
    classification_algo()
