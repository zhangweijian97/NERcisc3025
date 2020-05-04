#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# --------------------------------------------------
# Description:
# --------------------------------------------------
# Author: Konfido <konfido.du@outlook.com>
# Created Date : April 4th 2020, 17:45:05
# Last Modified: April 4th 2020, 17:45:05
# --------------------------------------------------
# Second Author: Zhang Weijian <db62685@um.edu.mo>
# Last Modified: May 4th 2020

import nltk

nltk.download('names')
nltk.download('stopwords')
nltk.download('gazetteers')

from nltk.tokenize import word_tokenize
from nltk.corpus import names, stopwords, gazetteers
from nltk.classify.maxent import MaxentClassifier
from sklearn.metrics import (accuracy_score, fbeta_score, precision_score,
                             recall_score)
import os
import pickle


class MEM():
    def __init__(self):
        self.train_path = "../data/train"
        self.dev_path = "../data/dev"
        self.beta = 0
        self.max_iter = 0
        self.classifier = None
        self.nltk_names = set(names.words())
        self.nltk_stopwords = set(stopwords.words())
        self.titles = ['Master', 'Mr.', 'Mr', 'Miss.', 'Miss', 'Mrs.', 'Mrs', 'Ms.', 'Ms',
                       'Mx.', 'Mx', 'Sir', 'Gentleman', 'Sire', 'Mistress', 'Madam', 'Dame',
                       'Lord', 'Lady', 'Esq', 'Excellency', 'Dr', 'Professor', 'QC', 'Cl', 'SCl',
                       'Eur Lng', 'Chancellor', 'Vice-Chancellor', 'Principal', 'President', 'Minister',
                       'Warden', 'Dean', 'Regent', 'Rector', 'Provost', 'Director', 'Chief Executive',
                       'manager', 'chairman', 'secretary', 'leader']
        self.say = ['say', 'said', 'says']
                    # 'speak', 'spoke', 'speaks'
                    # 'talk', 'told', 'talks',
                    # 'discuss', 'discusses', 'discussed',
                    # 'mention', 'mentioned', 'mentions']
        self.gazetteers = set(gazetteers.words())

    def features(self, words, previous_label, position):
        """
        Note: The previous label of current word is the only visible label.

        :param words: a list of the words in the entire corpus
        :param previous_label: the label for position-1 (or O if it's the start
                of a new sentence)
        :param position: the word you are adding features for
        """

        features = {}
        """ Baseline Features """
        current_word = words[position]
        features['has_(%s)' % current_word] = 1
        features['prev_label'] = previous_label
        if current_word[0].isupper():
            features['Titlecase'] = 1

        # ===== TODO: Add your features here =======#
        if position > 0:
            previous_word = words[position - 1]
        else:
            previous_word = ''

        if position < len(words) - 1:
            after_word = words[position + 1]
        else:
            after_word = ''

        # ...

        # nltk corpus names
        if current_word in self.nltk_names:
            features['in_nltk_names'] = 1

        # # after stopword
        # if previous_word in self.nltk_stopwords:
        #     features['before_is_stopword'] = 1

        # # after title
        # # source: https://en.wikipedia.org/wiki/English_honorifics
        # if previous_word in self.titles:
        #     features['before_is_title'] = 1

        # somebody say
        # if previous_word in self.say:
        #     features['somebody_say'] = 1

        # is titlecase
        if current_word[0].isupper():
            # with say
            if previous_word in self.say or after_word in self.say:
                features['titlecase_and_say'] = 1

            # after stopword
            if previous_word in self.nltk_stopwords:
                features['titlecae_after_stopword'] = 1

            # with title
            if previous_word in self.titles:
                features['titlecase_with_title'] = 1

            if current_word in self.gazetteers:
                features['titlecase_is_gazetteers'] = -1

        # if current_word in self.gazetteers:
        #     features['is_gazetteers'] = 1

        # =============== TODO: Done ================#
        return features

    def load_data(self, filename):
        words = []
        labels = []
        for line in open(filename, "r", encoding="utf-8"):
            doublet = line.strip().split("\t")
            if len(doublet) < 2:  # remove emtpy lines
                continue
            words.append(doublet[0])
            labels.append(doublet[1])
        return words, labels

    def train(self):
        print('Training classifier...')
        words, labels = self.load_data(self.train_path)
        previous_labels = ["O"] + labels
        features = [self.features(words, previous_labels[i], i)
                    for i in range(len(words))]
        train_samples = [(f, l) for (f, l) in zip(features, labels)]
        classifier = MaxentClassifier.train(
            train_samples, max_iter=self.max_iter)
        self.classifier = classifier

    def test(self):
        print('Testing classifier...')
        words, labels = self.load_data(self.dev_path)
        previous_labels = ["O"] + labels
        features = [self.features(words, previous_labels[i], i)
                    for i in range(len(words))]
        results = [self.classifier.classify(n) for n in features]

        f_score = fbeta_score(labels, results, average='macro', beta=self.beta)
        precision = precision_score(labels, results, average='macro')
        recall = recall_score(labels, results, average='macro')
        accuracy = accuracy_score(labels, results)

        print("%-15s %.4f\n%-15s %.4f\n%-15s %.4f\n%-15s %.4f\n" %
              ("f_score=", f_score, "accuracy=", accuracy, "recall=", recall,
               "precision=", precision))

        return True

    def show_samples(self, bound):
        """Show some sample probability distributions.
        """
        words, labels = self.load_data(self.train_path)
        previous_labels = ["O"] + labels
        features = [self.features(words, previous_labels[i], i)
                    for i in range(len(words))]
        (m, n) = bound
        pdists = self.classifier.prob_classify_many(features[m:n])

        print('  Words          P(PERSON)  P(O)\n' + '-' * 40)
        for (word, label, pdist) in list(zip(words, labels, pdists))[m:n]:
            if label == 'PERSON':
                fmt = '  %-15s *%6.4f   %6.4f'
            else:
                fmt = '  %-15s  %6.4f  *%6.4f'
            print(fmt % (word, pdist.prob('PERSON'), pdist.prob('O')))

    def dump_model(self):
        with open('../model.pkl', 'wb') as f:
            pickle.dump(self.classifier, f)

    def load_model(self):
        with open('../model.pkl', 'rb') as f:
            self.classifier = pickle.load(f)

    def my_test_look_feature(self):
        words, labels = self.load_data(self.train_path)
        previous_labels = ["O"] + labels
        features = [self.features(words, previous_labels[i], i)
                    for i in range(len(words))]
        train_samples = [(f, l) for (f, l) in zip(features, labels)]

        print(features[:10])

        print(train_samples[:10])

    def my_test_predict_input_sentence(self):
        # fake_input_sentence = "William, Ethan, Roy are playing Switch at Ethan's home."
        # fake_input_sentence = "William Ethan, Roy"
        # fake_input_sentence = "Roy Ethan, William"
        fake_input_sentence = "William Ethan and Roy go to school by bus."

        words = word_tokenize(fake_input_sentence)
        predict_labels = ["" for i in range(len(words))]
        previous_labels = ["O"] + predict_labels

        for i in range(len(words)):
            word_features = self.features(words, previous_labels[i], i)
            pdists = self.classifier.prob_classify_many([word_features])
            for pdist in pdists:
                prob_person = pdist.prob('PERSON')
                prob_non_person = pdist.prob('O')
                if prob_person >= prob_non_person:
                    predict_labels[i] = 'PERSON'
                else:
                    predict_labels[i] = 'O'

                previous_labels[i + 1] = predict_labels[i]

                print(prob_person)
                print(prob_non_person)
            print(word_features)
        print(words)
        print(predict_labels)

    def predict_person(self, input):

        words = word_tokenize(input)
        predict_labels = ["" for i in range(len(words))]
        previous_labels = ["O"] + predict_labels

        for i in range(len(words)):
            word_features = self.features(words, previous_labels[i], i)
            pdists = self.classifier.prob_classify_many([word_features])
            for pdist in pdists:
                prob_person = pdist.prob('PERSON')
                prob_non_person = pdist.prob('O')
                if prob_person >= prob_non_person:
                    predict_labels[i] = 'PERSON'
                else:
                    predict_labels[i] = 'O'

                previous_labels[i + 1] = predict_labels[i]

        output_list = []

        for i in range(len(words)):
            if predict_labels[i] == 'PERSON':
                output_list.append(words[i])

        output_string = ' '.join(output_list)

        return output_string

    def predict_a_word(self,word):
        words = [word]
        word_features = self.features(words, '', 0)
        pdists = self.classifier.prob_classify_many([word_features])