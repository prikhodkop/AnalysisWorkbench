import numpy as np

from da.macros import gtapprox

# from numbapro import autojit

from scipy.spatial import Delaunay
import sklearn.linear_model as lm
import sklearn.svm as svm
from sklearn.metrics import roc_curve, auc

import scipy.spatial.distance as scd

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import copy

import hashlib

import dill as pickle


class DataHandler(object):
    def __init__(self,
                 path_to_data='./ready_samples/',
                 load_time=True,
                 domains=None,
                 inputs_to_ignore=None,
                 verbose=False):

        self.path_to_data = path_to_data
        self.load_time = load_time
        if domains is None:
            self.domains = {
                0: {'IAS': {'<=': 45}, 'RA1_ALTI': {'<': 100, '>=': 3}},
                1: {'IAS': {'<=': 45}, 'RA1_ALTI': {'>=': 100}},
                2: {'IAS': {'>=': 45}, 'RA1_ALTI': {'<': 100, '>=': 3}},
                3: {'IAS': {'>=': 45}, 'RA1_ALTI': {'>=': 100}},
                4: {'IAS': {'<=': 45}, 'RA1_ALTI': {'<': 3}},
                5: {'IAS': {'>=': 45}, 'RA1_ALTI': {'<': 3}},
                }
        else:
            self.domains = domains

        if inputs_to_ignore is None:
            self.inputs_to_ignore = {
                0: ['flight_name', 'TEMPS', 'BETAI', 'IAS', 'Mu', 'MachTip'],
                1: ['flight_name', 'TEMPS', 'BETAI', 'IAS', 'Mu', 'MachTip', 'RA1_ALTI'],
                2: ['flight_name', 'TEMPS', 'Vy', 'Vx'],
                3: ['flight_name', 'TEMPS', 'Vy', 'Vx', 'RA1_ALTI'],
                4: ['flight_name', 'TEMPS', 'BETAI', 'IAS', 'Mu', 'MachTip'],
                5: ['flight_name', 'TEMPS', 'Vy', 'Vx']
            }
        else:
            self.inputs_to_ignore = inputs_to_ignore

        self.verbose = verbose

    def prepare_data(self, postfix='expert', verbose=False):
        '''
        Loads flight data samples
        '''

        data = pd.DataFrame.from_csv(self.path_to_data+'sample_joint_'+postfix+'.csv')

        if not self.load_time:
            del data['TEMPS']

        return data

    def save_data(self, data, postfix='expert'):
        '''
        May be used to save modified data sample
        '''
        data.to_csv(self.path_to_data+'sample_joint_'+postfix+'.csv')

    def set_domains(self, domains):
        self.domains = copy.deepcopy(domains)

    def form_query(self, rule):
        queries = []

        for variable in rule.keys():
            for condition in rule[variable].keys():
                queries.append('('+variable+condition+str(rule[variable][condition])+')')

        joint_query = '&'.join(queries)

        return joint_query

    def mark_domains(self, data, verbose=False):

        domain_names = self.domains.keys()

        copies = None

        for domain in domain_names:
            if type(self.domains[domain]) == str:
                query = self.domains[domain]
            elif type(self.domains[domain]) == dict:
                query = self.form_query(self.domains[domain])
            else:
                raise Exception('Wrong domain definition!')

            copy = data.query(query)
            if verbose:
                print 'domain', domain
                print 'query:', query
                print 'contatins', len(copy), 'points'
                print

            copy['domain'] = domain

            if copies is None:
                copies = copy
            else:
                copies = copies.append(copy)

        return copies

    def get_inputs_to_ignore(self, domain):
        return self.inputs_to_ignore[domain]

    def get_data(self, postfix, mark_domains=True, verbose=False):
        '''
          Allows to get flight data.
        '''

        data = self.prepare_data(postfix=postfix, verbose=verbose)

        if mark_domains:
            data = self.mark_domains(data, verbose=verbose)

        if verbose:
            print 'Total number of points selected: ', len(data)
            print

        return data

def get_matrices(sample, input_columns_to_remove=['flight_name', 'TEMPS'], output_columns=['output'], set_nan=False, nan_value=-999):

    sample = sample.copy()
    # try:
    for input_column_to_remove in input_columns_to_remove:
        if not input_column_to_remove in sample.columns.values:
            continue
        if set_nan and not input_column_to_remove in ['flight_name', 'TEMPS']:
            sample[input_column_to_remove] = nan_value
        else:
            del sample[input_column_to_remove]

    outputs = sample[output_columns].as_matrix()

    for output_column in output_columns:
        del sample[output_column]

    inputs = sample.as_matrix()

    return inputs, outputs