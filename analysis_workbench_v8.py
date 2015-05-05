import numpy as np

import numpy.random as rnd


import itertools

import pandas as pd
import os


from time import time

import sys

from data_utils_v2 import *
import models_builder as mb
import data_handler as dh
# import derivatives_computation as dc

import json

import argparse

import scipy.stats as ss
import sklearn.preprocessing as skp

import hashlib

try:
    import cPickle as pickle
except:
    import pickle


def get_parser():

  parser = argparse.ArgumentParser()
  parser.add_argument('--start', help='flight to start from', type=int, default=95)
  parser.add_argument('--end', help='flight to end on', type=int, default=127)
  parser.add_argument('-v', help='if set the script would provide more detailed log', action='store_const', const=True, default=False)

  parser.add_argument('-t', help='technique to build models with', type=str, default=None)
  parser.add_argument('--model_options', help='options to build model with', type=str, default=None)
  parser.add_argument('--run_options', help='options to run model with', type=str, default=None) # TODO

  parser.add_argument('-ot', help='output transform to do', type=str, default='none')
  parser.add_argument('-it', help='input transform to do', type=str, default='none')

  parser.add_argument('--sample_name', help='sample to build model with', type=str, default='expert_v2')
  parser.add_argument('--min_train', help='minimum train sample size to use for analysis', type=int, default=5)
  parser.add_argument('--max_train', help='maximum train sample size to use for analysis', type=int, default=0)
  parser.add_argument('--min_cenz', help='value of lower output cenz', type=int, default=None)
  parser.add_argument('--max_cenz', help='value of upper output cenz', type=int, default=None)
  parser.add_argument('--min_filter', help='value of lower output filter', type=int, default=None)
  parser.add_argument('--max_filter', help='value of upper output filter', type=int, default=None)

  parser.add_argument('--dom', help='domain name to use', type=str, default='dom2')
  parser.add_argument('--join_domains', help='build separate model for each domain or not', action='store_const', const=True, default=False)

  parser.add_argument('--train', help='which sample use to train', type=str, default='exclude_current') # 'current', 'exclude_current', 'all'
  parser.add_argument('--test', help='which sample use to test', type=str, default='current')

  parser.add_argument('--not_build_models', help='forbid to build models', action='store_const', const=False, default=True)
  parser.add_argument('--rebuild_models', help='if models should be rebuilt', action='store_const', const=True, default=False)
  parser.add_argument('--not_save_predictions', help='save predictions of each model on test', action='store_const', const=False, default=True)
  parser.add_argument('--save_train', help='save train sample for each model', action='store_const', const=True, default=False)
  parser.add_argument('--save_test', help='save test sample for each model', action='store_const', const=True, default=False)

  parser.add_argument('--save_report', help='if report should be saved after script run', action='store_const', const=True, default=False)
  parser.add_argument('--draw_errors', help='if error_plots_should_be_drawn', action='store_const', const=True, default=False)
  parser.add_argument('--noise_statistics', help='computes noise statistics and draws noise plots', action='store_const', const=True, default=False)
  parser.add_argument('--calc_distances', help='draw pairwise distances plot', action='store_const', const=True, default=False)
  parser.add_argument('--neighbors_plot', help='if error_plots_should_be_drawn', action='store_const', const=True, default=False)
  parser.add_argument('--rsm_report', help='write rsm coefficients to report', action='store_const', const=True, default=False)
  parser.add_argument('--error_statistics', help='computes noise statistics and draws noise plots', action='store_const', const=True, default=False)
  parser.add_argument('--scatter_all', help='computes noise statistics and draws noise plots', action='store_const', const=True, default=False)



  return parser


def analysis_workbench(args=None):

  global_options = {}

  #########################################################################################
  ### OPTIONS ############################################################################
  #######################################################################################

  verbose = args['v']

  START_FLIGHT = args['start']
  END_FLIGHT = args['end']

  DOMAINS_TO_CONSIDER = [
    0,
    1,
    2,
    3
  ]

  BUILD_MODELS = args['not_build_models']
  REBUILD_MODELS_IF_EXIST = args['rebuild_models']

  DRAW_SCATTERS = False
  DRAW_ERROR_PLOTS = args['draw_errors']
  DRAW_RANGES = False

  CALC_DISTANCES = args['calc_distances']

  CHECK_DEPENDENCIES_WHOLE = False
  CHECK_DEPENDENCIES_PER_FLIGHT = False

  SAVE_REPORT = args['save_report']
  RSM_REPORT = args['rsm_report']

  SAVE_TRAIN = args['save_train']
  SAVE_TEST = args['save_test']
  SAVE_PREDICTIONS = args['not_save_predictions']

  NEIGHBORS_PLOT= args['neighbors_plot']

  global_options['sample/min_considered_test_size'] = args['min_train']
  global_options['sample/train_size_to_use'] = args['max_train']

  global_options['sample/output_column'] = ['output']
  global_options['sample/output_transform'] = args['ot'].lower()
  global_options['sample/input_transform'] = args['it'].lower()

  global_options['model/join_domains_for_train'] = args['join_domains']
  global_options['sample/set_nans'] = global_options['model/join_domains_for_train']
  # global_options['sample/set_nans'] = True

  global_options['noise_statistics'] = args['noise_statistics']
  global_options['error_statistics'] = args['error_statistics']
  global_options['scatter_all'] = args['scatter_all']

  global_options['sample/postfix'] = args['sample_name']
  global_options['experiment_name'] = args['dom'] # 'dom2'

  global_options['sample/train'] = args['train']
  global_options['sample/test'] = args['test']

  if not (args['min_filter'] is None and args['max_filter'] is None):
    global_options['sample/train_output_filter'] = [args['min_filter'], args['max_filter']]
  else:
    global_options['sample/train_output_filter'] = None

  if not (args['min_cenz'] is None and args['max_cenz'] is None):
    global_options['sample/train_output_cenz'] = [args['min_cenz'], args['max_cenz']]
  else:
    global_options['sample/train_output_cenz'] = None


  global_options['result/errors_filter'] = {
      'all': [None, None],
      'high_loads': [1200, None]
  }


  if args['dom'] == 'dom2':
    domains = {
        0: '(IAS <= 45) & (RA1_ALTI <= 100)',
        1: '(IAS <= 45) & (RA1_ALTI >= 100)',
        2: '(IAS >= 45) & (RA1_ALTI <= 100)',
        3: '(IAS >= 45) & (RA1_ALTI >= 100)',
    }

  elif args['dom'] == 'base':
    domains = {
        0: '(IAS <= 30) & (RA1_ALTI <= 100)',
        1: '(IAS <= 30) & (RA1_ALTI >= 100)',
        2: '(IAS >= 30) & (RA1_ALTI <= 100)',
        3: '(IAS >= 30) & (RA1_ALTI >= 100)',
    }

  inputs_to_ignore = {
      -1: ['domain', 'flight_name', 'TEMPS'],
      0: ['domain', 'flight_name', 'TEMPS', 'BETAI', 'IAS', 'Mu', 'MachTip'],
      1: ['domain', 'flight_name', 'TEMPS', 'BETAI', 'IAS', 'Mu', 'MachTip', 'RA1_ALTI'],
      2: ['domain', 'flight_name', 'TEMPS', 'Vy', 'Vx'],
      3: ['domain', 'flight_name', 'TEMPS', 'Vy', 'Vx', 'RA1_ALTI'],

  }


  #########################################################################################

  print args['model_options']
  if not args['model_options'] is None:
    model_options = json.loads(args['model_options'])
  else:
    model_options = {}

  global_options['model/method'] = args['t']
  global_options['model/options'] = model_options
  global_options['model/default_options'] = mb.ModelsBuilder('./', technique=global_options['model/method']).get_options()

  global_options['sample/inputs_to_ignore'] = inputs_to_ignore
  global_options['general/domains'] = domains

  ## Generate full experiment name

  pretty_global_options = dict_pretty_string(global_options)
  pretty_model_options = dict_pretty_string(model_options, new_line=False)

  string_hash = calc_hash(pretty_global_options)

  folder_name = global_options['sample/postfix']+'_'+global_options['experiment_name']

  if global_options['model/join_domains_for_train']:
      folder_name += '_joinTrain'

  if not global_options['sample/train'] == 'exclude_current':
      folder_name += '_train='+global_options['sample/train']

  if not global_options['sample/test'] == 'current':
      folder_name += '_test='+global_options['sample/test']

  if not global_options['sample/output_transform'] == 'none':
      folder_name += '_ot='+global_options['sample/output_transform']

  if not global_options['sample/input_transform'] == 'none':
      folder_name += '_it='+global_options['sample/input_transform']

  if not global_options['sample/train_size_to_use'] == 0:
      folder_name += '_tSize='+str(global_options['sample/train_size_to_use'])

  folder_name += '_'+global_options['model/method']

  if not global_options['sample/train_output_cenz'] is None:
    folder_name += '_tc'+str(global_options['sample/train_output_cenz'][0])+'-'+str(global_options['sample/train_output_cenz'][1])

  if not global_options['sample/train_output_filter'] is None:
      folder_name += '_tf'+str(global_options['sample/train_output_filter'][0])+'-'+str(global_options['sample/train_output_filter'][1])


  # Create necessary working folders and options file
  save_folder_path = './results/'+folder_name+'_{'+pretty_model_options+'}/'

  report_name = save_folder_path+folder_name+'_{'+pretty_model_options+'}.xls'

  pictures_folder = 'pictures/'
  models_folder = 'models/'
  samples_folder = 'samples/'
  predictions_folder = samples_folder + 'predictions/'
  train_folder = samples_folder + 'train/'
  test_folder = samples_folder + 'test/'
  cache_folder = 'cache/'

  for folder in ['', pictures_folder, models_folder, samples_folder, predictions_folder, train_folder, test_folder, cache_folder]:
    try:
      if not os.path.exists(save_folder_path+folder):
          os.makedirs(save_folder_path+folder)
          print'Creating directory:', save_folder_path+folder, '\n'
    except:
      pass

  try:
    with open(save_folder_path+'!options.data', "wb") as options_file:
        options_file.write(pretty_global_options)
  except:
    pass

  # Load data
  if verbose:
      print 'Started load data...'
  start_time = time()
  analyzer = dh.DataHandler(load_time=True, domains=domains, inputs_to_ignore=inputs_to_ignore)

  data = analyzer.get_data(postfix=global_options['sample/postfix'], mark_domains=True, verbose=verbose)
  if verbose:
      print 'Done. Elapsed time: ', time() - start_time

  # Create model builder object
  builder = mb.ModelsBuilder(save_folder_path+models_folder, technique=global_options['model/method'], approx_options=model_options, rebuild_if_exist=REBUILD_MODELS_IF_EXIST)

  # TODO: Preprocess data
  # data['RA1_ALTI'][data['RA1_ALTI']>=3500] = 3500

  ###
  ## Start experiment

  domains = np.unique(data['domain'])
  flight_names = np.unique(data['flight_name'])

  if verbose:
    print 'We consider'
    print '    domains: ', ', '.join([str(name) for name in domains])
    print

    print '    flight_names: ', ', '.join([str(name) for name in flight_names])
    print

  all_predictions = {}
  all_outputs = {}
  if RSM_REPORT:
      rsm_report = {}

  # inputs transformation if needed
  # if global_options['sample/input_transform'] == 'mapstd':
  #   parameters = data.columns.values

  #   for parameter in parameters:
  #     # if parameter in global_options['sample/output_column']:
  #     #   continue

  #     if parameter in ['flight_name', 'domain', 'TEMPS']:
  #       continue

  #     mean = np.mean(data[parameter], axis=0)
  #     std = np.std(data[parameter], axis=0)

  #     data[parameter] = (data[parameter] - mean)/std

  # for domain in domains:
  for domain in DOMAINS_TO_CONSIDER:

    current_domain_data = data[data['domain'] == domain]
    # data_to_plot = current_domain_data.copy()
    del current_domain_data['domain']

    input_names = list(current_domain_data.columns.values)
    input_names = remove_sublist_from_origin(input_names, analyzer.get_inputs_to_ignore(domain)+global_options['sample/output_column'])

    # all_errors[domain] = {}
    all_predictions[domain] = {}
    all_outputs[domain] = {}
    if RSM_REPORT:
      rsm_report[domain] = {}


    # for del_input in inputs_to_ignore[domain]:
    #   del data_to_plot[del_input]

    # pd.scatter_matrix(data_to_plot, alpha=0.2)

    # plt.show()

    # continue

    print 'Domain ', str(domain)
    print 'Whole sample size:', len(current_domain_data)
    print

    if DRAW_RANGES:
      draw_ranges_for_flights(current_domain_data, split_parameter='flight_name', title='Domain '+str(domain),
                              save_path='./pictures/ranges_per_flight/')

    if CHECK_DEPENDENCIES_WHOLE:

      inputs, output = dh.get_matrices(current_domain_data, input_columns_to_remove=analyzer.get_inputs_to_ignore(domain),
                                    output_columns=global_options['sample/output_column'])

      print 'Started partial pearson computation...'
      start_time = time()
      result = gtsda.Analyzer().check(x=inputs, y=output,
                                      options={
                                      'GTSDA/Checker/DependenceType':'linear',
                                      'GTSDA/Checker/Linear/ScoreType':"PearsonPartialCorrelation",
                                      'GTSDA/Checker/PValuesCalculationType':'asymptotic'
                                      })

      print 'Scores (ppc): ',
      print result.scores[0]

      print 'P_values: ',
      print result.p_values[0]
      print 'Done. Elapsed time: ', time() - start_time

      pd.scatter_matrix(current_domain_data)

      plt.show()


    all_inputs, all_output = dh.get_matrices(current_domain_data, input_columns_to_remove=analyzer.get_inputs_to_ignore(domain),
                                          output_columns=global_options['sample/output_column'])


    all_outputs_list = []
    all_noise_list = []

    if not BUILD_MODELS:
      break

    ## Flights loop
    for flight in flight_names:

      if (int(flight) < START_FLIGHT) or (int(flight) > END_FLIGHT):
        all_predictions[domain][flight] = np.ones((0,1))
        all_outputs[domain][flight] = np.ones((0,1))
        continue

      print 'Current flight', str(int(flight)), '(d'+str(domain)+')'
      print

      # Prepare the train data
      if global_options['model/join_domains_for_train']:

        # Here we decide which flights would be considered in train sample
        if global_options['sample/train'] == 'current':
          train_data = data[(data['flight_name'] == flight)]
        elif global_options['sample/train'] == 'exclude_current':
          train_data = data[(data['flight_name'] != flight)]
        elif global_options['sample/train'] == 'all':
          train_data = data[(data['flight_name'] == data['flight_name'])]

        all_train_inputs = []
        all_train_output = []
        all_train_time = []
        for train_domain in domains:
          temp_train_inputs, temp_train_output = dh.get_matrices(train_data[(train_data['domain'] == train_domain)],
                                                                 input_columns_to_remove=analyzer.get_inputs_to_ignore(train_domain),
                                                                 output_columns=global_options['sample/output_column'],
                                                                 set_nan=global_options['sample/set_nans'])

          all_train_inputs.append(temp_train_inputs)
          all_train_output.append(temp_train_output)
          all_train_time.append(train_data['TEMPS'][(train_data['domain'] == train_domain)][:, np.newaxis])

        train_inputs = np.vstack(tuple(all_train_inputs))
        train_output = np.vstack(tuple(all_train_output))
        train_time = np.vstack(tuple(all_train_time))

        if global_options['noise_statistics']:

          smoother_options = {}
          # smoother_options['kernelfunction'] = 'twohills' # 'simple', kernel function is usual RBF, alternative 'twohills' - turns to zero at zero and more suited fot noisy data
          smoother_options['alpha'] = 1. # smoothing coefficient, the greater it is the smoother result is. Should be lower than 1 when data contains pikes.
          # smoother_options['neighborhoodsbound'] = 40

          temp_train_time = np.copy(train_time)
          ind = np.argsort(train_time[:,0])
          smoothed_train_output = dc.DerivativeCalculate().smooth_outputs(x=train_time[ind,0], y=train_output[ind,:], options=smoother_options)

          noise = (train_output[ind,:] - smoothed_train_output)

          # print train_output.shape, smoothed_train_output.shape, noise.shape

          # print smoothed_train_output

          # fig = plt.figure()

          # plt.scatter(train_time, train_output)
          # # # ind = np.argsort(train_time[:,0])
          # # # plt.plot(train_time[ind,:], smoothed_train_output[ind,:], c='r')
          # plt.plot(train_time[ind,0], smoothed_train_output, c='r')

          # plt.show()
          # plt.close()

          fig = plt.figure()
          plt.scatter(train_output[ind,:], noise)

          q95p = np.percentile(noise, 95)
          q05p = np.percentile(noise, 05)
          maxp = np.percentile(noise, 100)
          minp = np.percentile(noise, 0)

          mm = [0., 1.1*np.max(train_output)]

          plt.gca().set_xlim(mm)

          plt.plot(mm, [maxp, maxp], c='m')
          plt.plot(mm, [q95p, q95p], c='r')
          plt.plot(mm, [q05p, q05p], c='r')
          plt.plot(mm, [minp, minp], c='m')

          plt.legend(['min-max', '90% confidence interval'])

          # df = pd.DataFrame(np.hstack((train_output[ind,:], noise)), columns=['load', 'noise'] )
          # fig = df.plot(kind='hexbin', x='load', y='noise', gridsize=200)
          # plt.show()
          # plt.title(title)

          plt.ylabel('true load - smoothed load')
          plt.xlabel('load')

          save_path = save_folder_path+'pictures/noise_fn_'+str(int(flight))
          fig.savefig(save_path)

          plt.close()

          all_outputs_list.append(train_output[ind,:])
          all_noise_list.append(noise)

          print 'q90', np.percentile(np.abs(noise), 90)
          print 'max', np.percentile(np.abs(noise), 100)

          continue
      else:

        # Here we decide which flights would be considered in train sample
        if global_options['sample/train'] == 'current':
          train_data = data[(data['flight_name'] == flight)&(data['domain'] == domain)]
        elif global_options['sample/train'] == 'exclude_current':
          train_data = data[(data['flight_name'] != flight)&(data['domain'] == domain)]
        elif global_options['sample/train'] == 'all':
          train_data = data[(data['domain'] == domain)]


        train_inputs, train_output = dh.get_matrices(train_data,
                                                     input_columns_to_remove=analyzer.get_inputs_to_ignore(domain),
                                                     output_columns=global_options['sample/output_column'],
                                                     set_nan=global_options['sample/set_nans'])


      if not global_options['sample/train_output_filter'] is None:
        if not global_options['sample/train_output_filter'][0] is None:
          idx = train_output[:, 0] >= global_options['sample/train_output_filter'][0]
          train_inputs, train_output = train_inputs[idx, :], train_output[idx, :]

        if not global_options['sample/train_output_filter'][1] is None:
          idx = train_output[:, 0] <= global_options['sample/train_output_filter'][0]
          train_inputs, train_output = train_inputs[idx, :], train_output[idx, :]

      train_size = len(train_inputs)

      if global_options['sample/input_transform'] == 'mapstd':

        for input_dim in xrange(train_inputs.shape[1]):
          mean = np.mean(train_inputs[:, input_dim], axis=0)
          std = np.std(train_inputs[:, input_dim], axis=0)

          train_inputs[:, input_dim] = (train_inputs[:, input_dim] - mean)/std

      if global_options['sample/input_transform'] == 'mapminmax':

        for input_dim in xrange(train_inputs.shape[1]):
            min_value = np.min(train_inputs[:, input_dim], axis=0)
            max_value = np.max(train_inputs[:, input_dim], axis=0)

            if min_value == max_value:
              train_inputs[:, input_dim] = 0.5
              continue

            train_inputs[:, input_dim] = (train_inputs[:, input_dim] - min_value)/(max_value - min_value)

      if not global_options['sample/train_size_to_use'] == 0 and train_size > global_options['sample/train_size_to_use']:
          idxs = np.array(range(train_size))
          rnd.shuffle(idxs)
          train_inputs = train_inputs[idxs[:global_options['sample/train_size_to_use']], :]
          train_output = train_output[idxs[:global_options['sample/train_size_to_use']], :]

          train_size = global_options['sample/train_size_to_use']

      # Prepare the test data
      # Here we decide which flights would be considered in train sample
      if global_options['sample/test'] == 'current':
        test_data = data[(data['flight_name'] == flight)&(data['domain'] == domain)]
      elif global_options['sample/test'] == 'exclude_current':
        test_data = data[(data['flight_name'] != flight)&(data['domain'] == domain)]
      elif global_options['sample/test'] == 'all':
        test_data = data[(data['domain'] == domain)]

      test_inputs, test_output = dh.get_matrices(test_data, input_columns_to_remove=analyzer.get_inputs_to_ignore(domain),
                                              output_columns=global_options['sample/output_column'], set_nan=global_options['sample/set_nans'])

      test_time = test_data['TEMPS']

      if not global_options['sample/train_output_cenz'] is None:
        if not global_options['sample/train_output_cenz'][1] is None:
          train_output = np.minimum(train_output, global_options['sample/train_output_cenz'][1])
          test_output = np.minimum(test_output, global_options['sample/train_output_cenz'][1])

        if not global_options['sample/train_output_cenz'][0] is None:
          train_output = np.maximum(train_output, global_options['sample/train_output_cenz'][0])
          test_output = np.maximum(test_output, global_options['sample/train_output_cenz'][0])



      # Output transform
      if 'mapstd' in global_options['sample/output_transform']:

          mean = np.mean(train_output, axis=0)
          std = np.std(train_output, axis=0)

          train_output = (train_output - mean)/std

      if 'log' in global_options['sample/output_transform']:
        train_output = np.log(train_output)

      test_size = len(test_data)

      if verbose:
        print 'train shape', train_inputs.shape, train_output.shape
        print 'test shape', test_inputs.shape, test_output.shape
        print


      if test_size < global_options['sample/min_considered_test_size']:
        all_predictions[domain][flight] = np.ones((0,1))
        all_outputs[domain][flight] = np.ones((0,1))
        print '[w] Test size is too small, so flight is skipped'
        print
        continue

      if False:
        if verbose:
          print 'Started box...'
          start_time = time()
        box = get_box(test_inputs, split_parameter='flight_name')
        in_box = check_if_in_box(test_inputs, box)

        save_path = save_folder_path+'classify_'+model_name
        check_classify(train_data, test_data, model_type='logistic_regression', weighted=True, title='Classification results for flight '+str(flight), save_path=save_path)

        if verbose:
          print 'Done. Elapsed time: ', time() - start_time
          print


      # Check if predictions are already computed otherwise run model building
      predictions_file = save_folder_path+predictions_folder+'d_'+str(domain)+'_fn_'+str(flight)+'.csv'

      if not os.path.exists(predictions_file):
        if global_options['model/join_domains_for_train']:
          model_name = 'd_all'+'_fn_'+str(int(flight))
        else:
          model_name = 'd_'+str(domain)+'_fn_'+str(int(flight))

        if global_options['sample/output_transform']:
          model_name += '_'+global_options['sample/output_transform']

        if verbose:
          print 'Started build model...'
          start_time = time()

        model = builder.build_model(model_name, train_inputs, train_output)

        if RSM_REPORT:

          model_weights = list(model.details['models'][0]['weights'][0])
          model_design = model.details['models'][0]['design_matrix']
          model_bias = model.details['models'][0]['bias_weight'][0]

          # print model_design
          current_model = {}
          current_model['bias'] = model_bias

          for term_idx in xrange(len(model_weights)):
            active_features = []
            for feature_idx in xrange(model_design.shape[0]):
              if int(model_design[feature_idx, term_idx]) == 1:
                active_features.append(input_names[feature_idx])
              if int(model_design[feature_idx, term_idx]) == 2:
                active_features.append(input_names[feature_idx]+'^2')
            current_model[' * '.join(active_features)] = model_weights[term_idx]

          # print current_model
          rsm_report[domain][flight] = current_model

        # print details

        if verbose:
          print 'Done. Elapsed time: ', time() - start_time
          print

        if verbose:
          print 'Computing predictions...'
          start_time = time()

        prediction = builder.compute_predictions(model_name, test_inputs)

        # Output inverse transform
        if 'log' in global_options['sample/output_transform']:
          prediction = np.exp(prediction)

        if SAVE_PREDICTIONS:
          # df = pd.DataFrame(prediction, columns=['prediction'])
          # df.to_csv(save_folder_path+predictions_folder+'d_'+str(domain)+'_fn_'+str(flight)+'.csv')
          if verbose:
            print 'Save predictions...',
          np.savetxt(predictions_file, prediction, delimiter=',')
          if verbose:
            print 'Done!'

        if verbose:
          print 'Done. Elapsed time: ', time() - start_time
          print

      else:
        if verbose:
          print 'Load saved predictions...',

        prediction = np.loadtxt(predictions_file, delimiter=',')[:, np.newaxis]
        if verbose:
          print 'Done!'


      error_metrics = mb.compute_errors(test_output, prediction, error_filters=global_options['result/errors_filter'])

      # print 'Test errors: '
      # print dict_pretty_string(error_metrics)
      # print

      error_string0 = "%.2f" % error_metrics['all']['q95re']
      error_string1 = "%.2f" % error_metrics['high_loads']['q95re']
      title = 'domain '+str(domain)+' flight '+str(int(flight))+'\nq95re '+error_string0+' '+error_string1+'\ntrain: '+str(train_size)+' test: '+str(test_size)

      if SAVE_REPORT or NEIGHBORS_PLOT:
        all_predictions[domain][flight] = prediction
        all_outputs[domain][flight] = test_output


      if DRAW_SCATTERS:
        save_path = save_folder_path+'pictures/'+'error_scatter_'+'d_'+str(domain)+'_fn_'+str(int(flight))
        if verbose:
          print 'Draw scatters...'
          start_time = time()
        draw_error_scatters(test_data, prediction, save_path=save_path, title=title)
        if verbose:
          print 'Done. Elapsed time: ', time() - start_time
          print

      if DRAW_ERROR_PLOTS:
        save_path = save_folder_path+'pictures/'+'d_'+str(domain)+'_fn_'+str(int(flight))
        if verbose:
          print 'Draw plots...'
          start_time = time()
        draw_errors_plot_v2(test_output, prediction, save_path=save_path, title=title)
        if verbose:
          print 'Done. Elapsed time: ', time() - start_time
          print

  if global_options['sample/postfix'] == 'expert_v2':
    load_name = 'dynamic MR pitch rod load'
  elif global_options['sample/postfix'] == 'expert_v4':
    load_name = 'static MR pitch rod load'
  elif global_options['sample/postfix'] == 'expert_v5':
    load_name = 'mass bending moment'
  else:
    load_name = 'load'



  if global_options['noise_statistics']:

    all_outputs = np.vstack(tuple(all_outputs_list))
    all_noise = np.vstack(tuple(all_noise_list))

    plt.hist(all_noise/all_outputs, bins=100, color='blue')

    plt.show()

    plt.hist(all_noise[all_outputs[:,0]>1200,:]/all_outputs[all_outputs[:,0]>1200,:], bins=100, color='blue')

    plt.show()

    df = pd.DataFrame(np.hstack((train_output[ind,:], noise)), columns=['load', 'noise'] )

    fig = df.plot(kind='hexbin', x='load', y='noise', gridsize=200)

    plt.show()

    fig = plt.figure()
    plt.scatter(train_output[ind,:], noise)

    q95p = np.percentile(noise, 95)
    q05p = np.percentile(noise, 05)
    maxp = np.percentile(noise, 100)
    minp = np.percentile(noise, 0)

    mm = [0., 1.1*np.max(train_output)]

    plt.gca().set_xlim(mm)

    plt.plot(mm, [maxp, maxp], c='m')
    plt.plot(mm, [q95p, q95p], c='r')
    plt.plot(mm, [q05p, q05p], c='r')
    plt.plot(mm, [minp, minp], c='m')

    plt.legend(['min-max', '90% confidence interval'])

    plt.xlabel('load value')
    plt.ylabel('true load - smoothed load')

    save_path = save_folder_path+'pictures/noise_all'
    fig.savefig(save_path)

    plt.close()

    print 'q90', np.percentile(np.abs(all_noise), 90)
    print 'max', np.percentile(np.abs(all_noise), 100)

  if global_options['scatter_all']:

    domains = all_outputs.keys()

    all_outputs_joint = []
    all_predictions_joint = []
    for domain in domains:
      flights = all_outputs[domain].keys()
      for flight in flights:
        all_outputs_joint.append(all_outputs[domain][flight])
        all_predictions_joint.append(all_predictions[domain][flight])


    all_outputs_joint = np.vstack(all_outputs_joint)
    all_predictions_joint = np.vstack(all_predictions_joint)

    all_errors_joint = np.abs(all_predictions_joint - all_outputs_joint)

    fig = plt.figure()
    plt.scatter(np.abs(all_outputs_joint), np.abs(all_predictions_joint), s=0.01)

    q95p = np.percentile(all_errors_joint, 95)
    q05p = np.percentile(all_errors_joint, 05)
    maxp = np.percentile(all_errors_joint, 100)
    minp = np.percentile(all_errors_joint, 0)

    mm0 = [np.min((np.min(all_predictions_joint),np.min(all_outputs_joint))), np.max((np.max(all_predictions_joint),np.max(all_outputs_joint)))]

    mm = np.array([0., np.max(np.abs(all_outputs_joint)+700)])

    # mm = [0., 1.1*np.max(all_outputs_joint)]

    # plt.gca().set_xlim(mm)

    # plt.plot(mm, [maxp, maxp], c='m')
    # plt.plot(mm, [q95p, q95p], c='r')
    # plt.plot(mm, [q05p, q05p], c='r')
    # plt.plot(mm, [minp, minp], c='m')

    plt.plot(mm, mm, c='black')
    plt.plot(mm, 1.1*mm, c='g')
    plt.plot(mm, 1.2*mm, c='r')
    plt.plot(mm, 1.5*mm, c='c')
    plt.plot(mm, 2.*mm, c='m')
    plt.plot(mm, mm/1.1, c='g')
    plt.plot(mm, mm/1.2, c='r')
    plt.plot(mm, mm/1.5, c='c')
    plt.plot(mm, mm/2., c='m')

    plt.xlim(mm)
    plt.ylim(mm)

    plt.legend(['0%', '10%', '20%', '50%', '100%'])

    plt.xlabel('prediction')
    plt.ylabel('true')

    # plt.legend(['min-max', '90% confidence interval'])

    plt.xlabel('true '+load_name)
    plt.ylabel('predicted '+load_name)

    save_path = save_folder_path+'pictures/scatters_all'

    fig.savefig(save_path)
    plt.close()

    # fig = plt.figure()

    # plt.xlabel('prediction')
    # plt.ylabel('true')

    # # heatmap, xedges, yedges = np.histogram2d(all_outputs_joint[:,0], all_predictions_joint[:,0], bins=100)
    # # extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    # # plt.clf()
    # plt.hexbin(all_outputs_joint[:,0], all_predictions_joint[:,0])
    # plt.axis([mm[0], mm[1], mm[0], mm[1]])

    # # plt.imshow(heatmap, extent=extent)
    # fig.savefig(save_path+'_heat')
    # plt.close()

    print 'q90', q05p
    print 'max', maxp

  if global_options['error_statistics']:
    domains = all_outputs.keys()

    all_outputs_joint = []
    all_predictions_joint = []
    for domain in domains:
      flights = all_outputs[domain].keys()
      for flight in flights:
        all_outputs_joint.append(all_outputs[domain][flight])
        all_predictions_joint.append(all_predictions[domain][flight])


    all_outputs_joint = np.vstack(all_outputs_joint)
    all_predictions_joint = np.vstack(all_predictions_joint)

    idx = all_outputs_joint[:,0] > 30
    all_errors_joint = np.abs(all_predictions_joint[idx, :] - all_outputs_joint[idx, :])/np.abs(all_outputs_joint[idx, :])

    fig = plt.figure()
    plt.scatter(np.abs(all_outputs_joint[idx, :]), all_errors_joint, s=0.01)

    q95p = np.percentile(all_errors_joint, 95)
    q05p = np.percentile(all_errors_joint, 05)
    maxp = np.percentile(all_errors_joint, 100)
    minp = np.percentile(all_errors_joint, 0)


    mm = [0., np.max(np.abs(all_outputs_joint)+700)]

    # plt.gca().set_xlim(mm)

    plt.plot(mm, [maxp, maxp], c='m')
    plt.plot(mm, [q95p, q95p], c='r')
    plt.plot(mm, [q05p, q05p], c='r')
    plt.plot(mm, [minp, minp], c='m')

    plt.xlim(mm)
    # plt.ylim(mm)

    # plt.legend(['data', '0%', '10%', '20%', '50%', '100%'])

    plt.legend(['min-max', '90% confidence interval'])

    plt.xlabel('true '+load_name)
    plt.ylabel('relative error')

    save_path = save_folder_path+'pictures/errors_all'
    fig.savefig(save_path)
    plt.close()

    # fig = plt.figure()

    # plt.xlabel('true')
    # plt.ylabel('absolute error')

    # # heatmap, xedges, yedges = np.histogram2d(all_outputs_joint[:,0], all_errors_joint[:,0], bins=100)
    # # extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    # plt.hexbin(all_outputs_joint[:,0], all_errors_joint[:,0])
    # plt.axis([mm[0], mm[1], all_errors_joint[:,0].min(), all_errors_joint[:,0].max()])

    # # plt.clf()
    # # plt.imshow(heatmap, extent=extent)
    # fig.savefig(save_path+'_heat')
    # plt.close()

    print 'q90', q05p
    print 'max', maxp


  if RSM_REPORT:
    # print rsm_report

    writer = pd.ExcelWriter(save_folder_path+'rsm_report.xls')
    for domain, rsm_models in rsm_report.items():
      all_keys = []
      for flight, rsm_model in rsm_models.items():
        for coefficient in rsm_model.keys():
          if not coefficient in all_keys:
            all_keys.append(coefficient)

      # print all_keys

      all_keys.sort()
      all_keys = ['flight']+all_keys

      df = pd.DataFrame(columns=all_keys)

      for flight, rsm_model in rsm_models.items():
        row_to_add = []
        for coefficient in all_keys:
          if coefficient == 'flight':
            row_to_add.append(int(flight))
          elif coefficient in rsm_model.keys():
            row_to_add.append(float(rsm_model[coefficient]))
          else:
            row_to_add.append(None)
        # print row_to_add
        df.loc[len(df)] = row_to_add

      df.to_excel(writer, 'Domain '+str(domain), float_format='%.3f', startrow=0, startcol=0, index=False)# , float_format='%.3f'
    writer.save()


  if SAVE_REPORT:
    mb.save_errors_tables(report_name, all_outputs, all_predictions, global_options['result/errors_filter'])


  if NEIGHBORS_PLOT:

    all_inputs, all_output = dh.get_matrices(data[data['domain']==2], input_columns_to_remove=analyzer.get_inputs_to_ignore(2),
                                        output_columns=global_options['sample/output_column'])

    NORMALIZE = True

    if NORMALIZE:
      all_inputs = skp.normalize(all_inputs)

    occuring_flight_names = np.unique(current_domain_data['flight_name'])

    flight_ranks = ss.rankdata(occuring_flight_names)

    print occuring_flight_names
    print flight_ranks

    import itertools

    # colors = ['r','g','b','c','m','y','white',[0.7,0.7,0.7]]
    colors = ['black']

    # markers = ['o','D','v']
    markers = ['$'+str(int(name))+'$' for name in flight_ranks]

    merged_mc = itertools.product(markers, colors)

    no_dims = 2

    compressor_options = {'no_dims': no_dims, 'verbose':True, 'perplexity': 50, 'theta': 0.7}

    compressed_inputs_hash = hashlib.sha1(pickle.dumps(all_inputs)+pickle.dumps(compressor_options)).hexdigest()

    if os.path.exists(save_folder_path+cache_folder+compressed_inputs_hash+'.cache'):
      print 'Loading compressed data from cache'
      with open(save_folder_path+cache_folder+compressed_inputs_hash+'.cache', 'rb') as cache:
        compressed_inputs = pickle.load(cache)
    else:
      compressor = bhtsne.BHTsne()
      compressor.set_options(compressor_options)

      t0_compress = time()
      compressed_inputs = compressor.calc(all_inputs)
      print 'Elapsed time ',(time() - t0_compress)

      with open(save_folder_path+cache_folder+compressed_inputs_hash+'.cache', 'wb') as cache:
        pickle.dump(compressed_inputs, cache)

    fig = plt.figure()
    from mpl_toolkits.mplot3d import Axes3D

    if no_dims == 2:
      ax = fig.add_subplot(111)
    if no_dims == 3:
      ax = fig.add_subplot(111, projection='3d')

    plt.title('domain '+str(domain)+(' (normalized)' if NORMALIZE else ''))
    for flight in occuring_flight_names:
      idx = np.where(current_domain_data['flight_name'] == flight)[0]
      # print flight, idx
      plot_options = merged_mc.next()
      # print all_outputs.shape
      # plt.scatter(compressed_inputs[idx, 0], compressed_inputs[idx, 1], c=plot_options[1], marker=plot_options[0], s=50)
      if no_dims == 2:
        ax.scatter(compressed_inputs[idx, 0], compressed_inputs[idx, 1], color=rnd.rand(3,), marker=plot_options[0], s=all_output[idx,0]/10)
      if no_dims == 3:
        ax.scatter(compressed_inputs[idx, 0], compressed_inputs[idx, 1], compressed_inputs[idx, 2], color=rnd.rand(3,), marker=plot_options[0], s=50)#np.abs(all_outputs[domain][flight][idx,0]-all_predictions[domain][flight][idx,0]))
      # plt.scatter(compressed_inputs[idx, 0], compressed_inputs[idx, 1], c=rnd.rand(3,), marker=plot_options[0])

    plt.legend([int(name) for name in occuring_flight_names])

    plt.show()


  return

if __name__ == '__main__':
    parser = get_parser()
    args = vars(parser.parse_args())

    analysis_workbench(args)
