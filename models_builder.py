import numpy as np

# from numbapro import autojit

from sklearn import ensemble, tree

import os
import copy

import dill as pickle

import pandas as pd

import sys

sys.path.append('.models')

try:
  sys.path.append('./utils/xgboost-master/wrapper/')
  from da.macros import gtapprox
  MACROS_IMPORTED = True
except:
  MACROS_IMPORTED = False

try:
  from pyearth import Earth
  PYEARTH_IMPORTED = True
except:
  PYEARTH_IMPORTED = False
  pass

try:
  sys.path.append('./utils/xgboost-master/wrapper/')
  import xgboost as xg
  XGBOOST_IMPORTED = True
except:
  XGBOOST_IMPORTED = False

try:
  sys.path.append('./utils/xgboost_python_windows_x64/python/')
  import xgboost as xg
  XGBOOST_IMPORTED = True
except:
  XGBOOST_IMPORTED = False


if XGBOOST_IMPORTED:
  class Xgboost(object):
    '''
    Configures xgboost tree to be used for regression
    '''
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=10,
                 subsample=0.8, scale_factor=1, objective='reg:linear',
                 missing=-999., gamma=1., eval_metric='rmse'):
      param = {}
      param['objective'] = objective
      param['bst:eta'] = learning_rate
      param['bst:gamma'] = gamma
      param['bst:max_depth'] = max_depth
      param['nthread'] = 4
      param['bst:subsample'] = subsample
      param['eval_metric'] = eval_metric
      param['bst:silent'] = 1
      param['scale_pos_weight'] = 1.


      self.param = param
      self.num_round = n_estimators
      self.scale_factor = scale_factor
      self.missing = missing
      self.bst = xg.Booster()

    def fit(self, inputs, outputs, weights=None):
      '''
      Fits tree to given training sample
      '''
      plst = list(self.param.items())

      xgmat = xg.DMatrix(inputs, label=outputs, missing=self.missing, weight=weights)
      self.bst = xg.train(plst, xgmat, self.num_round)

    def predict(self, inputs):
      '''
      Predicts output value with constructed tree
      '''
      xgmat = xg.DMatrix(inputs, missing=self.missing)
      return self.bst.predict(xgmat)

    def save(self, name):
      '''
      Saves constructed tree to file
      '''
      self.bst.save_model(name)

    def load(self, name):
      '''
      Loads constructed tree from file
      '''
      self.bst.load_model(name)


class ModelsBuilder(object):
  '''
  Class configures, builds, saves and evaluates regression models
  '''
  def __init__(self, models_save_path, technique, approx_options=None, rebuild_if_exist=False):
    self.models_save_path = models_save_path
    self.rebuild_if_exist = rebuild_if_exist

    if technique is None:
      raise Exception("Please specify technique to build models with!")

    elif technique == 'xgboost':
      self.approx_technique = 'xgboost'
      self.approx_options = {
          'n_estimators': 500,
          'learning_rate': 0.1,
          'max_depth': 10,
          'subsample': 0.8,
          'eval_metric': 'rmse'
          }
      self.run_options = { ## currently not supported
          'ntree_limit': 1500
          }
      self.ext = '.bst'

    elif technique == 'quadxgboost':
      self.approx_technique = 'quadxgboost'
      self.approx_options = {
          'n_estimators': 500,
          'learning_rate': 0.1,
          'max_depth': 5,
          'subsample': 0.9,
          'eval_metric': 'rmse'
          }
      self.run_options = { ## currently not supported
          'ntree_limit': 1500
          }
      self.ext = '.bst'

    elif technique == 'skgboost':
      self.approx_technique = 'skgboost'
      self.approx_options = {
          'n_estimators': 1000,
          'learning_rate': 0.1,
          'max_depth': 10,
          'subsample': 0.8,
          'loss': 'lad'
          }
      self.ext = '.skgb'

    elif technique == 'linear':
      self.approx_technique = 'rsm'
      rsm_type = 'linear'
      self.approx_options = {
          'gtapprox/technique': self.approx_technique,
          'gtapprox/rsmtype': rsm_type,
          'gtapprox/rsmmapping': 'none',
          "gtapprox/rsmfeatureselection": "MultipleRidgeLS"
          }
      self.ext = '.gta'
    elif technique == 'linear_ridge':
      self.approx_technique = 'rsm'
      rsm_type = 'linear'
      self.approx_options = {
          'gtapprox/technique': self.approx_technique,
          'gtapprox/rsmtype': rsm_type,
          'gtapprox/rsmmapping': 'none',
          "gtapprox/rsmfeatureselection": "RidgeLS"
          }
      self.ext = '.gta'
    elif technique == 'quadratic':
      self.approx_technique = 'rsm'
      rsm_type = 'quadratic'
      self.approx_options = {
          'gtapprox/technique': self.approx_technique,
          'gtapprox/rsmtype': rsm_type,
          'gtapprox/rsmmapping': 'none',
          "gtapprox/rsmfeatureSelection": "MultipleRidgeLS"
          }
      self.ext = '.gta'
    elif technique == 'quadratic_ridge':
      self.approx_technique = 'rsm'
      rsm_type = 'quadratic'
      self.approx_options = {
          'gtapprox/technique': self.approx_technique,
          'gtapprox/rsmtype': rsm_type,
          'gtapprox/rsmmapping': 'none',
          "gtapprox/rsmfeatureSelection": "RidgeLS"
          }
      self.ext = '.gta'

    elif technique == 'hda':
      self.approx_technique = 'hda'
      self.approx_options = {
          'gtapprox/technique': self.approx_technique,
          'gtapprox/hdamultimin': 1,
          'gtapprox/hdamultimax': 1,
          'gtapprox/hdaphasecount': 1,
          'gtapprox/hdapmax': 450,
          'gtapprox/hdapmin': 450,
          # '/HDA/LM/MaxIterations':1,
          '/hda/hpaenabled': 0
      }
      self.ext = '.gta'

    elif technique == 'moa':
      self.approx_technique = 'moa'
      self.approx_options = {
          'gtapprox/technique': self.approx_technique,
          'gtapprox/moatypeofweights': 'sigmoid',
          'gtapprox/moapointsassignment': 'mahalanobis',
          'gtapprox/moanumberofclusters': [40],
          'gtapprox/moacovariancetype': 'diag',
          'gtapprox/moatechnique': 'hda',
          'gtapprox/hdamultimin': 1,
          'gtapprox/hdamultimax': 1,
          'gtapprox/hdaphasecount': 1,
          'gtapprox/hdapmax': 50,
          'gtapprox/hdapmin': 50,
          # '/HDA/LM/MaxIterations':1,
          '/hda/hpaenabled': 0
      }
      self.ext = '.gta'

    elif technique == 'zeros':
      self.approx_technique = 'zeros'
      self.approx_options = {
      }
      self.ext = '.zero'

    elif technique == 'means':
      self.approx_technique = 'means'
      self.approx_options = {
      }
      self.ext = '.mean'

    else:
      raise Exception("Wrong technique specifed!")

    if not approx_options is None:
      for key in approx_options.keys():
        self.approx_options[key.lower()] = approx_options[key]

  def get_options(self):
    '''
    Gets current options for approximation
    '''
    return self.approx_options

  # @autojit
  def build_model(self, model_name, inputs, outputs):
    '''
    Builds model of selected type
    '''
    approx_options = copy.copy(self.approx_options)

    # skip building model if it exists and rebuild option is off
    if not self.rebuild_if_exist and os.path.exists(self.models_save_path+model_name+self.ext):
      print 'Model exists, so no rebuilding'
      return

    # set weights if corresponding option is specified
    weights = None
    if 'use_weights' in approx_options.keys() and approx_options['use_weights']:
      weights = np.ones(outputs.shape)
      weights[outputs>=1000] = 2
      weights[outputs>=2000] = 4
      del approx_options['use_weights']

    # print outputs
    # print weights

    if self.approx_technique.lower() in ['rsm', 'moa', 'hda', 'gp']:
      if approx_options is None:
        raise Exception('Please specify method options!')

      if weights is None:
        model = gtapprox.Builder().build(inputs, outputs, options=approx_options)
      else:
        model = gtapprox.Builder().build(inputs, outputs, weights=weights, options=approx_options)


      model.save(self.models_save_path+model_name+self.ext)

    elif self.approx_technique.lower() == 'xgboost':

      if not self.approx_options is None:
        model = Xgboost(**approx_options)
      else:
        model = Xgboost()

      if weights is None:
        model.fit(inputs, outputs)
      else:
        model.fit(inputs, outputs, weights=weights)

      model.save(self.models_save_path+model_name+self.ext)

    elif self.approx_technique.lower() == 'quadxgboost':


      if weights is None:
        model0 = gtapprox.Builder().build(inputs, outputs, options={
            'gtapprox/technique': 'rsm',
            'gtapprox/rsmtype': 'quadratic',
            "gtapprox/rsmfeatureSelection": "RidgeLS"
        })
      else:
        model0 = gtapprox.Builder().build(inputs, outputs, weights=weights, options={
            'gtapprox/technique': 'rsm',
            'gtapprox/rsmtype': 'quadratic',
            "gtapprox/rsmfeatureSelection": "RidgeLS"
        })

      model0.save(self.models_save_path+model_name+'_0_'+self.ext)

      predicted_outputs = model0.calc(inputs)

      if not self.approx_options is None:
        model = Xgboost(**approx_options)
      else:
        model = Xgboost()

      if weights is None:
        model.fit(inputs, outputs-predicted_outputs)
      else:
        model.fit(inputs, outputs-predicted_outputs, weights=weights)

      model.save(self.models_save_path+model_name+'_1_'+self.ext)

    elif self.approx_technique.lower() == 'mars':

      model = Earth()
      model.fit(inputs, outputs)

      # model.save(self.models_save_path+model_name)
      with open(self.models_save_path+model_name+self.ext, 'wb') as f:
        pickle.dump(model, f)

    elif self.approx_technique.lower() == 'skgboost':

      model = ensemble.GradientBoostingRegressor(**approx_options)

      if weights is None:
        model.fit(inputs, outputs[:, 0])
      else:
        model.fit(inputs, outputs[:, 0], weights=weights)

      with open(self.models_save_path+model_name+self.ext, 'wb') as file_object:
        pickle.dump(model, file_object)

    elif self.approx_technique.lower() == 'adaboost':

      if approx_options['base_model'] == 'skgb':
        base_model = ensemble.GradientBoostingRegressor(**approx_options['base_params'])
      elif approx_options['base_model'] == 'xgb':
        base_model = Xgboost(**approx_options['base_params'])
      elif approx_options['base_model'] == 'tree':
        base_model = tree.DecisionTreeRegressor(**approx_options['base_params'])

      del approx_options['base_model']
      del approx_options['base_params']

      self.approx_options['base_estimator'] = base_model

      model = ensemble.AdaBoostRegressor(**approx_options)
      model.fit(inputs, outputs[:, 0])

      with open(self.models_save_path+model_name+self.ext, 'wb') as file_object:
        pickle.dump(model, file_object)

    elif self.approx_technique.lower() == 'zeros':
      pass

    elif self.approx_technique.lower() == 'means':

      model = np.mean(outputs[:, 0])

      with open(self.models_save_path+model_name+self.ext, 'wb') as file_object:
        pickle.dump(model, file_object)

    else:
      raise Exception('Wrong approx type specified!')

    return model

  # @autojit
  def compute_predictions(self, model_name, inputs):
    '''
    computes prediction for given sample and specified model
    '''

    if self.approx_technique.lower() in ['rsm', 'moa', 'hda', 'gp']:
      model = gtapprox.Model(self.models_save_path+model_name+self.ext)
      prediction = model.calc(inputs)

    elif self.approx_technique.lower() == 'xgboost':
      model = Xgboost()
      model.load(self.models_save_path+model_name+self.ext)
      prediction = model.predict(inputs)[:, np.newaxis]

    elif self.approx_technique.lower() == 'quadxgboost':
      model = Xgboost()
      model.load(self.models_save_path+model_name+'_1_'+self.ext)
      prediction = model.predict(inputs)[:, np.newaxis]

      model0 = gtapprox.Model(self.models_save_path+model_name+'_0_'+self.ext)
      prediction = prediction + model0.calc(inputs)

    elif self.approx_technique.lower() == 'mars':

      with open(self.models_save_path+model_name+self.ext, 'rb') as file_object:
        model = pickle.load(file_object)

      prediction = model.predict(inputs)[:, np.newaxis]

    elif self.approx_technique.lower() == 'skgboost':

      with open(self.models_save_path+model_name+self.ext, 'rb') as file_object:
        model = pickle.load(file_object)

      prediction = model.predict(inputs)[:, np.newaxis]

    elif self.approx_technique.lower() == 'adaboost':

      with open(self.models_save_path+model_name+self.ext, 'rb') as file_object:
        model = pickle.load(file_object)

      prediction = model.predict(inputs)[:, np.newaxis]

    elif self.approx_technique.lower() == 'zeros':
      prediction = np.zeros((inputs.shape[0], 1))

    elif self.approx_technique.lower() == 'means':

      with open(self.models_save_path+model_name+self.ext, 'rb') as file_object:
        model = pickle.load(file_object)

      prediction = model*np.ones((inputs.shape[0], 1))

    else:
      raise Exception('Wrong approx type specified!')

    return prediction


def compute_errors(true, prediction, error_filters=None):
  '''
  Computes prediction errors
  '''

  if len(true.shape) == 1:
    true = true[:, np.newaxis]
  if len(prediction.shape) == 1:
    prediction = prediction[:, np.newaxis]

  all_errors = {}

  if error_filters is None:
    error_filters = {'no_filter':[None, None]}


  for filter_name, error_filter in error_filters.items():

    filtered = np.ones((true.shape[0], 1), dtype=bool)

    if not error_filter[0] is None:
      filtered *= true >= error_filter[0]

    if not error_filter[1] is None:
      filtered *= true <= error_filter[1]

    filtered = filtered[:, 0]

    errors = {}

    count = np.sum(filtered)

    errors['count'] = count

    delt = np.abs((true[filtered, 0]-prediction[filtered, 0])/true[filtered, 0])

    sorted_errors_idx = np.argsort(delt)

    errors['mse'] = np.sqrt(np.mean((true[filtered, :]-prediction[filtered, :])**2)) if count > 0 else -1
    errors['map'] = np.mean(np.abs(true[filtered, :]-prediction[filtered, :])) if count > 0 else -1
    errors['maxre'] = np.max(np.abs((true[filtered, :]-prediction[filtered, :])/true[filtered, :])) if count > 0 else -1
    errors['maxre-1'] = delt[sorted_errors_idx[-2]] if count > 1 else -1
    errors['maxre-5'] = delt[sorted_errors_idx[-5]] if count > 5 else -1
    errors['maxre-10'] = delt[sorted_errors_idx[-10]] if count > 10 else -1
    errors['max'] = np.max(np.abs((true[filtered, :]-prediction[filtered, :]))) if count > 0 else -1
    errors['mre'] = np.mean(np.abs((true[filtered, :]-prediction[filtered, :])/true[filtered, :])) if count > 0 else -1

    try:
      errors['q99re'] = np.percentile(np.abs((true[filtered, :]-prediction[filtered, :])/true[filtered, :]), 99)
      errors['q999re'] = np.percentile(np.abs((true[filtered, :]-prediction[filtered, :])/true[filtered, :]), 99.9)
      errors['q9999re'] = np.percentile(np.abs((true[filtered, :]-prediction[filtered, :])/true[filtered, :]), 99.99)
      errors['q95re'] = np.percentile(np.abs((true[filtered, :]-prediction[filtered, :])/true[filtered, :]), 95)
      errors['q80re'] = np.percentile(np.abs((true[filtered, :]-prediction[filtered, :])/true[filtered, :]), 80)
      errors['q99'] = np.percentile(np.abs((true[filtered, :]-prediction[filtered, :])), 99)
      errors['q95'] = np.percentile(np.abs((true[filtered, :]-prediction[filtered, :])), 95)
      errors['q80'] = np.percentile(np.abs((true[filtered, :]-prediction[filtered, :])), 80)
    except:
      errors['q99re'] = -1
      errors['q999re'] = -1
      errors['q9999re'] = -1
      errors['q95re'] = -1
      errors['q80re'] = -1
      errors['q99'] = -1
      errors['q95'] = -1
      errors['q80'] = -1

    all_errors[filter_name] = errors

  return all_errors

def save_errors_tables(save_path, all_outputs, all_predictions, error_filters):
  '''
  Saves obtained errors to Excel file
  '''

  domains = all_outputs.keys()
  domains.sort()

  flights = all_outputs[domains[0]].keys()
  flights.sort()

  filters = error_filters.keys()
  filters.sort()

  writer = pd.ExcelWriter(save_path)

  dummy_error = compute_errors(all_outputs[domains[0]][flights[0]], all_predictions[domains[0]][flights[0]], error_filters=error_filters)

  error_types = dummy_error[filters[0]].keys()
  error_types.sort()


  # per filter errors
  for current_filter in filters:
    start_column_1 = 0
    start_row_1 = 0

    start_column_2 = 0
    start_row_2 = 0

    for error_type in error_types:

      ## Detailed_table ##
      table = []
      for flight in flights:
        row = [int(flight)]
        for domain in domains:
          error = compute_errors(all_outputs[domain][flight], all_predictions[domain][flight], error_filters=error_filters)
          row.append(error[current_filter][error_type])

        table.append(row)

      df = pd.DataFrame(table, columns=[error_type]+['D'+str(elem) for elem in domains])

      df.to_excel(writer, 'detailed '+current_filter, float_format='%.3f', startrow=start_row_1, startcol=start_column_1, index=False)# , float_format='%.3f'

      start_column_1 += len(domains) + 2

      if start_column_1 > 16:
        start_row_1 += len(flights) + 3
        start_column_1 = 0

    ## Total statistics table ##

    table = []
    joint_outputs = []
    joint_predictions = []
    row = []
    for domain in domains:
      for flight in flights:
        joint_outputs += list(all_outputs[domain][flight][:, 0])
        joint_predictions += list(all_predictions[domain][flight][:, 0])
    error = compute_errors(np.array(joint_outputs), np.array(joint_predictions), error_filters=error_filters)
    for error_type in error_types:
      row.append(error[current_filter][error_type])
    table.append(row)

    df = pd.DataFrame(table, columns=error_types)

    print df

    df.to_excel(writer, 'total '+current_filter, float_format='%.3f', startrow=start_row_2, startcol=start_column_2, index=False)# , float_format='%.3f'

    start_row_2 += 4

    table = []
    for domain in domains:
      row = [int(domain)]
      joint_outputs = []
      joint_predictions = []
      for flight in flights:
        joint_outputs += list(all_outputs[domain][flight][:, 0])
        joint_predictions += list(all_predictions[domain][flight][:, 0])
        # error = compute_errors(all_outputs[domain][flight][current_filter], all_predictions[domain][flight][current_filter])
        # row.append(error[error_type])
      error = compute_errors(np.array(joint_outputs), np.array(joint_predictions), error_filters=error_filters)
      for error_type in error_types:
        row.append(error[current_filter][error_type])
      table.append(row)

    df = pd.DataFrame(table, columns=['domain']+error_types)

    print df

    df.to_excel(writer, 'total '+current_filter, float_format='%.3f', startrow=start_row_2, startcol=start_column_2, index=False)# , float_format='%.3f'


    # start_column_2 += len(domains) + 2

    start_row_2 += len(domains) + 5

    table = []
    for flight in flights:
      row = [int(flight)]
      joint_outputs = []
      joint_predictions = []
      for domain in domains:
        joint_outputs += list(all_outputs[domain][flight][:, 0])
        joint_predictions += list(all_predictions[domain][flight][:, 0])
        # error = compute_errors(all_outputs[domain][flight][current_filter], all_predictions[domain][flight][current_filter])
        # row.append(error[error_type])
      error = compute_errors(np.array(joint_outputs), np.array(joint_predictions), error_filters=error_filters)
      for error_type in error_types:
        row.append(error[current_filter][error_type])
      table.append(row)

    df = pd.DataFrame(table, columns=['flight_name']+error_types)

    print df

    df.to_excel(writer, 'total '+current_filter, float_format='%.3f', startrow=start_row_2, startcol=start_column_2, index=False)# , float_format='%.3f'


  writer.save()