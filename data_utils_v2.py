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

import data_handler as dh

import copy

import hashlib

import dill as pickle


def in_hull(points, hull):
  """
  Test if points in `p` are in `hull`

  `p` should be a `NxK` coordinates of `N` points in `K` dimensions
  `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
  coordinates of `M` points in `K`dimensions for which Delaunay triangulation
  will be computed
  """
  # if not isinstance(hull,Delaunay):

  del points['flight_name']
  del points['output']
  del points['TEMPS']

  del hull['flight_name']
  del hull['output']
  del hull['TEMPS']

  hull = Delaunay(hull.as_matrix())

  return hull.find_simplex(points.as_matrix())>=0

class prettyfloat(float):
    def __repr__(self):
        return "%0.2f" % self


def draw_ranges_for_parameters(data, title='', save_path='./pictures/'):
  parameters = data.columns.values.tolist()

  # remove flight name parameter
  for idx, parameter in enumerate(parameters):
    if parameter == 'flight_name':
      del parameters[idx]

  flight_names = np.unique(data['flight_name'])

  print len(flight_names)

  for parameter in parameters:
    plt.figure()

    axis = plt.gca()

    # ax.set_xticks(numpy.arange(0,1,0.1))
    axis.set_yticks(flight_names)
    axis.tick_params(labelright=True)
    axis.set_ylim([94., 130.])
    plt.grid()

    plt.title(title)
    plt.xlabel(parameter)
    plt.ylabel('flight name')

    colors = iter(cm.rainbow(np.linspace(0, 1,len(flight_names))))

    for flight in flight_names:
      temp = data[data.flight_name == flight][parameter]

      plt.plot([np.min(temp), np.max(temp)], [flight, flight], c=next(colors), linewidth=2.0)
    plt.savefig(save_path+title+'_'+parameter+'.jpg')
    plt.close()

def draw_ranges_for_flights(data, split_parameter, title='', save_path='./pictures/'):

  if not save_path[-1] == '/':
    save_path += '/'

  parameters = data.columns.values.tolist()

  # remove flight name parameter
  for idx, parameter in enumerate(parameters):
    if parameter == split_parameter:
      del parameters[idx]

  print 'list of parameters:', parameters

  flight_names = np.unique(data[split_parameter])

  print 'split parameter', split_parameter, 'has', len(flight_names), 'values'

  all_size = len(data)

  for flight in flight_names:
    plt.figure()
    axis = plt.gca()

    # ax.set_xticks(numpy.arange(0,1,0.1))
    plt.yticks(range(len(parameters)), parameters)
    axis.tick_params(labelright=True)
    axis.set_ylim([-1., len(parameters)])
    plt.grid()

    plt.title(title)

    plt.xlabel(split_parameter+' '+str(int(flight)) + ' ('+str(len(data[data[split_parameter] == flight]))+ ' of '+str(all_size)+' points)')
    plt.ylabel('parameters')

    colors = iter(cm.rainbow(np.linspace(0, 1,len(flight_names))))

    for idx, parameter in enumerate(parameters):

      min_value = data[parameter].min()
      max_value = data[parameter].max()
      range_value = max_value - min_value

      temp = data[data[split_parameter] == flight][parameter]

      plt.plot([(np.min(temp)-min_value)/range_value, (np.max(temp)-min_value)/range_value], [idx+0.2, idx+0.2], c='r', linewidth=3.0)

      for flight2 in flight_names:
        temp2 = data[data[split_parameter] == flight2][parameter]
        if flight == flight2:
          continue
        plt.plot([(np.min(temp2)-min_value)/range_value, (np.max(temp2)-min_value)/range_value], [idx, idx], c='b', linewidth=3.0)

    plt.savefig(save_path+title+'_'+str(flight)+'.jpg')
    plt.close()

def get_box(data, split_parameter):
  '''
  Returns dict that contain ranges for all input parameters except split parameter
  '''

  box = {}

  # parameters = data.columns.values.tolist()

  # remove flight name parameter
  for idx in xrange(data.shape[1]):
    box[idx] = [data[:, idx].min(), data[:, idx].max()]

  return box

def check_if_in_box(data, box):

  decisions = np.ones(len(data), dtype=bool)

  for parameter in box.keys():
    decisions *= data[:, parameter] >= box[parameter][0]
    decisions *= data[:, parameter] <= box[parameter][1]

  return decisions

def generate_regressors(data, family='quadratic'):
  #TODO
  result = np.copy(data)
  if family.lower() == 'quadratic':
    result = np.hstack()

  return result


def check_classify(train, test, model_type='logistic_regression', weighted=True, title='', save_path=None):

  train_inputs, train_outputs = get_matrices(train, input_columns_to_remove=['flight_name', 'TEMPS'], output_columns=['output'])
  test_inputs, test_outputs = get_matrices(test, input_columns_to_remove=['flight_name', 'TEMPS'], output_columns=['output'])

  len_train = len(train_inputs)
  len_test = len(test_inputs)

  joint_inputs = np.vstack((train_inputs, test_inputs))
  classes = np.ones(len(joint_inputs))
  classes[:len_train] = 0

  if model_type.lower() == 'logistic_regression':

    if weighted:
      model = lm.LogisticRegression(class_weight='auto')
    else:
      model = lm.LogisticRegression()

    model.fit(joint_inputs, classes)
  elif model_type == 'svc':
    if weighted:
      model = svm.LinearSVC(class_weight='auto', C=100, fit_intercept=True)
    else:
      model = svm.LinearSVC(C=100, fit_intercept=True)
    model.fit(joint_inputs, classes)

    train_prediction = model.predict(train_inputs)

    # part_of_train = np.sum((1-train_prediction))/len_train
    test_prediction = model.predict(test_inputs)

    # # indexes = np.array(range(len(inputs)))
    # fig, (ax1, ax2) = plt.subplots(2,1)

    # ax1.hist(train_prediction)
    # ax1.set_title('train points')

    # ax2.hist(test_prediction)
    # ax2.set_title('test points')

    # plt.suptitle(title)
    # # plt.plot(indexes[:len_train], train_prediction, c='b')
    # # plt.plot(indexes[len_train:], test_prediction, c='r')
    # if not save_path is None:
    #   plt.savefig(save_path+'.png')
    # else:
    #   plt.show()

  probas_ = model.predict_proba(joint_inputs)

  # Compute ROC curve and area the curve
  fpr, tpr, thresholds = roc_curve(classes, probas_[:, 1])
  roc_auc = auc(fpr, tpr)
  print "Area under the ROC curve : %f" % roc_auc

  # Plot ROC curve
  plt.clf()
  plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
  plt.plot([0, 1], [0, 1], 'k--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.0])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  # plt.title('Receiver operating characteristic example')
  plt.title(title)
  plt.legend(loc="lower right")
  if not save_path is None:
    plt.savefig(save_path+'.png')
  else:
    plt.show()

  plt.close()

def calc_distances(train_inputs, test_inputs, batch_size=1000, metric='seuclidean', normalization=None):

  train_inputs = np.copy(train_inputs)
  test_inputs = np.copy(test_inputs)

  len_train = len(train_inputs)
  len_test = len(test_inputs)

  steps = np.floor(len(train_inputs)/batch_size)

  if not normalization is None:
    if 'mapstd' in normalization:

      for current_input in xrange(train_inputs.shape[1]):
        mean = np.mean(train_inputs[:, current_input])
        std = np.std(train_inputs[:, current_input])

        train_inputs[:, current_input] = (train_inputs[:, current_input] - mean)/std
        test_inputs[:, current_input] = (test_inputs[:, current_input] - mean)/std


  current_point = 0

  distances = np.zeros((len_test, 1))

  while current_point < len_test:
    current_idx = range(current_point, np.min((current_point+batch_size, len_test)))
    distances[current_idx, 0] = np.min(scd.cdist(train_inputs, test_inputs[current_idx, :], metric=metric), axis=0)
    current_point += batch_size

   # statement(s)

  # for step in steps:

  #   distances = np.min(scd.cdist(train_inputs, test_inputs, metric='seuclidean'), axis=0)

  # print np.min(distances)

  # print aaa

  return distances

def draw_errors_plot_v2(outputs, prediction, save_path, title=''):

  plt.figure()
  mm0 = [np.min((np.min(prediction),np.min(outputs))), np.max((np.max(prediction),np.max(outputs)))]

  mm = np.array([0., mm0[1]+700])

  plt.plot(mm, mm, c='black')
  plt.plot(mm, 1.1*mm, c='g')
  plt.plot(mm, 1.2*mm, c='r')
  plt.plot(mm, 1.5*mm, c='c')
  plt.plot(mm, 2.*mm, c='m')
  plt.plot(mm, mm/1.1, c='g')
  plt.plot(mm, mm/1.2, c='r')
  plt.plot(mm, mm/1.5, c='c')
  plt.plot(mm, mm/2., c='m')
  plt.title(title)

  plt.xlim(mm)
  plt.ylim(mm)

  # print 'started_scatter...'
  plt.scatter(prediction, outputs)
  # print 'finished'

  plt.legend(['0%', '10%', '20%', '50%', '100%'])

  plt.xlabel('prediction')
  plt.ylabel('true')

  paths = save_path.split('/')

  # plt.savefig(save_path+'.png')
  plt.savefig('/'.join(paths[:-1])+'/acc_'+paths[-1]+'.png')

  plt.close()
  # print 'started_hist...'
  plt.figure()
  plt.hist((outputs[:, 0]-prediction[:, 0])/outputs[:, 0], bins=100)
  # print 'finished'

  plt.savefig('/'.join(paths[:-1])+'/hist_'+paths[-1]+'.png')

  plt.close()


def draw_error_scatters(data, prediction, save_path, title=''):

  data = data.copy()

  # print data.columns.values

  if 'Vx' in data.columns.values:
    data['speed'] = np.sqrt(data['Vx']**2 + data['Vy']**2)

  inputs, output = dh.get_matrices(data, input_columns_to_remove=['flight_name', 'domain'], output_columns=['output'])

  parameter_names = data.columns.values

  # print parameter_names
  for idx, parameter in enumerate(parameter_names):
    parameter_names = [parameter for parameter in parameter_names if not parameter in ['flight_name', 'output', 'domain']]

  # parameter_names += ['output']

  axis_font = {'fontname':'Arial', 'size':'30'}

  fig, axes  = plt.subplots(1, inputs.shape[1]+1, figsize=(5*inputs.shape[1], 6))

  for current_input in xrange(inputs.shape[1]):
    axes[current_input].plot(inputs[:, current_input], np.log10(np.abs((output-prediction)/output)), linewidth=0.0, marker='o')
    axes[current_input].plot(inputs[:, current_input], np.log10(0.2)*np.ones(len(inputs)), color='r', linewidth=3.)
    axes[current_input].plot(inputs[:, current_input], np.log10(0.1)*np.ones(len(inputs)), color='g', linewidth=3.)

    axes[current_input].set_xlabel(parameter_names[current_input], **axis_font)
    # axes[current_input].set_ylabel('relative error', **axis_font)

  axes[inputs.shape[1]].plot(output, np.log10(np.abs((output-prediction)/output)), linewidth=0.0, marker='o')
  axes[inputs.shape[1]].plot(output, np.log10(0.2)*np.ones(len(inputs)), color='r', linewidth=3.)
  axes[inputs.shape[1]].plot(output, np.log10(0.1)*np.ones(len(inputs)), color='g', linewidth=3.)

  axes[inputs.shape[1]].set_xlabel('output', **axis_font)
  # axes[current_input].set_ylabel('relative error', **axis_font)

  plt.suptitle(title)
  plt.savefig(save_path+'.png')

  plt.close()

def smooth(x,window_len=100,window='flat'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman', 'min', 'max']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"

    if not window in ['max', 'min']:


      s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]

      # print s.shape
      #print(len(s))
      if window == 'flat': #moving average
          w=np.ones(window_len,'d')
      else:
          w=eval('np.'+window+'(window_len)')

      y=np.convolve(w/w.sum(),s,mode='valid')

      return y[(window_len/2-1):-(window_len/2)]

    else:

      s=np.r_[x[(window_len-1)/2:0:-1],x,x[-1:-(window_len-1)/2:-1]]

      if  window == 'max':
        y = np.array([np.max(s[idx:idx+window_len]) for idx, s_loc in enumerate(x)])
      elif window == 'min':
        y = np.array([np.min(s[idx:idx+window_len]) for idx, s_loc in enumerate(x)])
      return y

def dict_pretty_string(options, indent='  ', new_line=True):
  ''' Prints options as alphabetically sorted list
  '''
  sorted_keys = options.keys()
  sorted_keys.sort()

  string = ''

  if new_line:
    end_symbol = '\n'
    separator = ' : '
  else:
    end_symbol = ''
    indent = ''
    separator = '='

  for key in sorted_keys:
    if type(options[key]) == dict:
      # dict_pretty_print(options[key], print_function=print_function, indent=indent+'  ')
      string += indent+str(key)+separator+end_symbol
      string += dict_pretty_string(options[key], indent=indent+'  ')
    else:
      string += indent+str(key)+separator+str(options[key])+end_symbol
      if indent == '':
        indent = ','
      # print_function('  '+key+' : '+str(options[key]))
  # print_function('')
  string += end_symbol

  return string

def common_print(string):
  print string

def calc_hash(argument):
  argument_string = pickle.dumps(argument)
  signature = argument_string
  hasher = hashlib.sha256()
  hasher.update(signature)
  hash_string = hasher.hexdigest()
  return hash_string

def remove_sublist_from_origin(origin, sublist):
  for element in sublist:
    if element in origin: 
      origin.remove(element)

  return origin
