# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 14:00:37 2015
Modified on Fri Jul 17 11:01:00 2015

@author: alexandrebarachant
@modified by gavin.chan

Beat the benchmark with CSP and Logisitic regression.

General Idea :

The goal of this challenge is to detect events related to hand movements. Hand 
movements are caracterized by change in signal power in the mu (~10Hz) and beta
(~20Hz) frequency band over the sensorimotor cortex. CSP spatial filters are
trained to enhance signal comming from this brain area, instantaneous power is
extracted and smoothed, and then feeded into a logisitic regression.

Preprocessing :

Signal are bandpass-filtered between 7 and 30 Hz to catch most of the signal of
interest. 4 CSP spatial filter are then applied to the signal, resutlting to
4 new time series.  In order to train CSP spatial filters, EEG are epoched 
using a window of 1.5 second before and after the event 'Replace'. CSP training
needs two classes. the epochs before Replace event are assumed to contain 
patterns corresponding to hand movement, and epochs after are assumed to 
contain resting state.

Feature extraction :

Preprocessing is applied, spatialy filtered signal are the rectified and 
convolved with a 0.5 second rectangular window for smoothing. Then a logarithm
is applied. the resutl is a vector of dimention 4 for each time sample.

Classification :

For each of the 6 event type, a logistic regression is trained. For training 
only, features are downsampled in oder to speed up the process. Prediction are
the probailities of the logistic regression.

"""

print(__doc__)

import numpy as np
import pandas as pd
from train_para import TrainPara
from datetime import datetime
from mne.io import RawArray
from mne.channels import read_montage
from mne.epochs import concatenate_epochs
from mne import create_info, find_events, Epochs, concatenate_raws, pick_types
from mne.decoding import CSP

from sklearn.linear_model import LogisticRegression
from sklearn.lda import LDA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, auc

from scipy.signal import butter, lfilter, convolve, boxcar
from joblib import Parallel, delayed


def create_mne_raw_object(fname, read_events=True):
    """ Create a mne raw instance from csv file
    """

    # Read EEG file
    print('Reading file ' + fname + ' now...')
    data = pd.read_csv(fname)

    # get chanel names
    ch_names = list(data.columns[1:])

    # read EEG standard montage from mne
    montage = read_montage('standard_1005', ch_names)

    ch_type = ['eeg'] * len(ch_names)
    data = 1e-6 * np.array(data[ch_names]).T

    if read_events:
        # events file
        ev_fname = fname.replace('_data', '_events')
        # read event file
        events = pd.read_csv(ev_fname)
        events_names = events.columns[1:]
        events_data = np.array(events[events_names]).T

        # define channel type, the first is EEG, the last 6 are stimulations
        ch_type.extend(['stim'] * 6)
        ch_names.extend(events_names)
        # concatenate event file and data
        data = np.concatenate((data, events_data))

    # create and populate MNE info structure
    info = create_info(ch_names, sfreq=500.0, ch_types=ch_type, montage=montage)
    info['filename'] = fname

    # create raw object 
    raw = RawArray(data, info, verbose=False)

    return raw


def butterworth_filter(t,k,l):
    if t==0:
        freq=[k, l]
        b,a = butter(5,np.array(freq)/250.0,btype='bandpass')
    elif t==1:
        b,a = butter(3,k/250.0,btype='lowpass')
    elif t==2:
        b,a = butter(3,l/250.0,btype='highpass')
    return (b, a)


def filter_raw_data(raw, picks, training_parameters):
    """ Apply raw data filtering.

    :param raw: Raw data
    """
    # Filter data for alpha frequency and beta band
    # Note that MNE implement a zero phase (filtfilt) filtering not compatible
    # with the rule of future data.
    # Here we use left filter compatible with this constraint.
    # The function parallelized for speeding up the script

    # design a butterworth bandpass filter
    # freqs = [7, 30]
    # b, a = butter(5, np.array(freqs) / 250.0, btype='bandpass')
    # for i in range(10):
    b, a = butterworth_filter(training_parameters.butterworth_t,
                              training_parameters.butterworth_k,
                              training_parameters.butterworth_l)
    raw._data[picks] = np.array([(lfilter)(b, a, raw._data[i]) for i in picks])


def generate_csp_features(csp, raw, picks, nwin, nfilters):
    """ Generate csp features and then smooth the features by convolution with a rectangle window.

    :param csp: The trained csp filter
    :param raw: The raw data
    :return: The filtered features
    """

    # apply csp filters and rectify signal
    feat = np.dot(csp.filters_[0:nfilters], raw._data[picks]) ** 2

    # smoothing by convolution with a rectangle window
    feattr = np.array([(convolve)(feat[i], boxcar(nwin), 'full') for i in range(nfilters)])
    feattr = np.log(feattr[:, 0:feat.shape[1]])

    return feattr


def csp_training(raw, picks, nfilters):
    """ Implement CSP training

    :param raw: Raw data
    :return: The csp filter
    """

    epochs_tot = []
    y = []

    # get event position corresponding to Replace
    events = find_events(raw, stim_channel='HandStart', verbose=False)
    # epochs signal for 1.5 second before the movement
    epochs = Epochs(raw, events, {'during': 1}, 0, 2, proj=False,
                    picks=picks, baseline=None, preload=True,
                    add_eeg_ref=False, verbose=False)

    epochs_tot.append(epochs)
    y.extend([1] * len(epochs))

    # epochs signal for 1.5 second after the movement, this correspond to the
    # rest period.
    epochs_rest = Epochs(raw, events, {'before': 1}, -2, 0, proj=False,
                         picks=picks, baseline=None, preload=True,
                         add_eeg_ref=False, verbose=False)

    # Workaround to be able to concatenate epochs with MNE
    epochs_rest.times = epochs.times

    y.extend([-1] * len(epochs_rest))
    epochs_tot.append(epochs_rest)

    # Concatenate all epochs
    epochs = concatenate_epochs(epochs_tot)

    # get data
    X = epochs.get_data()
    y = np.array(y)

    # train CSP
    csp = CSP(n_components=nfilters, reg='lws')
    csp.fit(X, y)

    return csp

def model_training( subject , training_parameters):
    """ Training the model.

    :param subject: The subject index
    """

    # CSP parameters
    # Number of spatial filter to use
    nfilters = training_parameters.csp_nfilter

    # training subsample
    subsample = training_parameters.subsample

    # window
    nwin = training_parameters.nwin

    # batching const
    batch_start = 1
    batch_end = 6
    # unbatch_start = batch_end + 1
    unbatch_start = 7
    unbatch_end = 8

    ids_tot = []

    ################ READ DATA ################################################
    # fnames =  glob('../data/train/subj%d_series*_data.csv' % (subject))
    fnames = ['../data/train/subj%d_series%d_data.csv' % (subject, ix)
              for ix in range(batch_start, batch_end + 1)]

    # read and concatenate all the files
    raw = concatenate_raws([create_mne_raw_object(fname) for fname in fnames])

    # pick eeg signal
    picks = pick_types(raw.info, eeg=True)

    # filter
    filter_raw_data(raw, picks, training_parameters)

    ################ CSP Filters training #####################################
    csp = csp_training(raw, picks, nfilters)

    ################ Create Training Features #################################
    # apply csp and filtering
    feattr = generate_csp_features(csp, raw, picks, nwin, nfilters)

    # training labels
    # they are stored in the 6 last channels of the MNE raw object
    labels_batch = raw._data[32:]

    ################ Create test Features #####################################
    # read test data
    # fnames =  glob('../data/train/subj%d_series*_data.csv' % (subject))
    fnames = ['../data/train/subj%d_series%d_data.csv' % (subject, ix)
              for ix in range(unbatch_start, unbatch_end + 1)]
    raw = concatenate_raws([create_mne_raw_object(fname, read_events=True) for fname in fnames])

    # filter
    filter_raw_data(raw, picks, training_parameters)

    labels_unbatch = raw._data[32:]

    # apply preprocessing on test data
    featte = generate_csp_features(csp, raw, picks, nwin, nfilters)

    # read ids
    ids = np.concatenate([np.array(pd.read_csv(fname)['id']) for fname in fnames])
    ids_tot.append(ids)

    ################ Train classifiers ########################################
    lr = []
    if training_parameters.model == 1:
        lr = LogisticRegression()
    elif training_parameters.model == 2:
        lr = LDA()
    pred = np.empty((len(ids), 6))
    pred_roc_area = []
    for i in range(6):
        lr.fit(feattr[:, ::subsample].T, labels_batch[i, ::subsample])
        pred[:, i] = lr.predict_proba(featte.T)[:, 1]
        # calculate train data roc and its curve
        pred_roc_area.append(roc_auc_score(labels_unbatch[i, :].T, pred[:, i]))
        # fpr, tpr, thresholds = roc_curve(labels[i,:].T, pred_train[:,i])
        # roc_auc = auc(x=fpr, y=tpr, reorder=True)
        # plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
        # print('Train subject %d, class %s, event %d roc area = %.5f' % (subject, cols[i], i, pred_roc_area[i]))

    print('Roc area mean of train subject %d = %.5f' % (subject, np.mean(pred_roc_area)))
    # plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
    # plt.show()

subjects = range(1, 13)

def data_preprocess_test(X):
    scaler = StandardScaler()
    X_prep_normal = scaler.fit_transform(X)
    X_prep_low = np.zeros((np.shape(X_prep_normal)[0],10))



if __name__ == '__main__':

    training_parameters = TrainPara()
    training_parameters.nwin = 250
    training_parameters.csp_nfilter = 4
    training_parameters.butterworth_t = 0
    training_parameters.butterworth_k = 7
    training_parameters.butterworth_l = 30
    training_parameters.model = 1
    training_parameters.subsample = 10

    # time the execution
    run_start = datetime.now()

    Parallel(n_jobs=2)(delayed(model_training)(subject, training_parameters) for subject in subjects)
    # for subject in subjects:
    #     model_training(subject, training_parameters)

    print("Total run time : %s" % str(datetime.now() - run_start))

    ## create pandas object for sbmission
    #submission = pd.DataFrame(index=np.concatenate(ids_tot),
    #                          columns=cols,
    #                          data=np.concatenate(pred_tot))

    ## write file
    #submission.to_csv(submission_file, index_label='id', float_format='%.3f')
