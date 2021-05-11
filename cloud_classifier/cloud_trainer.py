#import tools.io as io
import tools.training as dh
import tools.plotting as pl

import xarray as xr
import numpy as np

from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2

from joblib import dump, load
import time

import importlib
importlib.reload(dh)
importlib.reload(pl)






class cloud_trainer:
    """
    Trainable Classifier for cloud cl_type prediction from satelite data.



    Methods
    -------
    add_training_sets(filename_data, filename_labels)
        Reads set of satelite data and according labels and adds it into the classifier
    
    create_trainig_set(n)
        Creates training vectors from all added trainig sets

    add_h5mask(filename, selected_mask = None)
        Reads mask-data from h5 file and sets mask for the classifier if specified

    """


    def __init__(self):
        self.training_vectors = None
        self.training_labels = None
        self.masked_indices = None

        self.pred_vectors = None
        self.pred_labels = None
        self.pred_indices = None
        self.pred_filename = None

        self.cl = None
        self.feat_select = None

        ### paramaeters
        self.cl_type = "Tree"
        self.max_depth = 20
        self.ccp_alpha = 0
        self.feature_preselection = False
        self.n_estimators = 75


    def set_training_paremeters(self, cl_type = "Tree", feature_preselection = False):

        self.cl_type = cl_type
        self.feature_preselection = feature_preselection


    def fit_feature_selection(self, k = 20):
        if(self.training_vectors is None or self.training_labels is None):
            print("No training vectors ceated")
            return
        self.feat_select = SelectKBest(k=k).fit(self.training_vectors, self.training_labels)



    def apply_feature_selection(self, vectors):
        if(self.feat_select is None):
            print("No feature selection fitted")
            return
        return self.feat_select.transform(vectors)



    def train_tree_classifier(self, training_vectors, training_labels):
        """
        Trains the classifier using previously created training_vectors
        
        Parameters
        ----------
        m_depth : int
            Maximal depth of the decision tree
        """

        if(self.cl_type == "Tree"):
            self.cl = tree.DecisionTreeClassifier(max_depth = self.max_depth, ccp_alpha = self.ccp_alpha)
        elif(self.cl_type == "Forest"): 
            self.cl = RandomForestClassifier(n_estimators = self.n_estimators, max_depth = self.max_depth, 
                                                ccp_alpha = self.ccp_alpha)

        if(training_vectors is None or training_labels is None):
            print("No training data!")
            return

        if(self.feature_preselection and not (self.feat_select is None)):
            training_vectors = self.apply_feature_selection(training_vectors)
        self.cl.fit(training_vectors, training_labels)



    def predict_labels(self, vectors):
        """
        Predicts the labels if a corresponding set of input vectors has been created.
        """
        if(self.cl is None):
            print("No classifer trained or loaded")
            return

        if(self.feature_preselection and not (self.feat_select is None)):
            vectors = self.apply_feature_selection(vectors)

        self.pred_labels =  self.cl.predict(vectors)
        return self.pred_labels





    def plot_labels(self):
        """
        Plots predicted labels
        """
        if (self.pred_labels is None or self.pred_filename is None
                or self.pred_indices is None):
            print("Unsufficant data for plotting labels")
            return
        data = dh.imbed_data(self.pred_labels, self.pred_indices, self.pred_filename)
        pl.plot_data(data)




    def evaluate_parameters(self, vectors, labels, verbose = True):
        """
        Evaluates the given parameters over a set of training vectors

        Training vectors are split into test and trainig set
        """

        train_v, test_v, train_l, test_l = train_test_split(vectors, labels, random_state=0)

        self.train_tree_classifier(train_v, train_l)

        pred_l = self.predict_labels(test_v)

        correct = np.sum(pred_l == test_l)
        total = len(pred_l)
        if(verbose):
            print("Correctly identified %i out of %i labels! \nPositve rate is: %f" % (correct, total, correct/total))
        return(correct/total)
        

        
    def evaluate_classifier(self, filename_data, filename_labels, hour = 0):
        """
        Evaluates an already trained classifier with a new set of data and labels
        
        Parameters
        ----------
        filename_data : string
            Filename of the sattelit data set

        filename_labels : string
            The data of the corresponding labels, if given  

        hour : int
            0-23, hour of the day at which the data sets are read
        """
        if(self.cl is None):
            print("No classifer trained or loaded")
            return
        self.create_test_vectors(filename_data, hour, self.cDV, self.kOV)
        self.predict_labels()
        org_labels = dh.exctract_labels_fromFile(filename_labels, self.pred_indices, hour)

        correct = np.sum(self.pred_labels == org_labels)
        total = len(org_labels)
        print("Correctly identified %i out of %i labels! \nPositve rate is: %f" % (correct, total, correct/total))
  








    ##################################################################################
    #################         Saving and Loading parts of the data
    ##################################################################################

    # def export_labels(self, filename):
    #     """
    #     Saves predicted labels as netcdf file

    #     Parameters
    #     ----------
    #     filename : string
    #         Name of the file in which the labels will be written
    #     """
    #     if (self.pred_labels is None or self.pred_filename is None
    #             or self.pred_indices is None):
    #         print("Unsufficant data for saving labels")
    #         return  
    #     data = dh.imbed_data(self.pred_labels, self.pred_indices, self.pred_filename)
    #     dh.write_NETCDF(data, filename)
    

    def save_classifier(self, filename):
        """
        Saves Classifier

        Parameters
        ----------
        filename : string
            Name of the file into which the classifier is saved.
        """
        dump(self.cl, filename)


    def load_classifier(self, filename):
        """
        Loads classifer        
        Parameters
        ----------
        filename : string
            Name if the file the classifier is loaded from.
        """
        self.cl = load(filename)



