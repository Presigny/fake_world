"""
author: Charley Presigny
These two functions load and save dictionnaries of data in the pickle format"""
import pickle

def load_results(net):
     with open(net,'rb') as f:
         dico = pickle.load(f)
     return dico
 
def save_results(name,class_to_save):
     with open(name, 'wb') as f:
         pickle.dump(class_to_save, f)

