#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'src\\python'))
	print(os.getcwd())
except:
	pass
#%% [markdown]
# # Data Transformation Experiments
# 
# ## This notebook presents a set of experiments running for data trans.
# ## First, we need to set up the environment and observers.

#%%

import pandas as pd

from sacred import Experiment
from sacred.observers import MongoObserver

#%% [markdown]
# ## The set of following functions below supports running these expriments.  

#%%
from pathlib import Path
from datafc.eval import Evaluator

data_folder = Path("../../data/standard")
ex = Experiment("jupyter_ex", interactive=True)
ex.observers.append(MongoObserver.create())
ex.add_config("../../conf/exp_config.yaml")

#%% [markdown]
# ## Results are added to MongoDB for experiment reproduction

#%%

@ex.main
def run_experiment(dataset, mapping_method, mapping_features, with_flashfill):
    evaluator = Evaluator(mapping_method, mapping_features)
    return evaluator.run_dataset(Path("../../data") / dataset, mapping_method, mapping_features, with_flashfill)

dataset_report = ex.run().result

#%%
scenarios_df = pd.DataFrame(dataset_report["scenarios"], columns=["name", "running_time", "active_learning_curve"]).round(2)
scenarios_df



