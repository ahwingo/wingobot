# Add the paths of the modules in this folder to the system path.
import os
import sys

# Add the score-estimator module.
path_to_add = os.path.join(__path__[0], "score_estimator")
sys.path.append(path_to_add)
