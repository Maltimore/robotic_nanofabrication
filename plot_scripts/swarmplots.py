import argparse
import os
# gridsearch analysis is the python package at github.com/maltimore/gridsearch_helper
from gridsearch_analysis import collect_results, plotting

parser = argparse.ArgumentParser()
parser.add_argument('--path', help='destination of results')
args = parser.parse_args()

# COLLECT RESULTS
results_path = os.path.join(args.path, 'results')
df = collect_results.collect_results(results_path)
df.to_csv(os.path.join(args.path, 'results.csv'))

# set first_sucess values of runs that DID finish but didn't have a success to 150 (just for plotting,
# will be clear in the paper that these were in fact >= 150)
df.loc[df['first_success'].isnull(), 'first_success'] = 150

# PLOTS
# Specify which columns to plot as strings in a list. List can be empty.
RELEVANT_PARAMETERS = ['name']  # list of strings (list can be empty)
# what variable to use as performance measure
TARGET_COLUMN = 'first_success'  # string
# is the performance better when the target variable is lower or when it is higher?
LOWER_IS_BETTER = True  # bool
# split up the analysis into separate parts for unique values in this column
SPLIT_ANALYSIS_COLUMN = None  # string or None
# only for 1 relevant parameter (len(RELEVANT_PARAMETERS) == 1): the order in which
# to present the swarm/bar plot variables
# should be None if no order is specified
# this can also be used to control *which* entries are presented at all by only including the
# relevant onces in the list
VAR_ORDER = ['default', 'model_based', 'rupture_avoidance', 'both']
plot_path = os.path.join('plots')
if not os.path.exists(plot_path):
    os.makedirs(plot_path)
plotting.plot(df, plot_path, RELEVANT_PARAMETERS, TARGET_COLUMN, LOWER_IS_BETTER, SPLIT_ANALYSIS_COLUMN, VAR_ORDER)
