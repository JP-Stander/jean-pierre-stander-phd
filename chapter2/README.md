The data used for the article is an open source data set called "labeled faces in the wild" and can be downloaded here http://vis-www.cs.umass.edu/lfw/#download.
When downloading the code all the code needs to be places in the same folder along with 2 folders called results and lfw. results is where the results of the scripts are saved
lfw is the folder containing the data as unzipped when downloaded.

In each python script there is a variable called working_directory which should be set to the location where the scripts are saved and then the scripts can be executed.
Note that add_and_smooth_noise_all_smoothers.py needs to be executed first.

Figure 2: Generated using increase_noise_systematically.py

Table 1: Values obtained from executing pmax_wmax_investigation.py

Table 2: Values obtained from executing hyperparameters_ls_afm_effect.py

Figure 4: Generated using plot_within_between_variation.py

Table 3:  Values obtained from executing create_summary_table_of_results.py after executing add_and_smooth_noise_all_smoothers.py

Figures in Appendix: Created by executing plot_results_of_simulation_study.py after executing add_and_smooth_noise_all_smoothers.py