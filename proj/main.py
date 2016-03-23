from setup_data import *
from baseline import baseline
from random_sampling import random_sampling
from least_confident import least_confident
from minimum_margin import minimum_margin
from maximum_entropy_simple_proba import maximum_entropy_simple_proba
from maximum_entropy_onnevsrest_proba_log_regr import maximum_entropy_onnevsrest_proba_log_regr
from minimum_margin_init import minimum_margin_init
from minimum_margin_select_start_wn import minimum_margin_select_start_wn
from minimum_margin_stemming import minimum_margin_stemming
from minimum_margin_select_start_wn_stemming import minimum_margin_select_start_wn_stemming


baseline()
random_sampling()
#least_confident()

minimum_margin()
#minimum_margin_select_start_wn()
#minimum_margin_select_start_wn_stemming()
minimum_margin_stemming()

#minimum_margin_init()

#maximum_entropy_simple_proba()
#maximum_entropy_onnevsrest_proba_log_regr()