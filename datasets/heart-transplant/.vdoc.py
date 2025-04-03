# type: ignore
# flake8: noqa
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| eval: false

"https://raw.githubusercontent.com/Aranaur/aranaur.rbind.io/main/datasets/heart-transplant/happiness.csv"
#
#
#
#| echo: false

import pandas as pd

pd.read_csv('happiness.csv', index_col=0, sep=',')
#
#
#
#
#
#
