import os
import sys
import builtins
import warnings

for path in [os.path.join(os.environ["REPO"], "py_src"), os.environ["REPO"]]:
    if path not in sys.path:
        print(f"Adding {path} to sys_path")
        sys.path.append(path)

from IPython import get_ipython
from IPython.lib import deepreload
from IPython.display import display, HTML
from IPython.core.display_functions import clear_output

builtins.reload = deepreload.reload
ipython = get_ipython()
if "ipython" in globals():
    print("loading ipython commands")
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("reload_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")
    ipython.run_line_magic("load_ext", "jupyter_black")

import git
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import *

warnings.filterwarnings(
    "ignore",
    message="Engine has switched to 'python' because numexpr does not support extension array dtypes",
    category=RuntimeWarning,
)

from pathvalidate import sanitize_filename
from functools import partial

import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import requests

requests.packages.urllib3.disable_warnings()

print(f"Startup Script succesfully loaded: {os.path.abspath(__file__)}")
