import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from ydata_profiling import ProfileReport
import webbrowser
import os

def data_profiling(df, output_dir, output_filename, dashboard_title):
    output_path = os.path.join(output_dir, output_filename)

    #Check direction
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    #Data profiling
    print("Running Profile Report...")
    profile = ProfileReport(df, title=dashboard_title, explorative=True)
    profile.to_file(output_path)

    #Open profiling
    full_path = os.path.abspath(output_path)
    print(f"Open report: {full_path}")
    webbrowser.open(f"file://{full_path}")

