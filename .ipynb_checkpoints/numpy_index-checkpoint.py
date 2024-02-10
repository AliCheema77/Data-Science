import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

mat = pd.read_csv("covid_data.csv")
print(np.max(mat["new_deaths"]) - np.min(mat["new_deaths"]))
plt.bar(["max", "mid"], [np.max(mat["new_deaths"])<10000, np.average(mat["new_deaths"])], color=["red", 'blue'])
plt.show()