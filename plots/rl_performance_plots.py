import seaborn as sns
import pandas
import matplotlib.pyplot as plt
import os

from dir import PROJECT_DIR

data_dir = os.path.join(PROJECT_DIR, "report")

df = pandas.read_csv(os.path.join(data_dir, "hit_rate_11262023101705.csv"))
step = 500
lens = len(df.index)
up = 0
low = step
df_res = []

for i in range(round(lens / step)):
    df_new = df.iloc[up:low]
    if df_new.empty:
        break
    df_len = len(df_new.index)
    data = {
        "step": [df_new["step"].iloc[-1]],
        "x_val": [df_new["x_val"].max()],
        "y_val": [df_new["y_val"].max()],
    }
    df_new_2 = pandas.DataFrame.from_dict(data)
    up = up + step
    low = low + step
    df_res.append(df_new_2)
df_res = pandas.concat(df_res)
print(df_res)

# csv = pandas.read_csv(os.path.join(data_dir, 'csv_data/performance.csv'))
# res = sns.lineplot(x="Name", y="Age", data=csv)

sns.set_theme(style="darkgrid")

# Load an example dataset with long-form data
# fmri = sns.load_dataset("fmri")

# x_val,y_val,step

# Plot the responses for different events and regions
s = sns.lineplot(y="x_val", x="step", data=df_res)

g = sns.lmplot(data=df_res, x="step", y="x_val", height=5)
g.set_axis_labels("steps", "rewards")

# s2 = sns.lineplot(y="y_val", x="step",
#              data=df_res)
#
# d = sns.lmplot(
#     data=df_res,
#     x="step", y="y_val",
#     height=5
# )
# d.set_axis_labels("steps", "location accuracy")


# sns.jointplot(x="step", y="x_val", data=csv,
#                   kind="reg", truncate=False,
#                   xlim=(0, 1000), ylim=(0, 120),
#                   color="m", height=7)

# sns.residplot(x="step", y="x_val", lowess=True, color="g", data=csv)
plt.show()
