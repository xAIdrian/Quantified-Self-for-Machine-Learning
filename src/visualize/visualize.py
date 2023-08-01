import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from IPython.display import display

# We want to recognize the differences in datasets
# These can be used to train a ML model

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df = pd.read_pickle("../../data/interim/01_data_processed.pkl")

# --------------------------------------------------------------
# Plot single columns
# --------------------------------------------------------------

# let's see why making our sets column is so convenient
# we want the original dataframe where the set column is set 1
set_df = df[df["set"] == 1]
plt.plot(set_df["acc_y"])

# using index we remove the fractions of time
plt.plot(set_df["acc_y"].reset_index(drop=True))

# --------------------------------------------------------------
# Plot all exercises
#
# create a loop to plot all exercises with multiple subsets
# --------------------------------------------------------------

for label in df["label"].unique():
    subset = df[df["label"] == label]
    # display(subset.head(2))

    # create a canvas to start that we can add to later
    fig, ax = plt.subplots()
    # new index for a new dataframe we creating. plot 1st 100 only
    plt.plot(subset[:100]["acc_y"].reset_index(drop=True), label=label)
    plt.legend()
    plt.show()

# --------------------------------------------------------------
# Adjust plot settings
#
# Matplotlib tricks. change RC params (runtime RC)
# --------------------------------------------------------------

mpl.style.use("seaborn-v0_8-deep")

mpl.rcParams["figure.figsize"] = (20, 5)
mpl.rcParams["figure.dpi"] = 100  # figure resolution

# --------------------------------------------------------------
# Compare medium vs. heavy sets
# --------------------------------------------------------------

# alternative method of getting specific dataframe data
category_df = df.query(" label == 'squat' ").query(" participant == 'A' ").reset_index()

# group them limit to a single column
fig, ax = plt.subplots()
category_df.groupby(["category"])["acc_y"].plot()
ax.set_ylabel("acc_y")
ax.set_xlabel("samples")
plt.legend()

# --------------------------------------------------------------
# Compare participants
# --------------------------------------------------------------

# note: when we query the index will not be ordered or starting from 0
# we sort and reset_index to start our new DF from 0
participant_df = df.query(" label == 'bench' ").sort_values("participant").reset_index()

fig, ax = plt.subplots()
participant_df.groupby(["participant"])["acc_y"].plot()
ax.set_ylabel("acc_y")
ax.set_xlabel("samples")
plt.legend()

# --------------------------------------------------------------
# Plot multiple axis
# --------------------------------------------------------------

label = "squat"
participant = "A"
all_axis_df = (
    df.query(f" label == '{label}' ")
    .query(f" participant == '{participant}' ")
    .reset_index()
)

fig, ax = plt.subplots()
all_axis_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax)
ax.set_ylabel("acc_y")
ax.set_xlabel("samples")
plt.legend()

# --------------------------------------------------------------
# Create a loop to plot all combinations per sensor
# --------------------------------------------------------------

labels = df["label"].unique()
participants = df["participant"].unique()

# accelerometer
for label in labels:
    for participant in participants:
        all_axis_df = (
            df.query(f" label == '{label}' ")
            .query(f" participant == '{participant}' ")
            .reset_index()
        )

        if len(all_axis_df) > 0:
            fig, ax = plt.subplots()
            all_axis_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax)
            ax.set_ylabel("acc_y")
            ax.set_xlabel("samples")
            plt.title(f"{label} {participant}".title())
            plt.legend()

for label in labels:
    for participant in participants:
        all_axis_df = (
            df.query(f" label == '{label}' ")
            .query(f" participant == '{participant}' ")
            .reset_index()
        )

        if len(all_axis_df) > 0:
            fig, ax = plt.subplots()
            all_axis_df[["gyr_x", "gyr_y", "gyr_z"]].plot(ax=ax)
            ax.set_ylabel("gyr_y")
            ax.set_xlabel("samples")
            plt.title(f"{label} {participant}".title())
            plt.legend()

# --------------------------------------------------------------
# Combine plots in one figure
# --------------------------------------------------------------

label = "row"
participant = "A"
combined_plot_df = (
    df.query(f" label == '{label}' ")
    .query(f" participant == '{participant}' ")
    .reset_index(drop=True)
)

fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
combined_plot_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax[0])
combined_plot_df[["gyr_x", "gyr_y", "gyr_z"]].plot(ax=ax[1])

ax[0].legend(
    loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True
)
ax[1].legend(
    loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True
)
ax[1].set_xlabel("samples")

# --------------------------------------------------------------
# Loop over all combinations and export for both sensors
# --------------------------------------------------------------

labels = df["label"].unique()
participants = df["participant"].unique()

for label in labels:
    for participant in participants:
        combined_plot_df = (
            df.query(f" label == '{label}' ")
            .query(f" participant == '{participant}' ")
            .reset_index()
        )

        if len(combined_plot_df) > 0:
            fixx, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
            combined_plot_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax[0])
            combined_plot_df[["gyr_x", "gyr_y", "gyr_z"]].plot(ax=ax[1])

            ax[0].legend(
                loc='upper center',
                bbox_to_anchor=(0.5, 1.15),
                ncol=3,
                fancybox=True,
                shadow=True
            )
            ax[1].legend(
                loc='upper center',
                bbox_to_anchor=(0.5, 1.15),
                ncol=3,
                fancybox=True,
                shadow=True
            )
            ax[1].set_xlabel('samples')

            # export to files
            plt.savefig(f"../../reports/figures/{label.title()} ({participant}).png")

            plt.show()


