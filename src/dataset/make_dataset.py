import pandas as pd
from glob import glob

# --------------------------------------------------------------
# Read single CSV file
# --------------------------------------------------------------

single_file_acc = pd.read_csv(
    "../../data/raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv"
)
single_file_gyr = pd.read_csv(
    "../../data/raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Gyroscope_25.000Hz_1.4.4.csv"
)

# --------------------------------------------------------------
# List only the csv files in the folder
# --------------------------------------------------------------

all_files = glob("../../data/raw/MetaMotion/*.csv")
print(f"we are working with {len(all_files)} files")

# --------------------------------------------------------------
# Extract features from filename
#
# we want to extract pieces of this file and add to dataframe
# this is where we will just validate the data preparation methods
# --------------------------------------------------------------

data_path = "../../data/raw/MetaMotion/"
file_path = all_files[0]
split_items = file_path.split("-")

participant = split_items[0].replace(data_path, "")
label = split_items[1]
# remove the number from our workout category
category = split_items[2].rstrip("123").rstrip("_MetaWear_2019")

# add vars to dataframe by creating new columns
df = pd.read_csv(file_path)
df["participant"] = participant
df["label"] = label
df["category"] = category

# --------------------------------------------------------------
# Read all files
#
# apply verified data preparation methods to all the files (try to avoid loops)
# --------------------------------------------------------------

acc_df = pd.DataFrame()
gyr_df = pd.DataFrame()

acc_set = 1
gyr_set = 1

for file in all_files:
    split_items = file_path.split("-")

    participant = split_items[0].replace(data_path, "")
    label = split_items[1]
    # remove the number from our workout category
    category = split_items[2].rstrip("123").rstrip("_MetaWear_2019")

    curr_df = pd.read_csv(file)
    curr_df["participant"] = participant
    curr_df["label"] = label
    curr_df["category"] = category

    # continue to add new rows (series) to the dataframe
    # 'set' is just an arbitrary identifier
    if "Accelerometer" in file:
        curr_df["set"] = acc_set
        acc_set += 1
        acc_df = pd.concat([acc_df, curr_df])

    if "Gyroscope" in file:
        curr_df["set"] = gyr_set
        gyr_set += 1
        gyr_df = pd.concat([gyr_df, curr_df])

# gyr_df[gyr_df['set'] == 1]

# --------------------------------------------------------------
# Working with datetimes
# --------------------------------------------------------------

acc_df.info()

pd.to_datetime(df["epoch (ms)"], unit="ms")

# set datetime as index to convert to a time series database
acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit="ms")
gyr_df.index = pd.to_datetime(gyr_df["epoch (ms)"], unit="ms")

# now we can get rid of all features referencing time
del acc_df["epoch (ms)"]
del acc_df["time (01:00)"]
del acc_df["elapsed (s)"]

del gyr_df["epoch (ms)"]
del gyr_df["time (01:00)"]
del gyr_df["elapsed (s)"]

# --------------------------------------------------------------
# Turn into function
# --------------------------------------------------------------

all_files = glob("../../data/raw/MetaMotion/*.csv")


def get_data_from_all_files(files=all_files):
    data_path = "../../data/raw/MetaMotion/"

    acc_df = pd.DataFrame()
    gyr_df = pd.DataFrame()

    acc_set = 1
    gyr_set = 1

    for file in files:
        split_items = file.split("-")

        participant = split_items[0].replace(data_path, "")
        label = split_items[1]
        # remove the number from our workout category
        category = split_items[2].rstrip("123").rstrip("_MetaWear_2019")

        curr_df = pd.read_csv(file)
        curr_df["participant"] = participant
        curr_df["label"] = label
        curr_df["category"] = category

        # continue to add new rows (series) to the dataframe
        # 'set' is just an arbitrary identifier
        if "Accelerometer" in file:
            curr_df["set"] = acc_set
            acc_set += 1
            acc_df = pd.concat([acc_df, curr_df])

        if "Gyroscope" in file:
            curr_df["set"] = gyr_set
            gyr_set += 1
            gyr_df = pd.concat([gyr_df, curr_df])

    # set datetime as index to convert to a time series database
    acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit="ms")
    gyr_df.index = pd.to_datetime(gyr_df["epoch (ms)"], unit="ms")

    # now we can get rid of all features referencing time
    del acc_df["epoch (ms)"]
    del acc_df["time (01:00)"]
    del acc_df["elapsed (s)"]

    del gyr_df["epoch (ms)"]
    del gyr_df["time (01:00)"]
    del gyr_df["elapsed (s)"]

    return acc_df, gyr_df


processed_acc_df, processed_gyr_df = get_data_from_all_files(all_files)

# --------------------------------------------------------------
# Merging datasets
#
# Ideally we have one dataframe
# --------------------------------------------------------------

data_merged = pd.concat(
    [
        # select a column range, first 3 columns
        processed_acc_df.iloc[:, :3],
        processed_gyr_df,
    ],
    axis=1,
)

# rename our columns
data_merged.columns = [
    "acc_x",
    "acc_y",
    "acc_z",
    "gyr_x",
    "gyr_y",
    "gyr_z",
    "participant",
    "label",
    "category",
    "set",
]

# --------------------------------------------------------------
# Resample data (frequency conversion)
#
# datasets were recorded at different frequencies
# we need to align our series to fix missing values
#
# Accelerometer:    12.500HZ
# Gyroscope:        25.000Hz (more meaurements per second)
#
# perform a calculation on all data items as we "adjust up/down"
# the frequency for our misaligned items
# --------------------------------------------------------------

# find our optimum "resampling rate"
# each resampling requires a custom aggregation
sampling = {
    "acc_x": "mean",
    "acc_y": "mean",
    "acc_z": "mean",
    "gyr_x": "mean",
    "gyr_y": "mean",
    "gyr_z": "mean",
    "participant": "last",
    "label": "last",
    "category": "last",
    "set": "last",
}

data_merged[:1000].resample(rule="200ms").apply(sampling)

# a bit of work is required before applying to our entire dataframe
# there would be too much data with un-configured imputation

# so lets split by day
# we are dropping null rows and then regrouping our data
days = [g for n, g in data_merged.groupby(pd.Grouper(freq="D"))]
data_resampled = pd.concat(
    [df.resample(rule="200ms").apply(sampling).dropna() for df in days]
)
data_resampled["set"] = data_resampled["set"].astype("int")

# --------------------------------------------------------------
# Export datasetå
# --------------------------------------------------------------

# we can export df in serialized format using pickle (static conversions, smaller, faster)
# keep a copy as .csv for sharing and knowledge share
data_resampled.to_pickle("../../data/interim/01_data_processed.pkl")
data_resampled
