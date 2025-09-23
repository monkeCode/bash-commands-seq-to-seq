import pandas as pd
import json

big_frame = pd.read_csv("data/generated_data.csv")
big_frame = big_frame.dropna()
big_frame = big_frame[big_frame.command.apply(lambda x:x[0]!="#")]

internet_frame = pd.read_csv("data/inet_dataset.csv")


pd.concat([big_frame[["command", "description"]], internet_frame[["command", "description"]]]).to_csv("data/train.csv", index=None)