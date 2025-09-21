import pandas as pd

big_frame = pd.read_csv("data/generated_data.csv")
big_frame = big_frame.dropna()
big_frame = big_frame[big_frame.command.apply(lambda x:x[0]!="#")]

big_frame[["command", "description"]].to_csv("data/filtered_data.csv", index=None)