import pandas as pd
import json

big_frame = pd.read_csv("data/generated_data.csv")
big_frame = big_frame.dropna()
big_frame = big_frame[big_frame.command.apply(lambda x:x[0]!="#")]

with open("data/chatGPT_generated_data.json") as f:
    d = f.read()
    j = json.loads(d)


gpt_frame = pd.DataFrame(j.values()).rename(columns={"invocation":"description", "cmd":"command"})


nl2bash_frame = pd.read_csv("data/nl2bash.csv").rename(columns={"nl_command":"description", "bash_code":"command"})


pd.concat([big_frame[["command", "description"]], gpt_frame[["command", "description"]], nl2bash_frame[["command", "description"]]]).to_csv("data/train.csv", index=None)