import pandas as pd

big_frame = pd.read_csv("data/generated_data.csv")
big_frame = big_frame.dropna()
big_frame = big_frame[big_frame.command.apply(lambda x: x[0] != "#")]
big_frame = big_frame[big_frame.is_command]

internet_frame = pd.read_csv("data/inet_dataset.csv")

yet_another_commands = pd.read_json("data/commands.json")
df = pd.read_csv("hf://datasets/westenfelder/NL2SH-ALFA/train.csv")
df1 = df[["nl", "bash"]].rename({"nl": "description", "bash": "command"}, axis=1)

r = pd.concat(
    [
        df1,
        big_frame[["command", "description"]],
        internet_frame[["command", "description"]],
        yet_another_commands[["command", "description"]],
    ]
).drop_duplicates(["description", "command"])
print(f"generated: {len(big_frame)}")
print(f"internet: {len(internet_frame)}")
print(f"NL2SH-ALFA: {len(df1)}")
print(f"result: {len(r)}")
r.to_csv("data/train.csv", index=None)
