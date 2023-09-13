import pandas as pd
from tqdm import tqdm


data = pd.read_csv("./dataset/IMDB.csv")
new_data = pd.DataFrame(columns=["text"])

for r, (i, s) in zip(tqdm(data["review"]), enumerate(data["sentiment"])):
    new_data.loc[i] = ["Below is an Review that describes a movie. Write a Sentiment that appropriately completes the request. ### Review: " + r + "### Sentiment: " + s]

print(new_data.head())
new_data.to_csv("./train.csv")
