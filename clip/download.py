import opendatasets as od 
import pandas as pd
import os 

# Download the data
url = "https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset"
root_path = os.path.dirname(os.path.realpath(__file__))

od.download(url)

# Prepare Labels
df = pd.read_csv("flickr-image-dataset/flickr30k_images/results.csv", delimiter="|")
df.columns = ['image', 'caption_number', 'caption']
df['caption'] = df['caption'].str.lstrip()
df['caption_number'] = df['caption_number'].str.lstrip()
df.loc[19999, 'caption_number'] = "4"
df.loc[19999, 'caption'] = "A dog runs across the grass ."
ids = [id_ for id_ in range(len(df) // 5) for i in range(5)]
df['id'] = ids
df.to_csv("flickr-image-dataset/captions.csv", index=False)
df.head()
