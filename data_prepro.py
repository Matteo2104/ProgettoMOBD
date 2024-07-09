import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

df = pd.read_csv("mushroom.csv")

# one hot encoding
one_hot_encoder = OneHotEncoder(sparse=False)
one_hot_encoded = one_hot_encoder.fit_transform(df[['class']])
one_hot_columns = [f'class_{i}' for i in range(one_hot_encoded.shape[1])]
one_hot_df = pd.DataFrame(one_hot_encoded, columns=one_hot_columns)
df = pd.concat([df.drop(columns=["class"]), one_hot_df], axis=1)

# normalizzo i valori delle features nel range [0,1]
df = (df - df.min()) / (df.max() - df.min())

# divido train set e test set
train_df, test_df = train_test_split(df, test_size=0.2)

# salvo i dataset
train_df.to_csv("mushroom_train.csv",index=False)
test_df.to_csv("mushroom_test.csv",index=False)
