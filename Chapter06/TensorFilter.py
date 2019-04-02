import numpy as np
import pandas as pd
import tensorflow as tf

Data = pd.read_csv('ratings.csv', sep=';', names=['user', 'item', 'rating', 'timestamp'], header=None)

Data = Data.iloc[:,0:3]

NumItems = Data.item.nunique() 
NumUsers = Data.user.nunique()

print('Item: ', NumItems)
print('Users: ', NumUsers)


Data['rating'] = Data['rating'].values.astype(float)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
DataScaled = pd.DataFrame(scaler.fit_transform(Data['rating'].values.reshape(-1,1)))
Data['rating'] = DataScaled

UserItemMatrix = Data.pivot(index='user', columns='item', values='rating')
UserItemMatrix.fillna(0, inplace=True)

Users = UserItemMatrix.index.tolist()
Items = UserItemMatrix.columns.tolist()

UserItemMatrix = UserItemMatrix.as_matrix()

NumInput = NumItems
NumHidden1 = 10
NumHidden2 = 5

X = tf.placeholder(tf.float64, [None, NumInput])

weights = {
    'EncoderH1': tf.Variable(tf.random_normal([NumInput, NumHidden1], dtype=tf.float64)),
    'EncoderH2': tf.Variable(tf.random_normal([NumHidden1, NumHidden2], dtype=tf.float64)),
    'DecoderH1': tf.Variable(tf.random_normal([NumHidden2, NumHidden1], dtype=tf.float64)),
    'DecoderH2': tf.Variable(tf.random_normal([NumHidden1, NumInput], dtype=tf.float64)),
}

biases = {
    'EncoderB1': tf.Variable(tf.random_normal([NumHidden1], dtype=tf.float64)),
    'EncoderB2': tf.Variable(tf.random_normal([NumHidden2], dtype=tf.float64)),
    'DecoderB1': tf.Variable(tf.random_normal([NumHidden1], dtype=tf.float64)),
    'DecoderB2': tf.Variable(tf.random_normal([NumInput], dtype=tf.float64)),
}


def encoder(x):
    Layer1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['EncoderH1']), biases['EncoderB1']))
    Layer2 = tf.nn.sigmoid(tf.add(tf.matmul(Layer1, weights['EncoderH2']), biases['EncoderB2']))
    return Layer2


# Building the decoder

def decoder(x):
    Layer1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['DecoderH1']), biases['DecoderB1']))
    Layer2 = tf.nn.sigmoid(tf.add(tf.matmul(Layer1, weights['DecoderH2']), biases['DecoderB2']))
    return Layer2


EncoderOp = encoder(X)
DecoderOp = decoder(EncoderOp)


YPred = DecoderOp

YTrue = X

loss = tf.losses.mean_squared_error(YTrue, YPred)
Optimizer = tf.train.RMSPropOptimizer(0.03).minimize(loss)


EvalX = tf.placeholder(tf.int32, )
EvalY = tf.placeholder(tf.int32, )
Pre, PreOp = tf.metrics.precision(labels=EvalX, predictions=EvalY)

Init = tf.global_variables_initializer()
LocalInit = tf.local_variables_initializer()

PredData = pd.DataFrame()

with tf.Session() as session:
    Epochs = 120
    BatchSize = 200

    session.run(Init)
    session.run(LocalInit)

    NumBatches = int(UserItemMatrix.shape[0] / BatchSize)
    UserItemMatrix = np.array_split(UserItemMatrix, NumBatches)
    
    for i in range(Epochs):

        AvgCost = 0

        for batch in UserItemMatrix:
            _, l = session.run([Optimizer, loss], feed_dict={X: batch})
            AvgCost += l

        AvgCost /= NumBatches

        print("Epoch: {} Loss: {}".format(i + 1, AvgCost))

    UserItemMatrix = np.concatenate(UserItemMatrix, axis=0)

    Preds = session.run(DecoderOp, feed_dict={X: UserItemMatrix})

    PredData = PredData.append(pd.DataFrame(Preds))

    PredData = PredData.stack().reset_index(name='rating')
    PredData.columns = ['user', 'item', 'rating']
    PredData['user'] = PredData['user'].map(lambda value: Users[value])
    PredData['item'] = PredData['item'].map(lambda value: Items[value])
    
    keys = ['user', 'item']
    Index1 = PredData.set_index(keys).index
    Index2 = Data.set_index(keys).index

    TopTenRanked = PredData[~Index1.isin(Index2)]
    TopTenRanked = TopTenRanked.sort_values(['user', 'rating'], ascending=[True, False])
    TopTenRanked = TopTenRanked.groupby('user').head(10)
    
    print(TopTenRanked.head(n=10))
    
    

    


