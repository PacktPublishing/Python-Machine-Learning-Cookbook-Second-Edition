import pyltr

with open('train.txt') as trainfile, \
        open('vali.txt') as valifile, \
        open('test.txt') as testfile:
    TrainX, Trainy, Trainqids, _ = pyltr.data.letor.read_dataset(trainfile)
    ValX, Valy, Valqids, _ = pyltr.data.letor.read_dataset(valifile)
    TestX, Testy, Testqids, _ = pyltr.data.letor.read_dataset(testfile)
    metric = pyltr.metrics.NDCG(k=10)

# Only needed if you want to perform validation (early stopping & trimming)
monitor = pyltr.models.monitors.ValidationMonitor(
    ValX, Valy, Valqids, metric=metric, stop_after=250)

model = pyltr.models.LambdaMART(
    metric=metric,
    n_estimators=1000,
    learning_rate=0.02,
    max_features=0.5,
    query_subsample=0.5,
    max_leaf_nodes=10,
    min_samples_leaf=64,
    verbose=1,
)

model.fit(TestX, Testy, Testqids, monitor=monitor)

Testpred = model.predict(TestX)
print('Random ranking:', metric.calc_mean_random(Testqids, Testy))
print('Our model:', metric.calc_mean(Testqids, Testy, Testpred))
