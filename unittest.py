import sparsenet

sn = sparsenet.SparseNet()
sn.run(batch_size=1000, num_trials=20)
sn.run(batch_size=5000, num_trials=20)
