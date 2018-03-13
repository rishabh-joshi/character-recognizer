develop/models/dense_cnn_balanced.h5: develop/src/models/model_balanced_dense.py
	python develop/src/models/model_balanced_dense.py

dense_cnn_balanced: develop/models/dense_cnn_balanced.h5

develop/models/sparse_cnn_balanced.h5: develop/src/models/model_balanced_sparse.py
	python develop/src/models/model_balanced_sparse.py

sparse_cnn_balanced: develop/models/sparse_cnn_balanced.h5

develop/models/dense_cnn_byclass.h5: develop/src/models/model_byclass_dense.py
	python develop/src/models/model_byclass_dense.py

dense_cnn_bycalss: develop/models/dense_cnn_bycalss.h5

develop/models/sparse_cnn_byclass.h5: develop/src/models/model_byclass_sparse.py
	python develop/src/models/model_byclass_sparse.py

sparse_cnn_byclass: develop/models/sparse_cnn_byclass.h5

all: dense_cnn_balanced sparse_cnn_balanced dense_cnn_bycalss sparse_cnn_byclass