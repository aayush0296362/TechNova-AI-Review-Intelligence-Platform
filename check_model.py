import pickle
import sys

try:
    m = pickle.load(open('model.pkl','rb'))
except Exception as e:
    print('LOAD_ERROR', e)
    sys.exit(1)

print('TYPE', type(m))
print('HAS_PREDICT_PROBA', hasattr(m, 'predict_proba'))
print('HAS_PREDICT', hasattr(m, 'predict'))
print('ATTRIBUTES', [a for a in dir(m) if not a.startswith('_')][:50])

try:
    import numpy as np
    if hasattr(m, 'classes_'):
        print('CLASSES_', m.classes_)
except Exception as e:
    print('META_ERROR', e)
