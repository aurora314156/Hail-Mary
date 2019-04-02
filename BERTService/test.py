a = [1,2,3,4,5,6]

import numpy as np


b = np.frombuffer(a, dtype=np.float)
print(type(b))


from sklearn.preprocessing import Normalizer
norm1 = Normalizer(norm='l2')
print(norm1.fit_transform(b))
