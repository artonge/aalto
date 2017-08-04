import numpy as np

def buildC(N, size, i=0, base=[]):
	if len(base) == 0:
		base = [0]*size
	if i == size   :
		return [base]

	subC = np.empty([0, size], dtype=int)

	for n in range(N+1):
		b = np.array(base)
		b[i] = n
		subC = np.concatenate((subC, buildC(N, size, i+1, b)))

	return subC


print(buildC(1, 2))
