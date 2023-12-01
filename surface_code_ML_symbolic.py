import numpy as np
from tqdm import tqdm
class surfaceMLdecoder():
    def __init__(self, lx, hx, lz, hz, pX, pY, pZ):
        self.lz = lz
        self.hz = hz
        self.lx = lx
        self.hx = hx
        self.pX = pX
        self.pZ = pZ
        self.pY = pY
        self.pI = 1 - pX - pY - pZ
        self.syndrome_dictionary = {} #key: syndrome, value: [weight I class, weight LZ class, weight LX class, weight LY class]
        self.error_loop()
    def error_loop(self):
        n = len(self.lz[0])
        #loop over all errors
        for i in tqdm(range(2**n)):
            for j in range(2**n):

                errorx = np.array([int(x) for x in bin(i)[2:].zfill(n)])
                errorz = np.array([int(x) for x in bin(j)[2:].zfill(n)])

                ey = errorx*errorz
                hamming_weight_x_errors = np.sum((ey+errorx)%2)
                hamming_weight_z_errors = np.sum((ey+errorz)%2)
                hamming_weight_y_errors = np.sum(ey)
                ham_sum = hamming_weight_x_errors + hamming_weight_z_errors + hamming_weight_y_errors
                error_probability = (self.pI)**(n-ham_sum)*self.pX**hamming_weight_x_errors*self.pZ**hamming_weight_z_errors*self.pY**hamming_weight_y_errors
                syndrome = np.append((self.hz@errorx)%2, (self.hx@errorz)%2)
                syndrome_string = np.array2string(syndrome)
                error_class_x = int((self.lz[0]@errorx)%2)
                error_class_z = int((self.lx[0]@errorz)%2)
                error_class = error_class_x*2+ error_class_z
                if syndrome_string in self.syndrome_dictionary:
                    self.syndrome_dictionary[syndrome_string][error_class] += error_probability
                else:
                    self.syndrome_dictionary[syndrome_string] = [0,0,0,0]
                    self.syndrome_dictionary[syndrome_string][error_class] += error_probability
    def decode(self, error):
        syndrome = (self.hz @ error) % 2
        syndrome_string = np.array2string(syndrome)
        class_weights = self.syndrome_dictionary[syndrome_string]
        return max(range(len(class_weights)), key=class_weights.__getitem__)



