from labeler import Labelers
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import map_overlay
import time


class EM():

    def show_probs(self):
        plt.figure()
        probs = self.G[:,1]
        mask = self.haiti_map.seg_convert(20, probs)
        plt.imshow(mask,cmap = 'seismic', norm = plt.Normalize(0,1))
        #plt.figure()
        #plt.imshow(self.haiti_map.mask_helper(self.haiti_map.img, mask>.5, .4))
        


    def __init__(self, haiti_map, labelers, indcs = None):
        self.labelers = labelers
        self.haiti_map = haiti_map
        if indcs is None:
            relevant_labels = self.haiti_map.unique_segs(20)
        else:
            relevant_labels = indcs
        probs = self.labelers.probability(label_indices = relevant_labels)
        labels = self.labelers.labels[:,relevant_labels] # (W, I)
        self.I = labels.shape[1] #Number of segments
        self.W = labels.shape[0] #Number of Labelers
        self.K = 2
        self.n = np.zeros((self.W, self.I, self.K)) # (W, I, K)
        self.n[:, :, 0] = (1-labels)
        self.n[:, :, 1] = labels
        self.G = np.sum(self.n, axis = 0) # (I, K)
        self.G = self.G/np.sum(self.G, axis = 1).reshape((self.I,1))
        self.alpha = np.zeros((self.W,self.K,self.K)) # (W, K, K)
        self.p = np.zeros(self.K)


    def M(self):
        old_G = self.G.copy()
        new_G = np.ones((self.I, self.K))
        for k in range(self.K):
            for w in range(self.W):
                for l in range(self.K):
                    new_G[:,k] =  new_G[:,k] * (self.alpha[w, k, l]**self.n[w, :, l])*self.p[l]
        self.G = new_G/np.sum(new_G, axis = 1).reshape((self.I,1))
        return np.allclose(self.G, old_G)


    def E(self):
        G = np.tile(self.G, (2,1,1)).transpose((1,2,0))
        n = np.tile(self.n, (2,1,1,1)).transpose((1,2,0,3))
        alpha = np.sum(n*G, axis = 1)
        alpha = (alpha.transpose((2,0,1)) / np.sum(alpha, axis = 2)).transpose((1,2,0))
        self.alpha = alpha
        self.p = np.sum(self.G, axis = 0)/self.I


    def run(self, runs = 15, v = 2):
        converged = False
        count = 0
        while not converged and count < 50:
            count+=1
            self.E()
            converged = self.M()
            if v > 0:
                print self.p
        if v >1:
            self.show_probs()
            print self.p
            for i in zip(self.labelers.emails, list(self.alpha)):
                print i[0]
                print i[1]
        

if __name__ == '__main__':
    haiti_map = map_overlay.haiti_setup().sub_map(np.ix_(np.arange(4096/3, 4096), np.arange(4096)))
    labelers = Labelers()
    test = EM(haiti_map, labelers)
    test.run()