#import matplotlib
#matplotlib.use('Agg')
from object_model import ObjectClassifier
from px_model import PxClassifier
import analyze_results
import map_overlay
import matplotlib.pyplot as plt
import timeit
import sys
import numpy as np
from convenience_tools import *

def indcs2bools(indcs, segs):
    nsegs = np.max(segs)+1
    seg_mask = np.zeros(nsegs)
    seg_mask[indcs] = 1
    return seg_mask


def better_order(middle = 50):
    if middle == 50:
        bases = [25, 12, 6, 3] 
    elif middle == 25:
        bases = [12, 6, 3]
    test = [middle]
    last_index = 0
    for i in range(len(bases)):
        cur_index = len(test)
        for j in range(last_index, cur_index):
            #print 'here', bases[i]
            test.append(test[j]-bases[i])
            test.append(bases[i]+test[j])
        last_index = cur_index
    
    cur_index = len(test)
    for j in range(last_index, cur_index):
            test.append(test[j]-2)
            test.append(2+test[j])
            test.append(test[j]-1)
            test.append(1+test[j])
    
    test.append(1)
    if middle == 50:
        test.append(middle-1)
        test.append(middle+1)
    test.insert(0,0)
    test.insert(0,2*middle-1)
    return test


def xAxis():
    segs = np.load('processed_segments/shapefilewithfeatures003-003-50.npy').astype('int')
    building_rando = np.loadtxt('damagelabels50/all_rooftops_random-3-3.csv').astype('int')
    real_damage = np.loadtxt('damagelabels50/Jared-3-3.csv', delimiter = ',').astype('int')
    px_axis = np.zeros(100)
    seg_axis = np.zeros(100)
    pbar = custom_progress()
    damage_count = np.sum(indcs2bools(real_damage, segs))
    for i in pbar(range(100)):
        building_count = np.sum(indcs2bools(building_rando[:i*10], segs))
        px_axis[i] = building_count/float(damage_count+building_count)
        seg_axis[i] = i*10.0/(i*10+real_damage.shape[0])

    seg_sim_axis = []
    px_sim_axis = []
    for i in range(50):
        noise = np.loadtxt('damagelabels50/Sim_{}-3-3.csv'.format(i), delimiter = ',').astype('int')
        noise_count = np.sum(indcs2bools(noise, segs))
        px_sim_axis.append(1-damage_count/noise_count)
        seg_sim_axis.append(1-real_damage.shape[0]/float(noise.shape[0]))
    #print seg_sim_axis
    #print seg_sim_axis[0], seg_sim_axis[10], seg_sim_axis[20], seg_sim_axis[30], seg_sim_axis[40], seg_sim_axis[49]
    print seg_sim_axis
    return seg_axis, px_axis, np.array(seg_sim_axis), np.array(px_sim_axis)


def tests(indcs = [0,1,2,3,4,5,6,7]):
    n = len(indcs)
    start_n = 50
    batch = 50
    total_segs = 32910.
    auc_lim = total_segs
    xaxes = xAxis()
    all_methods = [('object','', 'blue', 100, 0), ('object', '_random', 'red', 100, 0), ('object', '_dilate', 'green', 50, 2)]
    for method in all_methods:
        fn = 'ObjectNoiseProgress2/Damage_AUC_{}{}_'.format(method[0], method[1])
        x_axis = xaxes[method[4]]
        ave = np.zeros(method[3])
        all_plots = np.zeros((n, method[3]))
        for i in indcs:
            next_plot = np.load(fn+'{}.npy'.format(i))
            all_plots[indcs.index(i)] = 100*next_plot
        ave = np.mean(all_plots, axis = 0)
        print all_plots
        std_error = np.std(all_plots, axis = 0)/np.sqrt(n)
        ax = plt.axes()
        print x_axis.shape, ave.shape
        plt.plot(x_axis, ave, method[2], label = '{} {}'.format(method[0], method[1]))
        plt.fill_between(x_axis, ave-std_error, ave+std_error,alpha=0.5, edgecolor=method[2], facecolor=method[2])
        plt.legend()
        plt.title('Average AUC at 95% Recall over {} Runs'.format(n));plt.xlabel('% Segments Added (out of {})'.format(int(total_segs)));plt.ylabel('AUC (%)'); #plt.xlim([0, 100]), plt.ylim(0, 45)
    print 'here'
    plt.show()




class noise():
    def __init__(self, run_num = 0, model_type = 'px', random = False, dilate = True):
        assert(not (random and dilate))
        self.random = random
        self.model_type = model_type
        self.dilate = dilate
        self.postfix = '_'+model_type
        if random:
            self.postfix+='_random'
        elif dilate:
            self.postfix+='_dilate'
        self.postfix += '_{}'.format(run_num)

        self.model_con = ObjectClassifier if model_type == 'object' else PxClassifier
        self.p = 'ObjectNoiseProgress2/'
        self.map_2, self.map_3 = map_overlay.basic_setup([100], 50)
        if random:
            self.building_rando = np.loadtxt('damagelabels50/non_damage_random-3-3.csv').astype('int')
        else:
            self.building_rando = np.loadtxt('damagelabels50/all_rooftops_random-3-3.csv').astype('int')
        self.real_damage = np.loadtxt('damagelabels50/Jared-3-3.csv', delimiter = ',').astype('int')
        test_damage = np.loadtxt('damagelabels50/Jared-3-2.csv', delimiter = ',').astype('int')
        test_segs = self.map_2.segmentations[50].ravel().astype('int')
        segs = self.map_2.segmentations[50]
        self.damage_ground = indcs2bools(test_damage, test_segs)[segs].ravel()
        self.iters = 50 if dilate else 100
        self.order = better_order(self.iters/2)
        self.damage_AUCs = np.zeros(self.iters)

    def gen_training(self, i):
        if self.dilate:
            new_training = np.loadtxt('damagelabels50/Sim_{}-3-3.csv'.format(i)).astype('int')
        else:
            new_training = np.concatenate((self.building_rando[:i*10], self.real_damage))
        if self.model_type == 'px':
            return self.map_3.mask_segments_by_indx(new_training, 50, False).ravel()
        else:
            return indcs2bools(new_training,self.map_3.segmentations[50])

    def run(self):
        model = self.model_con()
        for i in range(self.iters):
            i = self.order[i]
            print 'Running with {} noise segs'.format(i*10)
            new_training = self.gen_training(i)
            pred = model.fit_and_predict(self.map_3, self.map_2, labels = new_training)
            d_roc, d_AUC, _, _, _, _ = analyze_results.ROC(self.damage_ground, pred.ravel(), '')
            self.damage_AUCs[i] = d_AUC
            np.save('{}Damage_AUC{}.npy'.format(self.p, self.postfix), self.damage_AUCs)
            plt.close(d_roc)


if __name__ == '__main__':
    options = [(False, False), (True, False), (False, True)]
    option = options[int(sys.argv[2])]
    n = noise(run_num = sys.argv[1], random = option[0], dilate = option[1])
    n.run()
    #tests()
