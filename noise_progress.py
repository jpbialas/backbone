import matplotlib
matplotlib.use('Agg')
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

def plot(title, fn, p, cat, xaxis):
    plt.figure(title)
    data = np.loadtxt(p+cat[0]+fn+'.csv', delimiter = ',')
    axis = xaxis[cat[2]]
    if cat[2] < 2:
        data = np.lib.pad(data, (0, 100-data.shape[0]), 'constant', constant_values = (0,data[-1]))
    else:
        data = np.lib.pad(data, (0, 50-data.shape[0]), 'constant', constant_values = (0,data[-1]))
    if cat[2] != 2:
        indcs = np.where(data != 0)[0]
        data = data[indcs]
        axis = xaxis[cat[2]][indcs]

    plt.plot(axis*100, data, cat[3], label = cat[1])
    plt.legend(loc = 0)
    plt.title(title)
    plt.xlim([0, 60])
    plt.xlabel('Noise (%)')
    plt.ylabel('AUC')

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


def visualize():
    axes = xAxis()
    print axes[1][60]
    p = 'ObjectNoiseProgress/'
    cats = [('randomAppend_/', 'Obj Random Noise', 0, 'r--'), ('randomAppend_px_/', 'Px Random Noise', 1, 'b--')
             #,('append_/', 'Obj Building Noise', 0, 'r-'),('append_px_/', 'Px Building Noise', 1, 'b-')
              ,('sim_seg/', 'Obj Geospatial Noise', 2, 'r-'), ('sim_px/', 'Px Geospatial Noise', 3, 'b-')
            ]
              #('segment_no_e/', 'Object FewFeatures', 0), ('segment_basic/', 'Object Nothing', 0),
              #('seg_over/', 'Seg Oversampling', 0), ('seg_over5/', 'Seg Oversampling MORE', 0), ('px_fewer/', 'Pixel Reduced Sample Size', 1),
              #('px_fewer_random/','Px fewer random',1)]
    
    
    for cat in cats:
        plot('Effect of Noise on Rubble Classification', 'Damage_AUC', p, cat, axes)
        #plot('Building AUCs', 'Building_AUC', p, cat, axes)
        #plot('Damage Percents', 'Damage_percents', p, cat, xaxis)
        #plot('Building Percents', 'Building_percents', p, cat, xaxis)
        #plot('Damage Threshs', 'Damage_threshs', p, cat, xaxis)
        #plot('Building Threshs', 'Building_threshs', p, cat,xaxis)



        #d_thresh = np.loadtxt(p+'Damage_threshs.csv', delimiter = ',')
        #b_thresh = np.loadtxt(p+'Building_threshs.csv', delimiter = ',')
        #d_p = np.loadtxt(p+'Damage_percents.csv', delimiter = ',')
        #b_p = np.loadtxt(p+'Building_percents.csv', delimiter = ',')
    
    plt.show()




class noise():
    def __init__(self, run_num = 0, model_type = 'object', random = False, dilate = True):
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
            return self.map_2.mask_segments_by_indx(new_training, 50, False).ravel()
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
