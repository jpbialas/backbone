import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from object_model import ObjectClassifier
from px_model import PxClassifier
import analyze_results
import map_overlay
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
    n_segs = np.max(segs)+1
    building_rando = np.loadtxt('damagelabels50/all_rooftops_random-3-3.csv').astype('int')
    real_damage = np.loadtxt('damagelabels50/Jared-3-3.csv', delimiter = ',').astype('int')
    px_axis, seg_axis = np.zeros(100), np.zeros(100)
    damage_count = np.sum(indcs2bools(real_damage, segs)[segs])
    all_indcs = np.zeros(n_segs)
    for i in range(100):
        all_indcs[building_rando[:i*10]] = 1
        building_count = np.sum(all_indcs[segs])
        px_axis[i] = building_count/(damage_count+building_count)#(segs.shape[0]*segs.shape[1])#
        seg_axis[i] = i*10.0/(i*10+real_damage.shape[0])#n_segs#
    seg_sim_axis, px_sim_axis = [], []
    for i in range(50):
        noise = np.loadtxt('damagelabels50/Sim_{}-3-3.csv'.format(i), delimiter = ',').astype('int')
        noise_count = np.sum(indcs2bools(noise, segs)[segs])
        px_sim_axis.append((noise_count-damage_count)/(noise_count)) ##THIS IS WRONG!
        seg_sim_axis.append((noise.shape[0]-real_damage.shape[0])/float(noise.shape[0]))
    print seg_axis, px_axis, np.array(seg_sim_axis), np.array(px_sim_axis)
    return seg_axis, px_axis, np.array(seg_sim_axis), np.array(px_sim_axis)
    #return np.array([0, 0.01367989, 0.02699055, 0.03994674, 0.05256242, 0.06485084, 0.07682458, 0.08849558, 0.09987516, 0.11097411, 0.12180268, 0.13237064, 0.14268728, 0.15276146, 0.16260163, 0.17221584, 0.1816118, 0.19079686, 0.19977802, 0.20856202, 0.21715527, 0.22556391, 0.23379384, 0.24185068, 0.24973985, 0.25746653, 0.26503568, 0.27245207, 0.27972028, 0.28684471, 0.29382958, 0.30067895, 0.30739673, 0.31398668, 0.3204524, 0.32679739, 0.33302498, 0.33913841, 0.34514078, 0.3510351, 0.35682426, 0.36251105, 0.36809816, 0.37358818, 0.37898363, 0.38428693, 0.38950042, 0.39462636, 0.39966694, 0.40462428, 0.40950041, 0.41429732, 0.41901692, 0.42366107, 0.42823156, 0.43273013, 0.43715847, 0.4415182, 0.44581091, 0.45003814, 0.45420136, 0.45830203, 0.46234154, 0.46632124, 0.47024247, 0.47410649, 0.47791455, 0.48166786, 0.48536759, 0.48901488, 0.49261084, 0.49615653, 0.49965302, 0.50310131, 0.5065024, 0.50985724, 0.51316678, 0.51643192, 0.51965356, 0.52283256, 0.52596976, 0.52906597, 0.532122, 0.53513862, 0.53811659, 0.54105665, 0.54395952, 0.5468259, 0.54965646, 0.55245189, 0.55521283, 0.55793991, 0.56063376, 0.56329497, 0.56592414, 0.56852184, 0.57108864, 0.57362507, 0.57613169, 0.578609]), np.array([0, 0.02385659, 0.03839607, 0.05505652, 0.07674098, 0.09819899, 0.11572629, 0.12782714, 0.13711813, 0.14721884, 0.17071743, 0.18050659, 0.18579152, 0.1965988, 0.2092049, 0.22343498, 0.23323942, 0.25122325, 0.26362742, 0.27572874, 0.28618251, 0.29373743, 0.30236008, 0.31565051, 0.33840805, 0.34528749, 0.3528892, 0.36051907, 0.37051948, 0.37771164, 0.3885556, 0.39763184, 0.40465691, 0.41293849, 0.41699841, 0.42413299, 0.42925339, 0.4347889, 0.43925886, 0.4456093, 0.45609541, 0.46067423, 0.46472449, 0.47009686, 0.47312196, 0.4777752, 0.4896882, 0.49366306, 0.49750815, 0.50360604, 0.50596803, 0.50890783, 0.5143787, 0.52031638, 0.52298996, 0.52671848, 0.53111845, 0.53731, 0.54291392, 0.5452255, 0.55240776, 0.55538041, 0.55800223, 0.56074653, 0.56449691, 0.56817298, 0.5721435, 0.57421226, 0.57695664, 0.57971603, 0.58387407, 0.58671933, 0.58810999, 0.59141547, 0.59518481, 0.59896241, 0.60210489, 0.60525707, 0.60854513, 0.61170765, 0.61340956, 0.61500207, 0.61879849, 0.62060129, 0.62261337, 0.62675238, 0.63023562, 0.63174326, 0.63459213, 0.63740854, 0.63877041, 0.64055724, 0.64479332, 0.64648391, 0.64943738, 0.65160516, 0.65456459, 0.6561598, 0.65861667, 0.66138956]), np.array([0, 0.00138504, 0.00138504, 0.00276625, 0.00961538, 0.02303523, 0.0462963, 0.06485084, 0.09422111, 0.11750306, 0.15474795, 0.18438914, 0.21545158, 0.2442348, 0.26950355, 0.29244357, 0.31723485, 0.32992565, 0.34869015, 0.37195122, 0.39156118, 0.40707237, 0.4232, 0.43627834, 0.44793262, 0.45992509, 0.46789668, 0.47601744, 0.48646724, 0.49474422, 0.50548697, 0.51118644, 0.51901268, 0.52783235, 0.53573728, 0.54511041, 0.54993758, 0.55766871, 0.56487628, 0.56903766, 0.57362507, 0.57885514, 0.58251303, 0.58776444, 0.59494382, 0.59944444, 0.60275482, 0.60644105, 0.61090124, 0.61608094]), np.array([0, 0.00454937, 0.00454937, 0.00545196, 0.00981025, 0.02556343, 0.04387587, 0.05589758, 0.08115702, 0.1065527, 0.14522988, 0.17869773, 0.21482793, 0.24709819, 0.27874613, 0.31233959, 0.34483815, 0.36034667, 0.38980319, 0.42484095, 0.46206155, 0.4879343, 0.51742191, 0.55024289, 0.57195416, 0.59625249, 0.62025133, 0.64165163, 0.66526996, 0.68864647, 0.72090527, 0.7344159, 0.75864549, 0.78546052, 0.81015492, 0.83637954, 0.85394773, 0.88142027, 0.90442322, 0.92246685, 0.94572593, 0.96694294, 0.98128169, 1.00157769, 1.03420351, 1.06658137, 1.08306039, 1.09708819, 1.12031413, 1.14479249])

def tests(indcs = [0,1,2,3,4,5,6,7]):
    import matplotlib.pyplot as plt
    n = len(indcs)
    start_n = 50
    batch = 50
    total_segs = 32910.
    auc_lim = total_segs
    xaxes = xAxis()
    all_methods = [('object','', 'blue', 100, 0), ('object', '_random', 'red', 100, 0), ('object', '_dilate', 'green', 50, 2), \
                    ('px','', 'cyan', 100, 1), ('px', '_random', 'magenta', 100, 1), ('px', '_dilate', 'teal', 50, 3)]
                    #('object','_minimal', 'brown', 100, 0), ('object', '_random_minimal', 'yellow', 100, 0), \
                    #('px','_fewer', 'green', 100, 1), ('px', '_random_fewer', 'teal', 100, 1)]
    for method in all_methods:
        fn = 'ObjectNoiseProgress2/Damage_FPR_{}{}_'.format(method[0], method[1])
        ave = np.zeros(method[3])
        all_plots = np.zeros((n, method[3]))
        for i in indcs:
            next_plot = np.load(fn+'{}.npy'.format(i))
            all_plots[indcs.index(i)] = 100*next_plot
        valid = np.where(np.min(all_plots, axis = 0) > 0)[0]
        ave = np.mean(all_plots, axis = 0)[valid]
        #ave = ave/ave[0]*100
        x_axis = 100*xaxes[method[4]][valid]
        std_error = (np.std(all_plots, axis = 0)/np.sqrt(n))[valid]
        ax = plt.axes()
        plt.plot(x_axis, ave, method[2], label = '{} {}'.format(method[0], method[1]))
        plt.fill_between(x_axis, ave-std_error, ave+std_error,alpha=0.5, edgecolor=method[2], facecolor=method[2])
        plt.legend(loc = 0)
        plt.title('Average AUC over {} Runs'.format(n));plt.xlabel('% Noise'.format(int(total_segs)));plt.ylabel('AUC (%)');# plt.xlim([0, 100]), plt.ylim(70, 100)
    plt.show()


class noise():
    def __init__(self, run_num = 0, model_type = 'px', random = False, dilate = True, minimal = False):
        assert(not (random and dilate))
        self.model_type = model_type
        self.dilate = dilate
        self._setup_postfix(run_num, model_type, random, dilate, minimal)
        self.model_con = ObjectClassifier if model_type == 'object' else PxClassifier
        self.p = 'ObjectNoiseProgress2/'
        self.damage_ground = indcs2bools(test_damage, test_segs)[segs].ravel()
        self.iters = 50 if dilate else 100
        self.order = better_order(self.iters/2)
        self.damage_AUCs = np.zeros(self.iters)

    def _setup_map(self, random):
        maps = [0,0,0,0]
        self.tr, self.te = 3, 2
        maps[2:4] = list(map_overlay.basic_setup([100], 50))
        if random:
            self.building_rando = np.loadtxt('damagelabels50/non_damage_random-3-{}.csv'.format(self.tr)).astype('int')
        else:
            self.building_rando = np.loadtxt('damagelabels50/all_rooftops_random-3-{}.csv'.format(self.tr)).astype('int')
        self.real_damage = np.loadtxt('damagelabels50/Jared-3-{}.csv'.format(self.tr), delimiter = ',').astype('int')
        test_damage = np.loadtxt('damagelabels50/Jared-3-{}.csv'.format(self.te), delimiter = ',').astype('int')
        test_segs = self.maps[self.te].segmentations[50].ravel().astype('int')
        segs = self.maps[self.te].segmentations[50]

    def _setup_postfix(self, run_num, model_type, random, dilate, minimal):
        self.postfix = '_'+model_type
        if random:
            self.postfix+='_random'
        elif dilate:
            self.postfix+='_dilate'
        if minimal and model_type == 'object':
            self.postfix+='_minimal'
        elif minimal and model_type == 'px':
            self.postfix+='_fewer'
        self.postfix += '_{}'.format(run_num)

    def gen_training(self, i):
        if self.dilate:
            new_training = np.loadtxt('damagelabels50/Sim_{}-3-{}.csv'.format(i, self.tr)).astype('int')
        else:
            new_training = np.concatenate((self.building_rando[:i*10], self.real_damage))
        if self.model_type == 'px':
            return self.maps[self.tr].mask_segments_by_indx(new_training, 50, False).ravel()
        else:
            return indcs2bools(new_training,self.maps[self.tr].segmentations[50])

    def run(self):
        model = self.model_con()
        for i in range(self.iters):
            i = self.order[i]
            print 'Running with {} noise segs'.format(i*10)
            new_training = self.gen_training(i)
            pred = model.fit_and_predict(self.maps[self.tr], self.maps[self.te], labels = new_training)
            #d_roc, d_AUC, _, _, _, _ = analyze_results.ROC(self.damage_ground, pred.ravel(), '')
            FPR = analyze_results.FPR_from_FNR(self.damage_ground, pred.ravel(), TPR = .9, precision = True)
            self.damage_AUCs[i] = FPR#d_AUC
            np.save('{}Damage_Prec{}.npy'.format(self.p, self.postfix), self.damage_AUCs)
            #plt.close(d_roc)


if __name__ == '__main__':
    if int(sys.argv[1]) == -1:
        tests()
    else:
        options = [(False, False), (True, False), (False, True)]
        option = options[int(sys.argv[2])]
        assert(sys.argv[3] == 'object' or sys.argv[3] == 'px')
        n = noise(run_num = sys.argv[1], random = option[0], dilate = option[1], model_type = sys.argv[3])
        n.run()
