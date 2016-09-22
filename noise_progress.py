from object_model import ObjectClassifier
from px_model import PxClassifier
import analyze_results
import map_overlay
from map_overlay import MapOverlay
import matplotlib.pyplot as plt
import timeit
import numpy as np
from convenience_tools import *

def indcs2bools(indcs, segs):
    nsegs = np.max(segs)+1
    seg_mask = np.zeros(nsegs)
    seg_mask[indcs] = 1
    return seg_mask[segs]

def plot(title, fn, p, cat, xaxis):
    plt.figure(title)
    data = np.loadtxt(p+cat[0]+fn+'.csv', delimiter = ',')
    axis = xaxis[cat[2]]
    if cat[2] != 2:
        data = np.lib.pad(data, (0, 100-data.shape[0]), 'constant', constant_values = (0,data[-1]))
        indcs = np.where(data != 0)[0]
        data = data[indcs]
        axis = xaxis[cat[2]][indcs]


    plt.plot(axis, data, label = cat[1])
    plt.legend(loc = 0)
    plt.title(title)
    plt.xlabel('Percent Noise')
    plt.ylabel('Percent')

def better_order():
    bases = [25, 12, 6, 3] 
    test = [50]
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
    test.append(49)
    test.append(51)
    test.insert(0,0)
    test.insert(0,99)
    return test

'''def calc_percents():
    seg_percents = []
    px_percents = []
    segs = np.load('processed_segments/shapefilwithfeatures003-003-50.npy')
    damage = np.loadtxt('damagelabels50/Jared-3-3.csv', delimiter = ',')
    damage_px = damage[segs]
    damage_count = np.sum(indcs2bools(real_damage, segs))
    for i in range(50):
        noise =  np.loadtxt('damagelabels50/Sim_{}-3-3.csv'.format(i), delimiter = ',')
        
        seg_percents.append(np.setdiff1d(noise, damage).shape[0]/float(np.setdiff1d(noise, damage).shape[0] + damage.shape[0]))
    print seg_percents
    return seg_percents'''



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
    return seg_axis, px_axis, np.array([0,0.15474794841735054, 0.39156118143459917, 0.5054869684499315, 0.5736250739207569])


def visualize():
    axes = xAxis()
    print axes[1][60]
    p = 'ObjectNoiseProgress/'
    cats = [('append_/', 'Object NotRandom', 0),('append_px_/', 'Pixel NotRandom', 1),
             ('randomAppend_/', 'Object Random', 0), ('randomAppend_px_/', 'Pixel Random', 1),
              #('segment_no_e/', 'Object FewFeatures', 0), ('segment_basic/', 'Object Nothing', 0),
              #('seg_over/', 'Seg Oversampling', 0), ('seg_over5/', 'Seg Oversampling MORE', 0), ('px_fewer/', 'Pixel Reduced Sample Size', 1),
              #('px_fewer_random/','Px fewer random',1)]
              ('simulated2/', 'Simulated', 2), ('sim_px/', 'Simulated Pixel', 2)]
    
    
    for cat in cats:
        plot('Damage AUCs', 'Damage_AUC', p, cat, axes)
        plot('Building AUCs', 'Building_AUC', p, cat, axes)
        #plot('Damage Percents', 'Damage_percents', p, cat, xaxis)
        #plot('Building Percents', 'Building_percents', p, cat, xaxis)
        #plot('Damage Threshs', 'Damage_threshs', p, cat, xaxis)
        #plot('Building Threshs', 'Building_threshs', p, cat,xaxis)



        #d_thresh = np.loadtxt(p+'Damage_threshs.csv', delimiter = ',')
        #b_thresh = np.loadtxt(p+'Building_threshs.csv', delimiter = ',')
        #d_p = np.loadtxt(p+'Damage_percents.csv', delimiter = ',')
        #b_p = np.loadtxt(p+'Building_percents.csv', delimiter = ',')
    
    plt.show()


def main_dilate_px():
    k = 1

    p = 'ObjectNoiseProgress/sim_px/'
    map_2, map_3 = map_overlay.basic_setup([100], 50, label_name = "Jared")
    building_rando = np.loadtxt('damagelabels50/non_damage_random-3-3.csv').astype('int')
    test_damage = np.loadtxt('damagelabels50/Jared-3-2.csv', delimiter = ',').astype('int')
    test_buildings = np.loadtxt('damagelabels50/all_buildings-3-2.csv', delimiter = ',').astype('int')
    test_segs = map_2.segmentations[50][1].ravel().astype('int')
    damage_ground = indcs2bools(test_damage, test_segs)
    building_ground = indcs2bools(test_buildings, test_segs)
    damage_AUCs = []
    building_AUCs = []
    damage_threshs = []
    building_threshs = []
    damage_perecents = []
    building_percents = []
    #model = ObjectClassifier(verbose = 0)
    #X, _ = model._get_X_y(map_3, "Jared", custom_labels = real_damage)
    model = PxClassifier(85, -1)
    X3,_ = model.gen_features(map_3)
    X2,_ = model.gen_features(map_2)
    for i in range(1,5):
        print 'Running with {} noise segs'.format(i*10)
        tic=timeit.default_timer()
        new_training = np.loadtxt('damagelabels50/Sim_{}-3-3.csv'.format(i)).astype('int')
        
        map_3.new_seg_mask(new_training, 50, 'damage')
        model = PxClassifier(85, -1, verbose = 0)
        model.fit(map_3, custom_data = X3)
        prediction = model.predict_proba(map_2, load = False, custom_data = X2)
        #model = ObjectClassifier()
        #model.fit(map_3, custom_labels = new_training)
        #prediction = model.predict_proba(map_2)
        for j in range(k-1):
            model = PxClassifier(85, -1, verbose = 0)
            model.fit(map_3, custom_data = X3)
            prediction += model.predict_proba(map_2, load = False, custom_data = X2)
            #model = ObjectClassifier(verbose = 0)
            #model.fit(map_3, custom_labels = new_training, custom_data = X)
            #prediction += model.predict_proba(map_2)
        prediction /= k
        prediction = prediction.reshape((map_2.rows, map_2.cols))

        d_roc, d_AUC, d_thresh, d_FPRs, d_TPRs, d_Threshs = analyze_results.ROC(map_2, damage_ground, prediction, model.test_name, save = False)
        b_roc, b_AUC, b_thresh, b_FPRs, b_TPRs, b_Threshs = analyze_results.ROC(map_2, building_ground, prediction, model.test_name+' building', save = False)
        d_perc = analyze_results.average_class_prob(map_2, damage_ground, prediction, model.test_name)
        b_perc = analyze_results.average_class_prob(map_2, building_ground, prediction, model.test_name)
        damage_AUCs.append(d_AUC)
        building_AUCs.append(b_AUC)
        damage_threshs.append(d_thresh)
        building_threshs.append(b_thresh)
        damage_perecents.append(d_perc)
        building_percents.append(b_perc)
        pred_fig = plt.figure()
        plt.imshow(prediction, cmap = 'seismic', norm = plt.Normalize(0,1))

        pred_fig.savefig(p+'Prediction_{}.png'.format(i), format = 'png')
        d_roc.savefig(p+'Damage_ROC_{}.png'.format(i), format = 'png')
        b_roc.savefig(p+'Building_ROC_{}.png'.format(i), format = 'png')
        np.savetxt(p+'Prediction_{}.csv'.format(i), prediction, delimiter = ',')

        np.savetxt(p+'Building_ROC_FPRs_{}.csv'.format(i), b_FPRs, delimiter = ',')
        np.savetxt(p+'Damage_ROC_FPRs_{}.csv'.format(i), d_FPRs, delimiter = ',')
        np.savetxt(p+'Building_ROC_TPRs_{}.csv'.format(i), b_TPRs, delimiter = ',')
        np.savetxt(p+'Damage_ROC_TPRs_{}.csv'.format(i), d_TPRs, delimiter = ',')
        np.savetxt(p+'Building_ROC_Threshs_{}.csv'.format(i), b_Threshs, delimiter = ',')
        np.savetxt(p+'Damage_ROC_FPRs_{}.csv'.format(i), d_Threshs, delimiter = ',')

        np.savetxt(p+'Damage_AUC.csv', damage_AUCs, delimiter = ',')
        np.savetxt(p+'Building_AUC.csv', building_AUCs, delimiter = ',')
        np.savetxt(p+'Damage_threshs.csv', damage_threshs, delimiter = ',')
        np.savetxt(p+'Building_threshs.csv', building_threshs, delimiter = ',')
        np.savetxt(p+'Damage_percents.csv', damage_perecents, delimiter = ',')
        np.savetxt(p+'Building_percents.csv', building_percents, delimiter = ',')

        plt.close(pred_fig)
        plt.close(d_roc)
        plt.close(b_roc)
        toc=timeit.default_timer()
        print 'That run took {} seconds!'.format(toc-tic)


def main_dilate():
    k=5
    p = 'ObjectNoiseProgress/sim_seg/'
    map_2, map_3 = map_overlay.basic_setup([100], 50, label_name = "Jared")
    real_damage = np.loadtxt('damagelabels50/Jared-3-3.csv', delimiter = ',').astype('int')
    test_damage = np.loadtxt('damagelabels50/Jared-3-2.csv', delimiter = ',').astype('int')
    test_buildings = np.loadtxt('damagelabels50/all_buildings-3-2.csv', delimiter = ',').astype('int')

    test_segs = map_2.segmentations[50][1].ravel().astype('int')
    damage_ground = indcs2bools(test_damage, test_segs)
    building_ground = indcs2bools(test_buildings, test_segs)
    damage_AUCs = []
    building_AUCs = []
    damage_threshs = []
    building_threshs = []
    damage_perecents = []
    building_percents = []
    model = ObjectClassifier(verbose = 0)
    X, _ = model._get_X_y(map_3, "Jared", custom_labels = real_damage)
    for i in range(50):
        print 'Running noise level {} noise segs'.format(i/10.0)
        new_training = np.loadtxt('damagelabels50/Sim_{}-3-3.csv'.format(i)).astype('int')
        tic=timeit.default_timer()
        model = ObjectClassifier()
        model.fit(map_3, custom_labels = new_training)
        prediction = model.predict_proba(map_2)
        for j in range(k-1):
            model = ObjectClassifier(verbose = 0)
            model.fit(map_3, custom_labels = new_training, custom_data = X)
            prediction += model.predict_proba(map_2)
        prediction /= k

        d_roc, d_AUC, d_thresh, d_FPRs, d_TPRs, d_Threshs = analyze_results.ROC(map_2, damage_ground, prediction, model.test_name, save = False)
        b_roc, b_AUC, b_thresh, b_FPRs, b_TPRs, b_Threshs = analyze_results.ROC(map_2, building_ground, prediction, model.test_name+' building', save = False)
        d_perc = analyze_results.average_class_prob(map_2, damage_ground, prediction, model.test_name)
        b_perc = analyze_results.average_class_prob(map_2, building_ground, prediction, model.test_name)
        damage_AUCs.append(d_AUC)
        building_AUCs.append(b_AUC)
        damage_threshs.append(d_thresh)
        building_threshs.append(b_thresh)
        damage_perecents.append(d_perc)
        building_percents.append(b_perc)
        pred_fig = plt.figure()
        plt.imshow(prediction, cmap = 'seismic', norm = plt.Normalize(0,1))

        pred_fig.savefig(p+'Prediction_{}.png'.format(i), format = 'png')
        d_roc.savefig(p+'Damage_ROC_{}.png'.format(i), format = 'png')
        b_roc.savefig(p+'Building_ROC_{}.png'.format(i), format = 'png')
        np.savetxt(p+'Prediction_{}.csv'.format(i), prediction, delimiter = ',')

        np.savetxt(p+'Building_ROC_FPRs_{}.csv'.format(i), b_FPRs, delimiter = ',')
        np.savetxt(p+'Damage_ROC_FPRs_{}.csv'.format(i), d_FPRs, delimiter = ',')
        np.savetxt(p+'Building_ROC_TPRs_{}.csv'.format(i), b_TPRs, delimiter = ',')
        np.savetxt(p+'Damage_ROC_TPRs_{}.csv'.format(i), d_TPRs, delimiter = ',')
        np.savetxt(p+'Building_ROC_Threshs_{}.csv'.format(i), b_Threshs, delimiter = ',')
        np.savetxt(p+'Damage_ROC_FPRs_{}.csv'.format(i), d_Threshs, delimiter = ',')

        np.savetxt(p+'Damage_AUC.csv', damage_AUCs, delimiter = ',')
        np.savetxt(p+'Building_AUC.csv', building_AUCs, delimiter = ',')
        np.savetxt(p+'Damage_threshs.csv', damage_threshs, delimiter = ',')
        np.savetxt(p+'Building_threshs.csv', building_threshs, delimiter = ',')
        np.savetxt(p+'Damage_percents.csv', damage_perecents, delimiter = ',')
        np.savetxt(p+'Building_percents.csv', building_percents, delimiter = ',')

        plt.close(pred_fig)
        plt.close(d_roc)
        plt.close(b_roc)
        toc=timeit.default_timer()
        print 'That run took {} seconds!'.format(toc-tic)

def main_segs():
    k = 3
    order = better_order()

    p = 'ObjectNoiseProgress/seg_over5/'
    map_2, map_3 = map_overlay.basic_setup([100], 50, label_name = "Jared")
    building_rando = np.loadtxt('damagelabels50/all_rooftops_random-3-3.csv').astype('int')
    real_damage = np.loadtxt('damagelabels50/Jared-3-3.csv', delimiter = ',').astype('int')
    test_damage = np.loadtxt('damagelabels50/Jared-3-2.csv', delimiter = ',').astype('int')
    test_buildings = np.loadtxt('damagelabels50/all_buildings-3-2.csv', delimiter = ',').astype('int')
    test_segs = map_2.segmentations[50][1].ravel().astype('int')
    damage_ground = indcs2bools(test_damage, test_segs)
    building_ground = indcs2bools(test_buildings, test_segs)
    damage_AUCs = np.zeros(100)
    building_AUCs = np.zeros(100)
    damage_threshs = np.zeros(100)
    building_threshs = np.zeros(100)
    damage_perecents = np.zeros(100)
    building_percents = np.zeros(100)
    model = ObjectClassifier(verbose = 0)
    X, _ = model._get_X_y(map_3, "Jared", custom_labels = real_damage)

    for i in range(100):
        i = order[i]
        print 'Running with {} noise segs'.format(i*10)
        tic=timeit.default_timer()
        new_training = np.concatenate((building_rando[:i*10], real_damage))
        

        model = ObjectClassifier()
        model.fit(map_3, custom_labels = new_training)
        prediction = model.predict_proba(map_2)
        for j in range(k-1):
            print ('here')
            model = ObjectClassifier(verbose = 0)
            model.fit(map_3, custom_labels = new_training, custom_data = X)
            prediction += model.predict_proba(map_2)
        prediction /= k

        d_roc, d_AUC, d_thresh, d_FPRs, d_TPRs, d_Threshs = analyze_results.ROC(map_2, damage_ground, prediction, model.test_name, save = False)
        b_roc, b_AUC, b_thresh, b_FPRs, b_TPRs, b_Threshs = analyze_results.ROC(map_2, building_ground, prediction, model.test_name+' building', save = False)
        d_perc = analyze_results.average_class_prob(map_2, damage_ground, prediction, model.test_name)
        b_perc = analyze_results.average_class_prob(map_2, building_ground, prediction, model.test_name)
        damage_AUCs[i] = d_AUC
        building_AUCs[i] = b_AUC
        damage_threshs[i] = d_thresh
        building_threshs[i] = b_thresh
        damage_perecents[i] = d_perc
        building_percents[i] = b_perc
        pred_fig = plt.figure()
        plt.imshow(prediction, cmap = 'seismic', norm = plt.Normalize(0,1))

        pred_fig.savefig(p+'Prediction_{}.png'.format(i), format = 'png')
        d_roc.savefig(p+'Damage_ROC_{}.png'.format(i), format = 'png')
        b_roc.savefig(p+'Building_ROC_{}.png'.format(i), format = 'png')
        np.savetxt(p+'Prediction_{}.csv'.format(i), prediction, delimiter = ',')

        np.savetxt(p+'Building_ROC_FPRs_{}.csv'.format(i), b_FPRs, delimiter = ',')
        np.savetxt(p+'Damage_ROC_FPRs_{}.csv'.format(i), d_FPRs, delimiter = ',')
        np.savetxt(p+'Building_ROC_TPRs_{}.csv'.format(i), b_TPRs, delimiter = ',')
        np.savetxt(p+'Damage_ROC_TPRs_{}.csv'.format(i), d_TPRs, delimiter = ',')
        np.savetxt(p+'Building_ROC_Threshs_{}.csv'.format(i), b_Threshs, delimiter = ',')
        np.savetxt(p+'Damage_ROC_FPRs_{}.csv'.format(i), d_Threshs, delimiter = ',')

        np.savetxt(p+'Damage_AUC.csv', damage_AUCs, delimiter = ',')
        np.savetxt(p+'Building_AUC.csv', building_AUCs, delimiter = ',')
        np.savetxt(p+'Damage_threshs.csv', damage_threshs, delimiter = ',')
        np.savetxt(p+'Building_threshs.csv', building_threshs, delimiter = ',')
        np.savetxt(p+'Damage_percents.csv', damage_perecents, delimiter = ',')
        np.savetxt(p+'Building_percents.csv', building_percents, delimiter = ',')

        plt.close(pred_fig)
        plt.close(d_roc)
        plt.close(b_roc)
        toc=timeit.default_timer()
        print 'That run took {} seconds!'.format(toc-tic)

def main():
    k = 1
    order = better_order()
    print order

    p = 'ObjectNoiseProgress/px_fewer_random/'
    map_2, map_3 = map_overlay.basic_setup([100], 50, label_name = "Jared")
    building_rando = np.loadtxt('damagelabels50/non_damage_random-3-3.csv').astype('int')
    real_damage = np.loadtxt('damagelabels50/Jared-3-3.csv', delimiter = ',').astype('int')
    test_damage = np.loadtxt('damagelabels50/Jared-3-2.csv', delimiter = ',').astype('int')
    test_buildings = np.loadtxt('damagelabels50/all_buildings-3-2.csv', delimiter = ',').astype('int')
    test_segs = map_2.segmentations[50][1].ravel().astype('int')
    damage_ground = indcs2bools(test_damage, test_segs)
    building_ground = indcs2bools(test_buildings, test_segs)
    damage_AUCs = np.zeros(100)
    building_AUCs = np.zeros(100)
    damage_threshs = np.zeros(100)
    building_threshs = np.zeros(100)
    damage_perecents = np.zeros(100)
    building_percents = np.zeros(100)
    #model = ObjectClassifier(verbose = 0)
    #X, _ = model._get_X_y(map_3, "Jared", custom_labels = real_damage)
    model = PxClassifier(85, -1)
    X3,_ = model.gen_features(map_3)
    X2,_ = model.gen_features(map_2)
    for i in range(100):
        i = order[i]
        print 'Running with {} noise segs'.format(i*10)
        tic=timeit.default_timer()
        new_training = np.concatenate((building_rando[:i*10], real_damage))
        
        map_3.new_seg_mask(new_training, 50, 'damage')
        model = PxClassifier(85, -1, n_segs = 1442+10*i, verbose = 0)
        print 'here', 1442+10*i
        model.fit(map_3, custom_data = X3)
        prediction = model.predict_proba(map_2, load = False, custom_data = X2)
        #model = ObjectClassifier()
        #model.fit(map_3, custom_labels = new_training)
        #prediction = model.predict_proba(map_2)
        for j in range(k-1):
            model = PxClassifier(85, -1, n_segs = 1442+10*i, verbose = 0)
            model.fit(map_3, custom_data = X3)
            prediction += model.predict_proba(map_2, load = False, custom_data = X2)
            #model = ObjectClassifier(verbose = 0)
            #model.fit(map_3, custom_labels = new_training, custom_data = X)
            #prediction += model.predict_proba(map_2)
        prediction /= k
        prediction = prediction.reshape((map_2.rows, map_2.cols))

        d_roc, d_AUC, d_thresh, d_FPRs, d_TPRs, d_Threshs = analyze_results.ROC(map_2, damage_ground, prediction, model.test_name, save = False)
        b_roc, b_AUC, b_thresh, b_FPRs, b_TPRs, b_Threshs = analyze_results.ROC(map_2, building_ground, prediction, model.test_name+' building', save = False)
        d_perc = analyze_results.average_class_prob(map_2, damage_ground, prediction, model.test_name)
        b_perc = analyze_results.average_class_prob(map_2, building_ground, prediction, model.test_name)
        damage_AUCs[i] = d_AUC
        building_AUCs[i] = b_AUC
        damage_threshs[i] = d_thresh
        building_threshs[i] = b_thresh
        damage_perecents[i] = d_perc
        building_percents[i] = b_perc
        pred_fig = plt.figure()
        plt.imshow(prediction, cmap = 'seismic', norm = plt.Normalize(0,1))

        pred_fig.savefig(p+'Prediction_{}.png'.format(i), format = 'png')
        d_roc.savefig(p+'Damage_ROC_{}.png'.format(i), format = 'png')
        b_roc.savefig(p+'Building_ROC_{}.png'.format(i), format = 'png')
        np.savetxt(p+'Prediction_{}.csv'.format(i), prediction, delimiter = ',')

        np.savetxt(p+'Building_ROC_FPRs_{}.csv'.format(i), b_FPRs, delimiter = ',')
        np.savetxt(p+'Damage_ROC_FPRs_{}.csv'.format(i), d_FPRs, delimiter = ',')
        np.savetxt(p+'Building_ROC_TPRs_{}.csv'.format(i), b_TPRs, delimiter = ',')
        np.savetxt(p+'Damage_ROC_TPRs_{}.csv'.format(i), d_TPRs, delimiter = ',')
        np.savetxt(p+'Building_ROC_Threshs_{}.csv'.format(i), b_Threshs, delimiter = ',')
        np.savetxt(p+'Damage_ROC_FPRs_{}.csv'.format(i), d_Threshs, delimiter = ',')

        np.savetxt(p+'Damage_AUC.csv', damage_AUCs, delimiter = ',')
        np.savetxt(p+'Building_AUC.csv', building_AUCs, delimiter = ',')
        np.savetxt(p+'Damage_threshs.csv', damage_threshs, delimiter = ',')
        np.savetxt(p+'Building_threshs.csv', building_threshs, delimiter = ',')
        np.savetxt(p+'Damage_percents.csv', damage_perecents, delimiter = ',')
        np.savetxt(p+'Building_percents.csv', building_percents, delimiter = ',')

        plt.close(pred_fig)
        plt.close(d_roc)
        plt.close(b_roc)
        toc=timeit.default_timer()
        print 'That run took {} seconds!'.format(toc-tic)


if __name__ == '__main__':
    #main()
    #test()
    #main_segs()
    main_dilate()
    #main_dilate_px()
    #visualize()
