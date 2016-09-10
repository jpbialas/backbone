from object_model import ObjectClassifier
import analyze_results
import map_overlay
from map_overlay import MapOverlay
import matplotlib.pyplot as plt
import timeit
import numpy as np

def indcs2bools(indcs, segs):
    nsegs = np.max(segs)+1
    seg_mask = np.zeros(nsegs)
    seg_mask[indcs] = 1
    return seg_mask[segs]

def main():
    k = 10

    p = 'ObjectNoiseProgress/append/'
    map_2, map_3 = map_overlay.basic_setup([100], 50, label_name = "Jared")
    building_rando = np.loadtxt('damagelabels50/all_rooftops_random-3-3.csv').astype('int')
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

    for i in range(100):
        print 'Running with {} noise segs'.format(i*10)
        tic=timeit.default_timer()
        new_training = np.concatenate((building_rando[:i*10], real_damage))
        

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


if __name__ == '__main__':
    main()