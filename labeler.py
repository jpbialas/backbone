import numpy as np
import cv2
import csv
import map_overlay
import matplotlib.pyplot as plt
from object_model import ObjectClassifier
from scipy.stats import t
from sklearn.externals.joblib import Parallel, delayed

def model_vote_help(train_map, labels, majority_vote, train_indcs, new_train):
    uniques = train_map.unique_segs(20)
    new_model = ObjectClassifier()
    indcs = np.array(train_indcs)
    new_model.fit(train_map, (labels == majority_vote)[uniques], indcs)
    probs = new_model.predict_proba_segs(train_map, new_train)
    good_indices = np.where((probs>0.8) & (labels[uniques[new_train]]>-1))
    return good_indices

class Labelers:
    def __init__(self, n = 97710, basic_setup = True):
        self.user_map = {}
        self.labels = np.array([])
        self.rewards = np.array([]) #nx3 matrix for n labelers, storing sum(x), sum(x^2), n_rewards
        self.label_models = []
        self.train_indcs = np.array([])
        self.n = n
        if basic_setup:
            self.basic_setup()

    

    @property
    def emails(self):
        all_emails = np.empty(len(self.user_map)).astype('string')
        for i in self.user_map:
            all_emails[self.user_map[i]] = i
        return all_emails

    def add_empty(self):
        self.train_indcs = np.hstack((self.train_indcs, np.array([0]).astype('object')))
        self.train_indcs[-1] = []


    def _unique_emails(self, fn):
        with open(fn, mode='r') as infile:
            reader = csv.reader(infile)
            self.user_map = {}
            for row in reader:
                email = row[0][:row[0].find('_')] 
                if not email in self.user_map:
                    self.user_map[email] = len(self.user_map)
                    self.add_empty()



    def basic_setup(self):
        fn = 'damagelabels20/labels4.csv'
        indices = [np.arange(0,30354), np.arange(30354,67105),np.arange(67105, 97710)]
        self._unique_emails(fn)
        self.rewards = np.tile(np.array([1,1,2]), (len(self.user_map),1))
        self.labels = np.ones((len(self.user_map), self.n))*-1
        with open(fn, mode='r') as infile:
            reader = csv.reader(infile)
            for row in reader:
                #Find label number, label those indices as all zeros then apply the label indices
                email = row[0][:row[0].find('_')]
                img_num = int(row[2])
                labels = np.array(map(int, row[3][1:-1].split(',')))
                self.labels[self.user_map[email]][indices[img_num]] = 0
                self.labels[self.user_map[email]][labels] = 1

    def add_labels(self, email, boolean_map):
        if email in self.user_map:
            console.log("ERROR: User already in system")
        elif boolean_map.shape[0]!=self.n:
            console.log("Error: new labels must be same shape as others")
        elif self.labels.shape[0]>0:
            self.labels = np.vstack((self.labels, boolean_map))
            self.user_map[email] = len(self.user_map)
            self.rewards = np.vstack((self.rewards,np.array([1,1,2])))
            self.add_empty()
        else:
            self.labels = np.array(boolean_map).reshape((1,self.n))
            self.user_map[email] = 0
            self.rewards = np.array([1,1,2]).reshape((1,3))
            self.add_empty()

    def add_label_index(self, email, indices):
        new_booleans = np.zeros(self.n)
        new_booleans[indices] = 1
        self.add_labels(new_booleans)

    def clear_rewards(self):
        self.rewards = np.tile(np.array([1,1,2]), (len(self.user_map),1))

    def UI(self):
        confidence_level = .95
        alpha = 1-confidence_level
        critical_prob = 1-alpha/2
        n = self.rewards[:,2].astype('float')
        mean = self.rewards[:,0]/n
        std = np.sqrt(self.rewards[:,1]/n-mean**2)
        t_crit = t.ppf(critical_prob, df = n-1)
        UI = mean + t_crit*std/np.sqrt(n)
        return UI

    def top_labelers(self, error):
        '''
            returns indices of top labelers within error of the max UI
        '''
        all_ui = self.UI()        
        max_ui = np.max(all_ui)
        threshold = error*max_ui
        return np.where(all_ui>threshold)[0]

    def model_start(self, train_map, new_train):
        uniques = train_map.unique_segs(20)
        for i in range(len(self.labels)):
            good_indices = new_train[np.where(self.labels[i][uniques[new_train]]>-1)]
            self.train_indcs[i]+=list(good_indices)


    def model_vote(self, train_map, new_train):
        uniques = train_map.unique_segs(20)
        total_vote = np.zeros(len(new_train))
        count = np.zeros(len(new_train))
        majority_vote = self.majority_vote()
        results = Parallel(n_jobs = -1, verbose=1)(delayed(model_vote_help)(train_map, self.labels[i], majority_vote[i], self.train_indcs[i], new_train) for i in range(len(self.labels)))
        for j in range(len(self.labels)):
            good_indices = results[j]
            self.train_indcs[j]+=list(new_train[good_indices])
            total_vote[good_indices] += self.labels[j][uniques[new_train[good_indices]]]
            count[good_indices] += 1
        return total_vote/count.astype('float')>=0.5
        

    def model_vote_serial(self, train_map, new_train):
        #1 Create total_vote array of length label_indices
        #2 Create count array of length label_indices
        #For each labeler
            #1 Train model on all current indices
            #3 Predict labeler probs for new indices (probs)
            #4 Usable indices are those whose prob is above the threshold (probs>.75)
            #5 Add the indices for those valid probs to current indices
            #6 Add those index labels to the total_vote and count arrays
        #Return total_vote/count.astype('float')>0.5

        uniques = train_map.unique_segs(20)
        total_vote = np.zeros(len(new_train))
        count = np.zeros(len(new_train))
        for i in range(len(self.labels)):
            new_model = ObjectClassifier()
            indcs = np.array(self.train_indcs[i])
            labels = (self.labels == self.majority_vote())[i][uniques]
            new_model.fit(train_map, labels, indcs)
            probs = new_model.predict_proba_segs(train_map, new_train)
            good_indices = np.where((probs>0.75) & (self.labels[i][uniques[new_train]]>-1))
            self.train_indcs[i]+=list(new_train[good_indices])
            total_vote[good_indices] += self.labels[i][uniques[new_train[good_indices]]]
            count[good_indices] += 1
        print count
        return total_vote/count.astype('float')>0.5


    def donmez_vote(self, label_indices, error, update = True):
        top_indices = self.top_labelers(error)
        top_voters = self.labels[np.ix_(top_indices, label_indices)]
        majority_vote = self.majority_vote(label_indices, top_indices)#np.sum(top_voters, axis = 0)/float(top_voters.shape[0])>=0.5
        if update:
            new_rewards = (top_voters == majority_vote)
            bonus = ((new_rewards+top_voters)>1)*14 #Bonus for getting 1 correct
            new_rewards = new_rewards+bonus
            self.rewards[top_indices,0] += np.sum(new_rewards, axis = 1)
            self.rewards[top_indices,1] += np.sum(new_rewards**2, axis = 1)
            bcount = np.bincount(np.where(top_voters>=0)[0])
            self.rewards[top_indices[:bcount.shape[0]],2] += bcount
            # NEED TO CHANGE THIS SO COUNT DOESNT INCREASE FOR -1s
            # subtract np.sum(np.where(top_voters<0)[0]) from count
            # real: self.rewards[...] += np.bincount(np.where(top_voters>=0)[0])
        return majority_vote


    def majority_vote(self, label_indices = None, labeler_indices = None):
        return self.probability(label_indices, labeler_indices)>=0.5


    def probability(self, label_indices = None, labeler_indices = None):
        if label_indices is None:
            label_indices = np.arange(self.n)
        if labeler_indices is None:
            labeler_indices = np.arange(self.labels.shape[0])
        indcs = np.ix_(labeler_indices, label_indices)
        return np.sum(self.labels[indcs]>0, axis = 0)/np.sum(self.labels[indcs]>=0, axis = 0).astype('float')


    def individual_vote(self, label_indices):
        pass

    def labeler(self, email):
        indx = self.user_map[email]
        return self.labels[indx]

    def show_img(self, img, title):

        fig = plt.figure(title)
        fig.subplots_adjust(bottom=0, left = 0, right = 1, top = 1, wspace = 0, hspace = 0)
        plt.imshow(img)
        plt.title(title), plt.xticks([]), plt.yticks([])


    def show_majority_vote(self, disp_map, level = 20):
        img = disp_map.mask_segments(self.majority_vote(np.arange(self.n)), level, with_img = True, opacity = .2)
        self.show_img(img, 'Majority Vote')
        return img


    def show_prob_vote(self, disp_map, level = 20):
        segs = disp_map.segmentations[level]
        probs = self.probability()
        img = disp_map.mask_helper(disp_map.img, probs[segs], opacity = .8)
        self.show_img(img, 'Probability Vote')
        return img


    def show_labeler(self, email, disp_map, level = 20):
        img = disp_map.mask_segments(self.labeler(email)>0, level, with_img = True, opacity = .4)
        self.show_img(img, email)
        return img


    def get_FPR_TPR(self, label_indices):
        probs = self.probability()
        TPRs = np.zeros(self.labels.shape[0])
        FPRs = np.zeros(self.labels.shape[0])
        for i in range(self.labels.shape[0]):
            labels = self.labels[i, label_indices]
            lab_probs = probs[label_indices]
            queriable = np.where(labels>=0)[0]
            labels = labels[queriable]
            lab_probs = lab_probs[queriable]
            TPR = np.sum(labels*lab_probs)/np.sum(lab_probs)
            FPR = np.sum(labels*(1-lab_probs))/np.sum(1-lab_probs)
            TPRs[i] = TPR
            FPRs[i] = FPR
        return FPRs, TPRs


    def show_FPR_TPR(self):
        relevant_labels = np.arange(30354,97710).astype('int')
        FPR, TPR = self.get_FPR_TPR(relevant_labels)
        fig, ax = plt.subplots()
        ax.scatter(FPR, TPR)
        names = self.emails
        ax.plot(np.array([0,1]), np.array([0,1]))
        for i in range(FPR.shape[0]):
            ax.annotate(names[i], (FPR[i], TPR[i]))
        print FPR.shape
        #ax.annotate('Under Labeler', (.002, .07), bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5))
        #ax.annotate('Over Labeler', (.092, .9), bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5))
        #ax.annotate('Good Labeler', (.002, .9), bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5))
        #ax.annotate('Bad Labeler', (.092, .07), bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5))
        plt.title('Label Comparison'), plt.xlabel('FPR'), plt.ylabel('TPR'), plt.ylim([0,1]), plt.xlim([0,1])



def test():
    '''label_list = Labelers(4, False)
    label_list.add_labels('jared', np.array([0,0,1,0]))
    print label_list.rewards
    print 'UI', label_list.UI()
    print label_list.emails
    print label_list.donmez_vote(np.array([0,1,2,3]), .5, True)
    print label_list.rewards
    print label_list.UI()
    label_list.clear_rewards()
    print label_list.UI()
    label_list.add_labels('joe', np.array([0,1,1,0]))
    label_list.add_labels('luke', np.array([0,0,0,0]))
    print label_list.donmez_vote(np.arange(3), .5, True) == np.array([0,0,1])
    print label_list.UI()'''

    haiti_map = map_overlay.haiti_setup()
    labelers = Labelers()
    labelers.show_FPR_TPR()
    labelers.show_labeler('kmkobosk@mtu.edu', haiti_map)
    #labelers.show_labeler('masexaue@mtu.edu', haiti_map)
    #labelers.show_labeler('alex@renda.org', haiti_map)
    #labelers.show_majority_vote(haiti_map)
    #labelers.show_prob_vote(haiti_map)
    plt.show()
    #print labelers.emails
    #print labelers.donmez_vote(np.arange(labelers.n), 0.5, False)
    #print labelers.UI()
    #labelers.donmez_vote(np.arange(labelers.n/3, labelers.n), 0.5, True)

    #print zip(labelers.emails, labelers.UI())
    '''labelers.donmez_vote(np.arange(labelers.n), 0.75, True)
    print labelers.UI()
    print labelers.rewards
    haiti_map = map_overlay.haiti_setup()
    labelers.show_majority_vote(haiti_map)
    labelers.show_prob_vote(haiti_map)
    labelers.show_labeler('jaredsfrank@gmail.com', haiti_map)
    labelers.show_labeler('dk72mi@gmail.com', haiti_map)
    labelers.get_FPR_TPR()
    plt.show()'''
   

if __name__ == '__main__':
    test()



