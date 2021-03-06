import numpy as np
import cv2
import csv
import map_overlay
import matplotlib.pyplot as plt
from object_model import ObjectClassifier
from scipy.stats import t
from sklearn.externals.joblib import Parallel, delayed

def model_vote_help(train_map, q_labels, majority_vote, new_train):
    new_model = ObjectClassifier(NZ = False)
    indcs = np.where(q_labels>-1)
    new_model.fit(train_map, (q_labels == majority_vote), indcs)
    probs = new_model.predict_proba_segs(train_map, new_train)
    probs[np.where(q_labels[new_train]==-2)] = 0
    return probs

class Labelers:
    def __init__(self, n = 97710, basic_setup = True):
        self.user_map = {}
        self.labels = np.array([])
        self.q_labels = np.array([])
        self.rewards = np.array([]) #nx3 matrix for n labelers, storing sum(x), sum(x^2), n_rewards
        self.n = n
        self.min_labeled = 2
        if basic_setup:
            self.basic_setup()

    

    @property
    def emails(self):
        all_emails = np.empty(len(self.user_map)).astype('string')
        for i in self.user_map:
            all_emails[self.user_map[i]] = i
        return all_emails

    def print_stats(self):
        image_count = np.array(self.image_count.values())
        stats = np.sum(image_count, axis = 0)
        double_labelers = np.sum(np.sum(image_count[:,1:], axis = 1)==2)
        image_one_labelers = stats[1] - double_labelers
        image_two_labelers = stats[2] - double_labelers
        print '{} Unique Labelers'.format(double_labelers+image_one_labelers+image_two_labelers)
        print '-------------------------------------'
        print '+    {} Labelers labeled both images'.format(double_labelers)
        print '+    {} Labelers only labeled image 1'.format(image_one_labelers)
        print '+    {} Labelers only labeled image 2'.format(image_two_labelers)


    def _unique_emails(self, fn):
        with open(fn, mode='r') as infile:
            reader = csv.reader(infile)
            self.user_map = {}
            self.image_count = {}
            for row in reader:
                email = row[0][:row[0].find('_')]
                img_num = int(row[2])
                if not email in self.image_count:
                    self.image_count[email] = [1,0,0]
                    self.image_count[email][img_num] = 1
                else:
                    self.image_count[email][0] += 1
                    self.image_count[email][img_num] = 1
                if self.image_count[email][0] == self.min_labeled:
                    self.user_map[email] = len(self.user_map)
                    np.sum(np.array(self.image_count.values()), axis = 0), len(self.image_count.values()), np.sum(np.sum(np.array(self.image_count.values())[:,1:], axis = 1)==2)
            #print self.image_count,np.array(self.image_count.values()), np.sum(1-np.array(self.image_count.values()), axis = 0)



    def basic_setup(self):
        fn = 'damagelabels20/labels9.csv'
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
                if email in self.user_map:
                    self.labels[self.user_map[email]][indices[img_num]] = 0
                    self.labels[self.user_map[email]][labels] = 1
        self.q_labels = (self.labels>-1).astype('int')-2


    def add_labels(self, email, boolean_map):
        if email in self.user_map:
            console.log("ERROR: User already in system")
        elif boolean_map.shape[0]!=self.n:
            console.log("ERROR: new labels must be same shape as others")
        elif self.labels.shape[0]>0:
            self.labels = np.vstack((self.labels, boolean_map))
            self.q_labels = np.vstack((self.q_labels, (boolean_map>-1).astype('int')-2))
            self.user_map[email] = len(self.user_map)
            self.rewards = np.vstack((self.rewards,np.array([1,1,2])))
        else:
            self.labels = np.array(boolean_map).reshape((1,self.n))
            self.q_labels = (self.labels>-1).astype('int')-2
            self.user_map[email] = 0
            self.rewards = np.array([1,1,2]).reshape((1,3))

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
        return np.where(all_ui>=threshold)[0]

    def model_start_old(self, train_map, new_train):
        tr_indcs = train_map.unique_segs(20)
        for i in range(len(self.labels)):
            new_train_indcs = tr_indcs[new_train]
            good_indices = new_train_indcs[np.where(self.labels[i,new_train_indcs]>-1)]
            self.q_labels[i,good_indices] = self.labels[i,good_indices]


    def model_vote_old(self, train_map, new_train):
        tr_indcs = train_map.unique_segs(20)
        new_train_indcs = tr_indcs[new_train]
        total_vote = np.zeros(len(new_train))
        count = np.zeros(len(new_train))
        majority_vote = self.majority_vote(labels = self.q_labels)
        P = Parallel(n_jobs = -1, verbose=1)(delayed(model_vote_help)(train_map, self.q_labels[i,tr_indcs], majority_vote[tr_indcs], new_train) for i in range(len(self.labels)))
        P = np.array(P)
        best_labelers = np.argmax(P, axis = 0)
        best_labels = self.labels[best_labelers,new_train_indcs]
        self.q_labels[best_labelers,new_train_indcs] = best_labels
        return best_labels

    def model_start(self, train_map, indcs, truth = None):
        predictions = np.zeros((self.labels.shape[0], train_map.unique_segs(20).shape[0]))
        agreement = (self.labels == self.majority_vote())[:,train_map.unique_segs(20)]
        if truth is not None:
            for i in range(agreement.shape[0]):
                print agreement[i,indcs].astype('int') - (self.labels[i,train_map.unique_segs(20)[indcs]] == truth).astype('int')

            agreement[:,indcs] = self.labels[:,train_map.unique_segs(20)[indcs]] == truth
            print "HERE"
            for i in range(agreement.shape[0]):
                print agreement[i,indcs].astype('int') - (self.labels[i,train_map.unique_segs(20)[indcs]] == truth).astype('int')
        for i in range(self.labels.shape[0]):
            new_model = ObjectClassifier(NZ = False)
            new_model.fit(train_map, agreement[i], indcs)
            probs = new_model.predict_proba_segs(train_map)
            predictions[i] = probs
            print predictions
        best_labelers = np.argmax(predictions, axis = 0)
        print best_labelers
        np.save('predictions.npy', predictions)
        np.save('best.npy',best_labelers)
        assert(best_labelers.shape[0] == train_map.unique_segs(20).shape[0])
        self.model_labels = self.labels[best_labelers,train_map.unique_segs(20)]
        np.save('vote.npy', self.model_labels)

    def model_vote(self, indcs):
        return self.model_labels[indcs]

        
    def donmez_pick_1(self, label_indices):
        all_ui = self.UI()
        P = self.labels[:, label_indices]>=0
        P = P*all_ui.reshape((all_ui.shape[0], 1))
        best_labelers = np.argmax(P, axis = 0)
        best_labels = self.labels[best_labelers,label_indices]
        return best_labels


    def donmez_vote(self, label_indices, error, update = True):
        top_indices = self.top_labelers(error)
        top_voters = self.labels[np.ix_(top_indices, label_indices)]
        majority_vote = self.majority_vote(label_indices, top_indices)#np.sum(top_voters, axis = 0)/float(top_voters.shape[0])>=0.5
        if update:
            new_rewards = (top_voters == majority_vote)
            #bonus = ((new_rewards+top_voters)>1)*14 #Bonus for getting 1 correct
            new_rewards = new_rewards#+bonus
            self.rewards[top_indices,0] += np.sum(new_rewards, axis = 1)
            self.rewards[top_indices,1] += np.sum(new_rewards**2, axis = 1)
            bcount = np.bincount(np.where(top_voters>=0)[0])
            self.rewards[top_indices[:bcount.shape[0]],2] += bcount
        return majority_vote


    def majority_vote(self, label_indices = None, labeler_indices = None, labels = None):
        labels = self.labels if labels is None else labels
        return self.probability(label_indices, labeler_indices, labels)>=0.5


    def probability(self, label_indices = None, labeler_indices = None, labels = None):
        labels = self.labels if labels is None else labels
        if label_indices is None:
            label_indices = np.arange(self.n)
        if labeler_indices is None:
            labeler_indices = np.arange(labels.shape[0])
        indcs = np.ix_(labeler_indices, label_indices)
        return np.sum(labels[indcs]>0, axis = 0)/np.sum(labels[indcs]>=0, axis = 0).astype('float')


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
        img = disp_map.mask_segments(self.majority_vote(np.arange(self.n)), level, with_img = True, opacity = 1)
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
        ax.annotate('Under Labeler', (.002, .07), bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5))
        ax.annotate('Over Labeler', (.092, .9), bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5))
        ax.annotate('Good Labeler', (.002, .9), bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5))
        ax.annotate('Bad Labeler', (.092, .07), bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5))
        plt.title('Label Comparison'), plt.xlabel('FPR'), plt.ylabel('TPR'), plt.ylim([0,1]), plt.xlim([0,1])



def test():
    
    haiti_map = map_overlay.haiti_setup()
    labelers = Labelers()
    labelers.print_stats()
    l = []
    for i,j in zip(labelers.image_count.keys(), np.array(labelers.image_count.values())[:,0]):
        l.append([i,j])
        print i,j
    np.savetxt('all_emails.csv', np.array(l), delimiter = ',', fmt = '%s')
    #labelers.show_labeler('kmkobosk@mtu.edu', haiti_map)
    #labelers.show_labeler('xuelingl@mtu.edu', haiti_map)
    #train     = np.ix_(np.arange(4096/3, 4096), np.arange(4096/2))
    #train_map = haiti_map.sub_map(train)
    #labelers.show_majority_vote(haiti_map)
    #labelers.show_prob_vote(haiti_map)
    #plt.figure()
   

if __name__ == '__main__':
    test()
    plt.show()



