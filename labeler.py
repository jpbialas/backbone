import numpy as np
import cv2
import csv
import map_overlay
import matplotlib.pyplot as plt
from scipy.stats import t

class Labelers:
    def __init__(self, n = 97710, basic_setup = True):
        self.user_map = {}
        self.labels = np.array([])
        self.rewards = np.array([]) #nx3 matrix for n labelers, storing sum(x), sum(x^2), n_rewards
        self.n = n
        if basic_setup:
            self.basic_setup()

    

    @property
    def emails(self):
        all_emails = np.empty(len(self.user_map)).astype('string')
        for i in self.user_map:
            all_emails[self.user_map[i]] = i
        return all_emails


    def _unique_emails(self, fn):
        with open(fn, mode='r') as infile:
            reader = csv.reader(infile)
            self.user_map = {}
            for row in reader:
                email = row[0][:row[0].find('_')] 
                if not email in self.user_map:
                    self.user_map[email] = len(self.user_map)

    def basic_setup(self):
        fn = 'damagelabels20/labels.csv'
        self._unique_emails(fn)
        self.rewards = np.tile(np.array([1,1,2]), (len(self.user_map),1))
        self.labels = np.zeros((len(self.user_map), self.n))
        with open(fn, mode='r') as infile:
            reader = csv.reader(infile)
            for row in reader:
                email = row[0][:row[0].find('_')]
                labels = np.array(map(int, row[3][1:-1].split(',')))
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
        else:
            self.labels = np.array(boolean_map).reshape((1,self.n))
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
        return np.where(all_ui>threshold)[0]


    def donmez_vote(self, label_indices, error, update = True):
        top_indices = self.top_labelers(error)
        top_voters = self.labels[top_indices][:, label_indices]
        majority_vote = np.sum(top_voters, axis = 0)/float(top_voters.shape[0])>=0.5
        if update:
            new_rewards = (top_voters == majority_vote)
            bonus = ((new_rewards+top_voters)>1)*19 #Bonus for getting 1 correct
            new_rewards = new_rewards+bonus
            self.rewards[top_indices,0] += np.sum(new_rewards, axis = 1)
            self.rewards[top_indices,1] += np.sum(new_rewards**2, axis = 1)
            self.rewards[top_indices,2] += new_rewards.shape[1] 
        return majority_vote


    def majority_vote(self, label_indices):
        return np.sum(self.labels[:,label_indices], axis = 0)/float(self.labels.shape[0])>=0.5


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
        img = disp_map.mask_segments(self.majority_vote(np.arange(self.n)), level, with_img = True, opacity = .4)
        self.show_img(img, 'Majority Vote')


    def show_prob_vote(self, disp_map, level = 20):
        segs = disp_map.segmentations[level][1]
        probs = np.sum(self.labels, axis = 0).astype('float')/float(len(self.user_map))
        img = disp_map.mask_helper(disp_map.img, probs[segs], opacity = .8)
        self.show_img(img, 'Probability Vote')


    def show_labeler(self, email, disp_map, level = 20):
        img = disp_map.mask_segments(self.labeler(email), level, with_img = True, opacity = .4)
        self.show_img(img, email)


    def get_FPR_TPR(self):
        probs = np.sum(self.labels, axis = 0).astype('float')/float(len(self.user_map))
        TPR = np.sum(self.labels*probs, axis = 1)/np.sum(probs)
        FPR = np.sum(self.labels*(1-probs), axis = 1)/np.sum(1-probs)
        fig, ax = plt.subplots()
        ax.scatter(FPR, TPR)
        names = self.emails
        ax.plot(np.array([0,1]), np.array([0,1]))
        for i in range(FPR.shape[0]):
            ax.annotate(names[i], (FPR[i], TPR[i]))
        ax.annotate('Under Labeler', (.002, .07), bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5))
        ax.annotate('Over Labeler', (.052, .9), bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5))
        ax.annotate('Good Labeler', (.002, .9), bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5))
        ax.annotate('Bad Labeler', (.052, .07), bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5))
        plt.title('Label Comparison'), plt.xlabel('FPR'), plt.ylabel('TPR'), plt.ylim([0,1]), plt.xlim([0,.06])



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

    labelers = Labelers()
    #print labelers.emails
    #print labelers.donmez_vote(np.arange(labelers.n), 0.5, False)
    #print labelers.UI()
    labelers.donmez_vote(np.arange(labelers.n), 0.5, True)
    print labelers.UI()
    print labelers.emails
    labelers.donmez_vote(np.arange(labelers.n), 0.75, True)
    print labelers.UI()
    print labelers.rewards
    haiti_map = map_overlay.haiti_setup()
    labelers.show_majority_vote(haiti_map)
    labelers.show_prob_vote(haiti_map)
    labelers.show_labeler('jaredsfrank@gmail.com', haiti_map)
    labelers.show_labeler('dk72mi@gmail.com', haiti_map)
    labelers.get_FPR_TPR()
    plt.show()
   

if __name__ == '__main__':
    test()



