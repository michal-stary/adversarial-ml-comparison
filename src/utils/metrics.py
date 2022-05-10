import numpy as np

# def QD_at_step(step_norm, step_acc, clean_acc):
#     # avoid inplace ops
#     step_norm_ = step_norm.copy()

#     step_norm_[step_acc==1] = np.inf
#     step_norm_[clean_acc==0] = 0
#     #assert step_acc[clean_acc==0].sum() == 0
    
#     return np.median(step_norm_)

def adversify(step_norm, step_acc, clean_acc):
    step_norm_ = step_norm.copy()
    step_norm_[step_acc==1] = np.inf
    step_norm_[clean_acc==0] = 0
    return step_norm_
    
    
def min_norms(step_norms, step_accs):
    clean_acc = step_accs[0]

    running_min_norms = adversify(step_norms[0], step_accs[0], clean_acc)
    for step in range(len(step_norms)):
        running_min_norms = np.minimum(running_min_norms, adversify(step_norms[step], step_accs[step], clean_acc))
        
    return running_min_norms
    
def SE(step_norms, step_accs, n_thrs=500, fixed_thrs=[8/255]):
    
    min_norms_ = min_norms(step_norms, step_accs)
    
    min_ = 0 #min_norms_.min()
    max_ = min_norms_[min_norms_!= np.inf].max()
    
    # set the "norm" of clean misclassified samples to -1
    min_norms_[step_accs[0]==0] = -1
    
    step_size = (max_ - min_)/n_thrs
    
    rob_acc = np.ones(shape=(n_thrs+len(fixed_thrs)))
    
    thresholds_ = (np.arange(n_thrs) * step_size) + min_
    
    eps_thrs = np.sort(np.concatenate([thresholds_, fixed_thrs]))
    
    for i, thr in enumerate(eps_thrs):
        rob_acc[i] = (min_norms_ >= thr).mean()
        
    return rob_acc, eps_thrs
    
def QD(step_norms, step_accs):
    clean_acc = step_accs[0]
    qd = list()
    
    running_min_norms = adversify(step_norms[0], step_accs[0], clean_acc)
    for step in range(len(step_norms)):
        running_min_norms = np.minimum(running_min_norms, adversify(step_norms[step], step_accs[step], clean_acc))
        qd.append(np.median(running_min_norms))
    return qd

def min_median(qd):
    return qd[-1]

def n_qs_to_reach(margin, qd):
    array = np.array(qd)
    mm = min_median(qd)
    
    return np.argmax(array<=mm*1.1)
    
def attack_succes_rate(step_accs):
    clean_acc = step_accs[0]
    robust = np.ones_like(clean_acc)
    for step_acc in step_accs:
        robust = np.minimum(robust, step_acc)
    return (1 - robust[clean_acc == 1].mean()) *100

