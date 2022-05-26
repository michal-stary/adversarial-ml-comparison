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
    
def min_norms_pred(step_norms, step_accs, step_preds):
    clean_acc = step_accs[0]
    
    running_min_norms = adversify(step_norms[0], step_accs[0], clean_acc)
    running_min_preds = step_preds[0]
    for step in range(len(step_norms)):
        stacked_norms = np.stack([running_min_norms, adversify(step_norms[step], step_accs[step], clean_acc)])
        stacked_preds = np.stack([running_min_preds, step_preds[step]])
        min_indc = np.argmin(stacked_norms, axis=0, keepdims=True)
#         print(min_indc)
        
#         print(stacked_preds.shape, stacked_norms.shape)
        running_min_norms = np.take_along_axis(stacked_norms, min_indc, axis=0)[0]
        running_min_preds = np.take_along_axis(stacked_preds, np.expand_dims(min_indc,  axis=2), axis=0)[0]
#         print(running_min_norms)

    return running_min_norms,running_min_preds


def SE(step_norms, step_accs, n_thrs=500, fixed_thrs=[8/255], max_eps=None):
    
    min_norms_ = min_norms(step_norms, step_accs)
    
    min_ = 0 #min_norms_.min()
    if max_eps is None:
        max_ = min_norms_[min_norms_!= np.inf].max()
    else:
        max_ = max_eps
        
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



def steps_to_find_adv(step_accs):
    """
    return the first step on which was adversarial perturbation found for each sample
    """
    first = np.argmin(step_accs, axis=0).astype(float)
    first[(step_accs==1).all(axis=0)] = np.nan
    return first

def first_to_final_ratio(step_norms, step_accs):
    """
    return the norm ratio of first missclasifed vs the last found 
    """
    
    first = steps_to_find_adv(step_accs)
    res = np.zeros(len(first))
    for i, adv_step in enumerate(first):
        # adversarial pert not found at all
        if np.isnan(adv_step):
            res[i] = np.nan
            continue
            
        # clean sample missclasified
        if step_accs[0][i] == 0:
            res[i] = np.nan
            continue
            
        res[i] = step_norms[int(adv_step)][i]/step_norms[-1][i]
    return res