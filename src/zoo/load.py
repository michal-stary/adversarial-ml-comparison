from robustbench.utils import load_model as load_rb_model

from zoo.kwta.load_model import load_model as load_kwta
from zoo.robust_union.load_model import load_model_avg as load_union_avg
from zoo.robust_union.load_model import load_model_msd as load_union_msd

ROBUSTBENCH_MODELS = {
    'Augustin2020Adversarial_34_10_extra': ('L2', 'Augustin2020Adversarial_34_10_extra',),
    'Engstrom2019Robustness_l2': ('L2', 'Engstrom2019Robustness',),
    'Engstrom2019Robustness_linf': ('Linf', 'Engstrom2019Robustness',),
    'Rice2020Overfitting': ('L2', 'Rice2020Overfitting',),
    'Carmon2019Unlabeled': ('Linf', 'Carmon2019Unlabeled',),
}

EXTERNAL_MODELS = {
    'Xiao2020Enhancing':  # https://github.com/iclrsubmission/kwta
        ('Linf', load_kwta),
    'Maini2020MultipleAVG':  # https://github.com/locuslab/robust_union
        ('Lp', load_union_avg),
    'Maini2020MultipleMSD':  # https://github.com/locuslab/robust_union
        ('Lp', load_union_msd),
}

ALL_MODELS = list(EXTERNAL_MODELS.keys()) + list(ROBUSTBENCH_MODELS.keys())


def load_robustbench(model_name, dataset, model_dir):
    threat_model, model_key = ROBUSTBENCH_MODELS[model_name]
    model = load_rb_model(model_name=model_key, dataset=dataset, threat_model=threat_model, model_dir=model_dir)
    model.eval()
    return model


def load_externals(model_name, dataset, model_dir):
    threat_model , load_fn = EXTERNAL_MODELS[model_name]
    model = load_fn(dataset, threat_model, model_dir)
    return model


def load_model(model_name, dataset, model_dir):
    if model_name in EXTERNAL_MODELS.keys():
        return load_externals(model_name, dataset, model_dir)
    try:
        return load_robustbench(model_name=model_name, dataset=dataset, model_dir=model_dir)
    except ValueError as e:
        print(f'Model not available in {threat_model}')
      