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
    'Rade2021Helper_R18_ddpm': ('L2', 'Rade2021Helper_R18_ddpm',),
    'Rebuffi2021Fixing_R18_cutmix_ddpm': ('L2', 'Rebuffi2021Fixing_R18_cutmix_ddpm',),
    'Gowal2021Improving_R18_ddpm_100m': ('Linf', 'Gowal2021Improving_R18_ddpm_100m', ),
    'Rade2021Helper_R18_extra': ('Linf', 'Rade2021Helper_R18_extra',),
    'Rebuffi2021Fixing_70_16_cutmix_extra': ('Linf', 'Rebuffi2021Fixing_70_16_cutmix_extra',),
    'Gowal2021Improving_70_16_ddpm_100m': ('Linf', 'Gowal2021Improving_70_16_ddpm_100m',),
    'Gowal2020Uncovering_70_16_extra': ('Linf', 'Gowal2020Uncovering_70_16_extra', ),
    'Kang2021Stable': ('Linf', 'Kang2021Stable'),
    'Gowal2020Uncovering_extra': ('L2', 'Gowal2020Uncovering_extra'),
    'Rebuffi2021Fixing_70_16_cutmix_extra': ('L2', 'Rebuffi2021Fixing_70_16_cutmix_extra'),
    'Rebuffi2021Fixing_70_16_cutmix_ddpm': ('L2', 'Rebuffi2021Fixing_70_16_cutmix_ddpm')
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
      