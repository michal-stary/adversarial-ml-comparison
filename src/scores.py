THRESHOLDS = {
    "L1":[8, 12],
    
    "L2": [0.5],
    
    "Linf":[8/255]
}


SCORES = {
    "L1":[{"Maini2020MultipleAVG": .603, ## L1 apgd paper (AA, APGD)
            "Maini2020MultipleMSD": .582, ## L1 apgd paper (WC)
            "Augustin2020Adversarial_34_10_extra": .507, ## L1 apgd paper (AA)
            "Engstrom2019Robustness_l2": .442,  ## L1 apgd paper (AA)
            "Rice2020Overfitting": .429, ## L1 apgd paper (AA, APGD)
           "Xiao2020Enhancing": .224, ## L1 apgd paper (AA)
            "Engstrom2019Robustness_linf": .16 ## L1 apgd paper (AA)

           },
        {"Maini2020MultipleAVG": .468, ## L1 apgd paper (AA, APGD)
            "Maini2020MultipleMSD": .465, ## L1 apgd paper (AA, APGD)
            "Xiao2020Enhancing": .169, ## L1 apgd paper (AA)
            "Augustin2020Adversarial_34_10_extra": .31, ## L1 apgd paper (AA)
            "Engstrom2019Robustness_linf": .049, ## L1 apgd paper (AA, APGD)
            "Engstrom2019Robustness_l2": .269,  ## L1 apgd paper (AA, APGD)
            "Rice2020Overfitting": .237 ## L1 apgd paper (WC)
           }],
    
    "L2": [{"Rebuffi2021Fixing_70_16_cutmix_extra_l2": .8232, ## RB
            "Gowal2020Uncovering_extra": .8053, ## RB
            "Rebuffi2021Fixing_70_16_cutmix_ddpm": .8042, ## RB
            "Rebuffi2021Fixing_R18_cutmix_ddpm": .7586, ## RB
            "Rade2021Helper_R18_ddpm": .7615, ## RB
            "Engstrom2019Robustness_l2": .6924 ##RB
            }],
    
    "Linf":[{"Rebuffi2021Fixing_70_16_cutmix_extra_linf": .6656, ## RB
            "Gowal2021Improving_70_16_ddpm_100m": .6610, ## RB
            "Gowal2020Uncovering_70_16_extra": .6587, ## RB
            "Rade2021Helper_R18_extra": .5767, ## RB
            "Kang2021Stable": .6420, ##RB
            "Engstrom2019Robustness_linf": .4925, ##RB
            "Gowal2021Improving_R18_ddpm_100m": .585 ##RB
            }]
}