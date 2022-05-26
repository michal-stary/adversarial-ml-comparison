OPTIMAL_HYPERS_NORM = {
    "L1":{"norm":"L1", "α_init":"1-", "γ_init":0.05, "primal_lr":1, "max_eps":24},
    
    "L2": {"norm":"L2", "α_init":"5", "γ_init":0.05, "init_norm":3, "primal_lr":1, "max_eps":1.0},
    
    "Linf":{"norm":"Linf", "γ_init":"0.05", "α_init":"10-", "primal_lr":"1", "max_eps":0.1}

}


OPTIMAL_HYPERS_BEST = {
    "L1":{"norm":"L1"},
    
    "L2": {"norm":"L2"},
    
    "Linf":{"norm":"Linf"}

}