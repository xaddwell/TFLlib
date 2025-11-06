
from .poison import *
from .privacy import *


_backdoor_client_attacker_dict_ = {"DBA":DBA, "Cerp":CerP, "EdgeCase":EdgeCase, "A3FL":A3FL, "Replace":Replace, "Neurotoxin":Neurotoxin, "None":BasePoisonAttack}
                  
_byzantine_client_attacker_dict_ = {"IPM":IPM, "Noise":Noise, "LabelFlip":LabelFlip, "SignGuard":SignGuard, "LIE":LIE, "UpdateFlip":UpdateFlip, "MinMax":MinMax, "Fang":Fang, "MedianTailored":MedianTailored, "SignFlip":SignFlip, "None":BaseByzantineAttack}

_server_attacker_dict_ = {"dlg": OptBasedAttack.OptimizationBasedAttack, "inv_grads": OptBasedAttack.OptimizationBasedAttack, "see_thru_grads": OptBasedAttack.OptimizationBasedAttack, "LOKI": LOKI.LOKIAttack, "RobFed": RobFed.RobFedAttack,"iDLG": iDLG.iDLGAttack,"GI": GI.GIAttack,"iLRG": iLRG.iLRGAttack,"RLG": RLG.RLGAttack,"Shokri": ShokriAttack.ShokriAttacker, "Nasr": Nasr.WhiteboxPartialAttacker, "ML_leaks": ML_leaks.MLleaksAttacker,"SIA": SIA.SIAAttacker, "AMI": AMI.AMIAttacker, "Zari": Zari.WhiteboxEffectiveAttacker}

def initialize_client_attacker(client, conf, attack="backdoor"):

    bd_name = conf.bd_attack
    bzt_name = conf.bzt_attack
    
    if attack== "backdoor":
        
        if bd_name != "":
            assert bd_name in _backdoor_client_attacker_dict_.keys(),f"{bd_name} is not surported"
        elif bd_name == "":
            return BasePoisonAttack(attacker,conf)
        else:
            raise ValueError("bd_attack should be in [DBA, CerP, EdgeCase, A3FL, Replace, Neurotoxin, None]")
        
        attacker = _backdoor_client_attacker_dict_[bd_name](client,conf)
    
    elif attack == "byzantine":
        
        if bzt_name != "":
            assert bzt_name in _byzantine_client_attacker_dict_.keys(),f"{bzt_name} is not surported"
        elif bzt_name == "":
            return BaseByzantineAttack(attacker,conf)
        else:
            raise ValueError("bzt_attack should be in [IPM, Noise, LabelFlip, SignGuard, Lie, Zero, UpdateFlip, MinMax, MinSum, Fang, MedianTailored, None]")
        
        attacker = _byzantine_client_attacker_dict_[bzt_name](client,conf)
    
    else:

        attacker = BaseClientAttack(client,conf)
    
    return attacker

def initialize_server_attacker(server, conf):
    attack_name = conf.privacy_attack
    if attack_name != "":
        assert attack_name in _server_attacker_dict_.keys(),f"{attack_name} is not surported"
    else:
        raise ValueError("You should pass an attack name!")
    attacker = _server_attacker_dict_[attack_name](server,conf)

    return attacker

