def get_train_fn(alg_name):
    if alg_name == 'mfvi':
        from algorithms.mfvi.mfvi_trainer import mfvi_trainer
        return mfvi_trainer

    elif alg_name == 'gmmvi':
        from algorithms.gmmvi.gmmvi_trainer import gmmvi_trainer
        return gmmvi_trainer

    elif alg_name == 'nfvi':
        from algorithms.nfvi.nfvi_trainer import nfvi_trainer
        return nfvi_trainer

    elif alg_name == 'smc':
        from algorithms.smc.smc_trainer import smc_trainer
        return smc_trainer

    elif alg_name == 'aft':
        from algorithms.aft.aft_trainer import aft_trainer
        return aft_trainer

    elif alg_name == 'craft':
        from algorithms.craft.craft_trainer import craft_trainer
        return craft_trainer

    elif alg_name == 'fab':
        from algorithms.fab.train.fab_trainer import fab_trainer
        return fab_trainer

    elif alg_name == 'ula':
        from algorithms.ula.ula_trainer import ula_trainer
        return ula_trainer

    elif alg_name == 'uha':
        from algorithms.uha.uha_trainer import uha_trainer
        return uha_trainer

    elif alg_name == 'mcd':
        from algorithms.mcd.mcd_trainer import mcd_trainer
        return mcd_trainer

    elif alg_name == 'cmcd':
        from algorithms.cmcd.cmcd_trainer import cmcd_trainer
        return cmcd_trainer

    elif alg_name == 'ldvi_depr':
        from algorithms.langevin_diffusion.ldvi_trainer import ldvi_trainer
        return ldvi_trainer

    elif alg_name == 'ldvi':
        from algorithms.ldvi.ldvi_trainer import ldvi_trainer
        return ldvi_trainer

    elif alg_name == 'dis':
        from algorithms.dis.dis_trainer import dis_trainer
        return dis_trainer

    elif alg_name == 'pis':
        from algorithms.pis.pis_trainer import pis_trainer
        return pis_trainer

    elif alg_name == 'dds':
        from algorithms.dds.dds_trainer import dds_trainer
        return dds_trainer

    elif alg_name == 'gfn':
        from algorithms.gfn.gfn_trainer import gfn_trainer
        return gfn_trainer

    elif alg_name == 'gbs':
        from algorithms.gbs.gbs_trainer import gbs_trainer
        return gbs_trainer

    elif alg_name == 'scld':
        from algorithms.scld.scld import scld_trainer
        return scld_trainer

    else:
        raise ValueError(f'No algorithm named {alg_name}.')
