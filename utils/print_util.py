
def print_results(step, logger, config):
    if config.verbose:
        try:
            print(f'Step {int(step)}: ELBO {float(logger["KL/elbo"][-1])}; Δ lnZ {float(logger["logZ/reverse"][-1])}; '
                  f'reverse_ESS {float(logger["ESS/reverse"][-1])}')
        except:
            print(f'Step {int(step)}: ELBO {float(logger["KL/elbo"][-1])}; Δ lnZ {float(logger["logZ/reverse"][-1])}')
