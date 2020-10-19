# #stastic helper functions-----------------------------------------
# def accuracy(out, yb): #
#      return (torch.argmax(out, dim=1)==yb).float().mean()

# def normalize(x, m, s):
#     return (x-m)/s

# def prev_pow_2(x): #consider converting to generator
#     """returns the previous power of 2 of X"""
#     return 2**math.floor(math.log2(x))

# #Pytorch Statistic Hooks
# def append_stats(hook, mod, inp, outp):
#     """collect statistics using Pytorch Hooks, creates a histogram of the collected data"""
#     if not hasattr(hook,'stats'): hook.stats = ([],[],[])
#     means,stds,hists = hook.stats
#     means.append(outp.data.mean().cpu())
#     stds .append(outp.data.std().cpu())
#     hists.append(outp.data.cpu().histc(40,-7,7))




# def append_mean_std(hook, model, inp, outp):
#     """attaches a hook that gets the mean and standard deviation"""
#     if not hasattr(hook, 'stats'):
#         hook.stats=([],[])
#     means, stds = hook.stats
#     if model.training:
#         means.append(outp.data.mean())
#         stds.append(outp.data.std())