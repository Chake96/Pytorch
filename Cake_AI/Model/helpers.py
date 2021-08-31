def model_summary(runner, model, valid_dl, find_all=False):
    runner.in_train = False #monkey patch to getaround callback ordering issues
    xb,yb = get_one_batch(valid_dl, runner)
    device = next(model.parameters()).device#Model may not be on the GPU yet
    xb,yb = xb.to(device),yb.to(device)
    mods = find_modules(model, is_lin_layer) if find_all else model.children() #find the linear layers, or the immediate children
    f = lambda hook,mod,inp,out: print(f"{mod}\n{out.shape}\n")
    with Hooks(mods, f) as hooks: model(xb)

#helper function to find modules within a model recursively
def get_all_modules(model, predictate):
    if predictate(model):
        return [model]
    return sum([get_all_modules(o, predictate) for o in model.children()], [])

#passed as predicate function to get_all_modules
def is_linear_layer(layer):
    lin_layers = (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear, nn.ReLU)
    return isinstance(layer, lin_layers)