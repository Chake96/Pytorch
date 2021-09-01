class Processor(): 
    def process(self, items): return items

class CategoryProcessor(Processor):
    def __init__(self): self.vocab=None
    
    def __call__(self, items):
        #The vocab is defined for the training set, so create the vocab
        if self.vocab is None:
            self.vocab = get_unique_keys(items)
            self.otoi  = {v:k for k,v in enumerate(self.vocab)} #reverse mapping 
        return [self.process_one(o) for o in items]
    def process_one(self, item):
        return self.otoi[item]
    
    def deprocess(self, idxs):
        assert self.vocab is not None
        return [self.deprocess_one(idx) for idx in idxs]
    def deprocess_one(self, idx): 
        return self.vocab[idx]
