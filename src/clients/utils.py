class AverageMeter(object):
    """
    Computes and stores the average and current value of metrics.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """
        String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)


class BatchIterator:
    def __init__(self, n, x):
        self.idx = 0
        self.n = n
        self.x = x
        self.ite_x = iter(x)

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx < self.n:
            try:
                res = next(self.ite_x)
            except StopIteration:
                self.ite_x = iter(self.x)
                res = next(self.ite_x)
            self.idx += 1
            return res
        else:
            raise StopIteration

    def __len__(self):
        return self.n
