

class SGD(object):
    def __init__(self, params, lr=0.001, l2=0., momentum=0.):
        self.params = params
        self.lr = lr
        self.l2 = l2
        self.momentum = momentum
        self.velocity = None

    def zero_grad(self):
        for name in self.params:
            for weight in self.params[name]:
                self.params[name][weight].grad = 0.

    def step(self):
        if self.velocity is None:
            self.velocity = {}
            for name in self.params:
                self.velocity[name] = {}
                for weight in self.params[name]:
                    if weight != 'b':
                        gradient = self.lr * (self.params[name][weight].grad + (self.l2 * self.params[name][weight].data))
                        self.velocity[name][weight] = gradient
                        self.params[name][weight].data -= self.velocity[name][weight]
                    else:
                        gradient = (self.lr * self.params[name][weight].grad)
                        self.velocity[name][weight] = gradient
                        self.params[name][weight].data -= self.velocity[name][weight]
        else:
            for name in self.params:
                for weight in self.params[name]:
                    if weight != 'b':
                        gradient = self.lr * (self.params[name][weight].grad + (self.l2 * self.params[name][weight].data))
                        self.velocity[name][weight] = (self.momentum * self.velocity[name][weight]) + gradient
                        self.params[name][weight].data -= self.velocity[name][weight]
                    else:
                        gradient = (self.lr * self.params[name][weight].grad)
                        self.velocity[name][weight] = (self.momentum * self.velocity[name][weight]) + gradient
                        self.params[name][weight].data -= self.velocity[name][weight]
