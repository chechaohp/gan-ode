import torch

class GANODETrainer(object):

    def __init__(self, g_params, dImg_params, dVid_params, g_loss, dImg_loss, dVid_loss , lr = 0.02, reg=0.01, method='rk4', d_iter = 2, g_iter = 1):
        self.g_params = list(g_params)
        self.dImg_params = list(dImg_params)
        if dVid_params is not None:
            self.dVid_params = list(dVid_params)
        else:
            self.dVid_params = None
        self.g_loss = g_loss
        self.dImg_loss = dImg_loss
        self.dVid_loss = dVid_loss
        self.lr = lr
        self.reg = reg
        self.method = method
        self.ode_step = self.choose_method()
        self.penalty = self.reg > 0
        self.d_iter = d_iter
        self.g_iter = g_iter


    def choose_method(self):
        assert self.method in ['euler','rk2','rk4'], "Choose method between 'euler', 'rk2' and 'rk4'"
        if self.method == 'euler':
            print('Choosing Euler method')
            return self.euler_step
        elif self.method == 'rk2':
            print('Choosing Huen method')
            return self.rk2_step
        else:
            print('Choosing Runge-Kutta 4 method')
            return self.rk4_step
    

    def step(self,x = None, model = 'gen'):
        assert model in ['gen','dis_img','dis_vid']
        # self.dt_loss = dt_loss
        if model == 'gen':
            loss = self.ode_step(self.g_params, self.g_loss, x, False)
        if model == 'dis_img':
            loss = self.ode_step(self.dImg_params, self.dImg_loss, x, self.penalty)
        if model == 'dis_vid':
            loss = self.ode_step(self.dVid_params, self.dVid_loss, x, self.penalty)
        return loss

    def calculate_reg(self):
        g_loss = self.g_loss()
        g_grad = torch.autograd.grad(g_loss, self.g_params,create_graph = True, allow_unused=True)
        g_grad_magnitude = sum(g.square().sum() for g in g_grad if g is not None)
        # d_penalty = torch.autograd.grad(g_grad_magnitude, d_params, allow_unused=True)
        for g in g_grad:
            if g is not None:
                g.detach()
        # del g_grad_magnitude
        return g_grad_magnitude

    def euler_step(self, params, loss_fn, x = None, penalty = False):
        """ Euler step for all model, discriminator and generator (x) is abstract
        """
        # find loss
        if x is not None:
            loss = loss_fn(x)
        else:
            loss = loss_fn()
        # find gradient
        grad1 = torch.autograd.grad(loss,params, allow_unused=True)
        if penalty:
            g_grad_magnitude = self.calculate_reg()
            grad_penalty = torch.autograd.grad(g_grad_magnitude, params)
        # update parameter
        with torch.no_grad():
            if penalty:
                for (param, grad, gp) in zip(params, grad1, grad_penalty):
                    param.add_(self.lr * (-grad) + self.reg * (-gp))
            else:
                for (param, grad) in zip(params, grad1):
                    if grad is None:
                        continue
                    param.add_(self.lr * (-grad))
        return loss

    def rk2_step(self, params, loss_fn, x = None, penalty =False):
        """ RK2 step for all model,
        """
        if x is not None:
            loss1 = loss_fn(x)
        else:
            loss1 = loss_fn()
        # find gradient
        grad1 = torch.autograd.grad(loss1, params, allow_unused=True)
        if penalty:
            g_grad_magnitude = self.calculate_reg()
            grad_penalty = torch.autograd.grad(g_grad_magnitude, params)
        # update for the first time
        # x1~ = x1 + h *grad1
        with torch.no_grad():
            # phi tilde
            for (param, grad) in zip(params, grad1):
                if grad is None:
                    continue
                param.add_(self.lr * (-grad))

        # mew loss after update parameters
        loss2 = loss_fn() if x is None else loss_fn(x)
        grad2 = torch.autograd.grad(loss2, params,allow_unused=True)

        # update the second time
        # x1~ = x1 + h*grad1
        # x2 = x1 + h/2(grad1+grad2) 
        #    = x1~ - h*grad1 + h/2(grad1 + grad2) 
        #    = x1~ + h/2(-grad1 + grad2)
        with torch.no_grad():
            # update parameter
            if penalty:
                for (param, g1, g2, gp) in zip(params, grad1, grad2, grad_penalty):
                    if g1 is None:
                        continue
                    param.add_(0.5 * self.lr * (-g2+g1) - self.reg * self.lr * gp)
            else:
                for (param, g1, g2) in zip(params, grad1, grad2):
                    if g1 is None:
                        continue
                    param.add_(0.5 * self.lr * (-g2+g1))
        return loss1

    def rk4_step(self, params, loss_fn, x=None, penalty=False):
        """ Rk4
        """
        if x is not None:
            loss1 = loss_fn(x)
        else:
            loss1 = loss_fn()
        # find gradient
        grad1 = torch.autograd.grad(loss1, params, allow_unused=True)
        if penalty:
            g_grad_magnitude = self.calculate_reg()
            grad_penalty = torch.autograd.grad(g_grad_magnitude, params)

        # update the first time
        #x_k2 =  x_k1 + h/2 * grad1
        with torch.no_grad():
            # phi tilde
            for (param, grad) in zip(params, grad1):
                if grad is None:
                    continue
                param.add_(self.lr / 2 * (-grad))

        # new loss
        loss2 = loss_fn() if x is None else loss_fn(x)
        grad2 = torch.autograd.grad(loss2, params,allow_unused=True)
        # update the second time
        #x_k3 = x_k1 + h/2 * grad2 
        #     = x_k2 - h/2 * grad1 + h/2* grad2
        #     = x_k2 + h/2 * (-grad1 + grad2)
        with torch.no_grad():
            # phi tilde
            for (param, g1, g2) in zip(params, grad1, grad2):
                if g1 is None:
                    continue
                param.add_(self.lr / 2 * (g1 - g2))
        
        # new loss
        loss3 = loss_fn() if x is None else loss_fn(x)
        grad3 = torch.autograd.grad(loss3, params,allow_unused=True)
        
        # third update
        #x_k4 = x_k1 + h * grad3
        #     = x_k2 - h/2 *grad1 + h*grad3
        #     = x_k3 - h/2(-grad1 + grad2) - h/2 *grad1 + h*grad3
        #     = x_k3 + h * (-grad2/2 + grad3)
        with torch.no_grad():
            for (param, g2, g3) in zip(params, grad2, grad3):
                if g2 is None:
                    continue
                param.add_(self.lr * (g2 / 2 - g3))

        # new loss
        loss4 = loss_fn() if x is None else loss_fn(x)
        grad4 = torch.autograd.grad(loss4, params,allow_unused=True)

        # final update
        #x_{k+1} = x_k1 + h/6 *(grad1 + 2 * grad2 + 2 * grad3 + grad4) 
        #        = x_k2 - h/2 * grad1 + h/6 *(grad1 + 2 * grad2 + 2 * grad3 + grad4) 
        #        = x_k3 - h/2(-grad1 + grad2) - h/2*grad1 + h/6 *(grad1 + 2 * grad2 + 2 * grad3 + grad4) 
        #        = x_k4 - h * (-grad2/2 + grad3) - h/2(-grad1 + grad2) - h/2*grad1 + h/6 *(grad1 + 2 * grad2 + 2 * grad3 + grad4) 
        #        = x_k4 + h * (grad1/6 + grad2/3 -2*grad3/3 + grad4/6)
        with torch.no_grad():
            if penalty:
                for (param, g1, g2, g3, g4, gp) in zip(params, grad1, grad2, grad3, grad4, grad_penalty):
                    if g1 is None:
                        continue
                    param.add_(self.lr * (-g1/6 - g2/3 + 2*g3/3 - g4/6) - self.reg * self.lr * gp)
            else:
                for (param, g1, g2, g3, g4) in zip(params, grad1, grad2, grad3, grad4):
                    if g1 is None:
                        continue
                    param.add_(self.lr * (-g1/6 - g2/3 + 2*g3/3 - g4/6))
                    
        return loss1
