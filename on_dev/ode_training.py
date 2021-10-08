import torch
import torch.nn as nn

class GeneratorGradientReg(nn.Module):
    """ For
    """
    def __init__(self, g_params, reg):
        super().__init__()
        self.g_params = g_params
        self.reg = reg
    
    def forward(self, g_loss):
        g_grad = torch.autograd.grad(g_loss, self.g_params, create_graph = True)
        g_grad_magnitude = sum(g.square().sum() for g in g_grad)
        return self.reg * g_grad_magnitude

class GANODETrainer(object):

    def __init__(self, g_params, dImg_params, dVid_params, g_loss, dImg_loss, dVid_loss , lr = 0.02, reg=0.01, method='rk4', d_iter = 2, g_iter = 1):
        self.g_params = list(g_params)
        self.dImg_params = list(dImg_params)
        self.dVid_params = list(dVid_params)
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
            return self.euler_step
        elif self.method == 'rk2':
            return self.rk2_step
        else:
            return self.rk4_step
    

    def step(self,x = None, model = 'gen'):
        assert model in ['gen','dis_img','dis_vid']
        # self.dt_loss = dt_loss
        if model == 'gen':
            loss = self.ode_step(self.g_params, self.g_loss, x, False)
        if model == 'dis_img':
            loss = self.ode_step(self.dImg_params, self.dImg_loss, x, True)
        if model == 'dis_vid':
            loss = self.ode_step(self.dVid_params, self.dVid_loss, x, True)
        return loss

    def calculate_reg(self, g_grad, d_params):
        g_grad_magnitude = sum(g.square().sum() for g in g_grad)
        d_penalty = torch.autograd.grad(g_grad_magnitude, d_params)
        # dt_penalty = torch.autograd.grad(g_grad_magnitude, self.dt_params)
        for g in g_grad:
            g.detach()
        del g_grad_magnitude
        return d_penalty

    def euler_step(self, params, loss_fn, x = None, penalty = False):
        """ Euler step for all model, discriminator and generator (x) is abstract
        """
        # find loss
        if x is not None:
            loss = loss_fn(x)
        else:
            loss = loss_fn()
        # find gradient
        grad1 = torch.autograd.grad(loss,params, create_graph=penalty)
        if penalty:
            grad_penalty = self.calculate_reg(self.g_params, params)
        # update parameter
        with torch.no_grad():
            if penalty:
                for (param, grad, gp) in zip(self.ds_params, grad1, grad_penalty):
                    param.add_(self.lr * (-grad) + self.reg * (-gp))
            else:
                for (param, grad) in zip(params, grad1):
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
        grad1 = torch.autograd.grad(loss1, params, create_graph=penalty)
        if penalty:
            grad_penalty = self.calculate_reg(self.g_params, params)
        # update for the first time
        # x1~ = x1 + h *grad1
        with torch.no_grad():
            # phi tilde
            for (param, grad) in zip(params, grad1):
                param.add_(self.lr * (-grad))

        # mew loss after update parameters
        loss2 = loss_fn() if x is None else loss_fn(x)
        grad2 = torch.autograd.grad(loss2, params)

        # update the second time
        # x1~ = x1 + h*grad1
        # x2 = x1 + h/2(grad1+grad2) 
        #    = x1~ - h*grad1 + h/2(grad1 + grad2) 
        #    = x1~ + h/2(-grad1 + grad2)
        with torch.no_grad():
            # update parameter
            if penalty:
                for (param, g1, g2, gp) in zip(params, grad1, grad2, grad_penalty):
                    param.add_(0.5 * self.lr * (-g2+g1) - self.reg * self.lr * gp)
            else:
                for (param, g1, g2, gp) in zip(params, grad1, grad2, grad_penalty):
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
        grad1 = torch.autograd.grad(loss1, params, create_graph=penalty)
        if penalty:
            grad_penalty = self.calculate_reg(self.g_params, params)

        # update the first time
        #x_k2 =  x_k1 + h/2 * grad1
        with torch.no_grad():
            # phi tilde
            for (param, grad) in zip(params, grad1):
                param.add_(self.lr / 2 * (-grad))

        # new loss
        loss2 = loss_fn() if x is None else loss_fn(x)
        grad2 = torch.autograd.grad(loss2, params)
        # update the second time
        #x_k3 = x_k1 + h/2 * grad2 
        #     = x_k2 - h/2 * grad1 + h/2* grad2
        #     = x_k2 + h/2 * (-grad1 + grad2)
        with torch.no_grad():
            # phi tilde
            for (param, g1, g2) in zip(params, grad1, grad2):
                param.add_(self.lr / 2 * (g1 - g2))
        
        # new loss
        loss3 = loss_fn() if x is None else loss_fn(x)
        grad3 = torch.autograd.grad(loss3, params)
        
        # third update
        #x_k4 = x_k1 + h * grad3
        #     = x_k2 - h/2 *grad1 + h*grad3
        #     = x_k3 - h/2(-grad1 + grad2) - h/2 *grad1 + h*grad3
        #     = x_k3 + h * (-grad2/2 + grad3)
        with torch.no_grad():
            for (param, g2, g3) in zip(params, grad2, grad3):
                param.add_(self.lr * (g2 / 2 - g3))

        # new loss
        loss4 = loss_fn() if x is None else loss_fn(x)
        grad4 = torch.autograd.grad(loss4, params)

        # final update
        #x_{k+1} = x_k1 + h/6 *(grad1 + 2 * grad2 + 2 * grad3 + grad4) 
        #        = x_k2 - h/2 * grad1 + h/6 *(grad1 + 2 * grad2 + 2 * grad3 + grad4) 
        #        = x_k3 - h/2(-grad1 + grad2) - h/2*grad1 + h/6 *(grad1 + 2 * grad2 + 2 * grad3 + grad4) 
        #        = x_k4 - h * (-grad2/2 + grad3) - h/2(-grad1 + grad2) - h/2*grad1 + h/6 *(grad1 + 2 * grad2 + 2 * grad3 + grad4) 
        #        = x_k4 + h * (grad1/6 + grad2/3 -2*grad3/3 + grad4/6)
        with torch.no_grad():
            if penalty:
                for (param, g1, g2, g3, g4, gp) in zip(params, grad1, grad2, grad3, grad4, grad_penalty):
                    param.add_(self.lr * (-g1/6 - g2/3 + 2*g3/3 - g4/6) - self.reg * self.lr * gp)
            else:
                for (param, g1, g2, g3, g4) in zip(params, grad1, grad2, grad3, grad4):
                    param.add_(self.lr * (-g1/6 - g2/3 + 2*g3/3 - g4/6))
                    
        return loss1

    def euler(self,x):
        """ Euler Method
        """
        dloss1 = self.ds_loss(x)
        gloss1 = self.g_loss()
        dloss = dloss1.item()
        gloss = gloss1.item()
        # v_theta
        ds_grad1 = torch.autograd.grad(dloss1, self.ds_params)
        dt_grad1 = torch.autograd.grad(self.dt_loss(), self.dt_params)
        # v_phi
        g_grad1 = torch.autograd.grad(gloss1, self.g_params, create_graph=self.penalty)

        if self.penalty:
            ds_penalty, dt_penalty = self.calculate_reg(g_grad1)
        
        # update parameter
        with torch.no_grad():
            # update G
            for (param, grad) in zip(self.g_params, g_grad1):
                param.add_(self.lr * (-grad))
            # update D
            if self.penalty:
                for (param, grad, gp) in zip(self.ds_params, ds_grad1, ds_penalty):
                    param.add_(self.lr * (-grad) + self.reg * (-gp))
                for (param, grad, gp) in zip(self.ds_params, dt_grad1, dt_penalty):
                    param.add_(self.lr * (-grad) + self.reg * (-gp))
            else:
                for (param, grad) in zip(self.ds_params, ds_grad1):
                    param.add_(self.lr * (-grad))
                for (param, grad) in zip(self.ds_params, dt_grad1):
                    param.add_(self.lr * (-grad))
        return gloss, dloss


    def rk2(self,x):
        """ Heun's Method
        """
        dloss1 = self.ds_loss(x)
        gloss1 = self.g_loss()
        dloss = dloss1.item()
        gloss = gloss1.item()
        # v_theta
        ds_grad1 = torch.autograd.grad(dloss1, self.ds_params)
        dt_grad1 = -torch.autograd.grad(self.dt_loss(), self.dt_params)
        # v_phi
        g_grad1 = torch.autograd.grad(gloss1, self.g_params, create_graph=self.penalty)
        
        if self.penalty:
            ds_penalty, dt_penalty = self.calculate_reg(g_grad1)
        
        # x1~ = x1 + h *grad1
        with torch.no_grad():
            # phi tilde
            for (param, grad) in zip(self.g_params, g_grad1):
                param.add_(self.lr * (-grad))
            # theta tilde
            for (param, grad) in zip(self.ds_params, ds_grad1):
                param.add_(self.lr * (-grad))
            for (param, grad) in zip(self.dt_params, dt_grad1):
                param.add_(self.lr * (-grad))
        

        g_grad2 = torch.autograd.grad(self.g_loss(), self.g_params)
        ds_grad2 = torch.autograd.grad(self.ds_loss(x), self.ds_params)
        dt_grad2 = torch.autograd.grad(self.dt_loss(x), self.dt_params)
        
        # x1~ = x1 + h*grad1
        # x2 = x1 + h/2(grad1+grad2) 
        #    = x1~ - h*grad1 + h/2(grad1 + grad2) 
        #    = x1~ + h/2(-grad1 + grad2)
        with torch.no_grad():
            # update G
            for (param, g1, g2) in zip(self.g_params, g_grad1, g_grad2):
                param.add_(0.5 * self.lr * (-g2+g1))
            # update D
            # D get additional -reg*lr*penalty for gradient regularization
            if self.penalty:
                for (param, g1, g2, gp) in zip(self.ds_params, ds_grad1, ds_grad2, ds_penalty):
                    param.add_(0.5 * self.lr * (-g2+g1) - self.reg * self.lr * gp)
                for (param, g1, g2, gp) in zip(self.dt_params, dt_grad1, dt_grad2, dt_penalty):
                    param.add_(0.5 * self.lr * (-g2+g1) - self.reg * self.lr * gp)
            else:
                for (param, g1, g2) in zip(self.ds_params, ds_grad1, ds_grad2):
                    param.add_(0.5 * self.lr * (-g2+g1))
                for (param, g1, g2) in zip(self.dt_params, dt_grad1, dt_grad2):
                    param.add_(0.5 * self.lr * (-g2+g1))
        return gloss, dloss

    def rk4(self,x):
        """ Runge Kutta 4
        """
        dloss1 = self.ds_loss(x)
        gloss1 = self.g_loss()
        dloss = dloss1.item()
        gloss = gloss1.item()
        # v_theta
        ds_grad1 = torch.autograd.grad(dloss1, self.ds_params)
        dt_grad1 = -torch.autograd.grad(self.dt_loss(), self.dt_params)
        # v_phi
        g_grad1 = torch.autograd.grad(gloss1, self.g_params, create_graph=self.penalty)

        if self.penalty:
            ds_penalty, dt_penalty = self.calculate_reg(g_grad1)

        #x_k2 =  x_k1 + h/2 * grad1
        with torch.no_grad():
            # phi tilde
            for (param, grad) in zip(self.g_params, g_grad1):
                param.add_(self.lr / 2 * (-grad))
            # theta tilde
            for (param, grad) in zip(self.ds_params, ds_grad1):
                param.add_(self.lr / 2 * (-grad))
            for (param, grad) in zip(self.dt_params, dt_grad1):
                param.add_(self.lr / 2 * (-grad))
        
        g_grad2 = torch.autograd.grad(self.g_loss(), self.g_params)
        ds_grad2 = torch.autograd.grad(self.ds_loss(x), self.ds_params)
        dt_grad2 = torch.autograd.grad(self.dt_loss(), self.dt_params)

        #x_k3 = x_k1 + h/2 * grad2 
        #     = x_k2 - h/2 * grad1 + h/2* grad2
        #     = x_k2 + h/2 * (-grad1 + grad2)
        with torch.no_grad():
            # phi tilde
            for (param, g1, g2) in zip(self.g_params, g_grad1, g_grad2):
                param.add_(self.lr / 2 * (g1 - g2))
            # theta tilde
            for (param, g1, g2) in zip(self.ds_params, ds_grad1, ds_grad2):
                param.add_(self.lr / 2 * (g1 - g2))
            for (param, g1, g2) in zip(self.ds_params, dt_grad1, dt_grad2):
                param.add_(self.lr / 2 * (g1 - g2))

        g_grad3 = torch.autograd.grad(self.g_loss(), self.g_params)
        ds_grad3 = torch.autograd.grad(self.ds_loss(x), self.ds_params)
        dt_grad3 = torch.autograd.grad(self.dt_loss(), self.dt_params)
        
        #x_k4 = x_k1 + h * grad3
        #     = x_k2 - h/2 *grad1 + h*grad3
        #     = x_k3 - h/2(-grad1 + grad2) - h/2 *grad1 + h*grad3
        #     = x_k3 + h * (-grad2/2 + grad3)

        with torch.no_grad():
            # phi tilde
            for (param, g2, g3) in zip(self.g_params, g_grad2, g_grad3):
                param.add_(self.lr * (g2 / 2 - g3))
            # theta tilde
            for (param, g2, g3) in zip(self.ds_params, ds_grad2, ds_grad3):
                param.add_(self.lr * (g2 / 2 - g3))
            for (param, g2, g3) in zip(self.ds_params, dt_grad2, dt_grad3):
                param.add_(self.lr * (g2 / 2 - g3))

        g_grad4 = torch.autograd.grad(self.g_loss(), self.g_params)
        ds_grad4 = torch.autograd.grad(self.ds_loss(x), self.ds_params)
        dt_grad4 = torch.autograd.grad(self.dt_loss(), self.dt_params)

        #x_{k+1} = x_k1 + h/6 *(grad1 + 2 * grad2 + 2 * grad3 + grad4) 
        #        = x_k2 - h/2 * grad1 + h/6 *(grad1 + 2 * grad2 + 2 * grad3 + grad4) 
        #        = x_k3 - h/2(-grad1 + grad2) - h/2*grad1 + h/6 *(grad1 + 2 * grad2 + 2 * grad3 + grad4) 
        #        = x_k4 - h * (-grad2/2 + grad3) - h/2(-grad1 + grad2) - h/2*grad1 + h/6 *(grad1 + 2 * grad2 + 2 * grad3 + grad4) 
        #        = x_k4 + h * (grad1/6 + grad2/3 -2*grad3/3 + grad4/6)

        with torch.no_grad():
            # update G
            for (param, g1, g2, g3, g4) in zip(self.g_params, g_grad1, g_grad2, g_grad3, g_grad4):
                param.add_(self.lr * (-g1/6 - g2/3 + 2*g3/3 - g4/6))
            # update D
            # D get additional -reg*lr*penalty for gradient regularization
            if self.penalty:
                for (param, g1, g2, g3, g4, gp) in zip(self.ds_params, ds_grad1, ds_grad2, ds_grad3, ds_grad4, ds_penalty):
                    param.add_(self.lr * (-g1/6 - g2/3 + 2*g3/3 - g4/6) - self.reg * self.lr * gp)
                for (param, g1, g2, g3, g4, gp) in zip(self.ds_params, ds_grad1, ds_grad2, ds_grad3, ds_grad4, dt_penalty):
                    param.add_(self.lr * (-g1/6 - g2/3 + 2*g3/3 - g4/6) - self.reg * self.lr * gp)
            else:
                for (param, g1, g2, g3, g4) in zip(self.ds_params, ds_grad1, ds_grad2, ds_grad3, ds_grad4):
                    param.sub_(self.lr * (-g1/6 - g2/3 + 2*g3/3 - g4/6))
                for (param, g1, g2, g3, g4) in zip(self.dt_params, dt_grad1, dt_grad2, dt_grad3, dt_grad4):
                    param.sub_(self.lr * (-g1/6 - g2/3 + 2*g3/3 - g4/6))
        return gloss, dloss


