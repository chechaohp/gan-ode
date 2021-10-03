import torch

class GANODETrainer(object):

    def __init__(self, g_params, ds_params, dt_params, g_loss, ds_loss, dt_loss, 
                lr = 0.005, reg=0.002, method='rk2'):
        self.g_params = list(g_params)
        self.ds_params = list(ds_params)
        self.dt_params = list(dt_params)
        self.g_loss = g_loss
        self.ds_loss = ds_loss
        self.dt_loss = dt_loss
        self.lr = lr
        self.reg = reg
        self.method = method
        self.ode_step = self.choose_method()
        self.penalty = self.reg > 0
        self.g_iters = 1
        self.d_iters = 1

    def choose_method(self):
        assert self.method in ['euler','rk2','rk4'], "Choose method between 'euler', 'rk2' and 'rk4'"
        if self.method == 'euler':
            return self.euler
        elif self.method == 'rk2':
            return self.rk2
        else:
            return self.rk4

    def step(self,real_videos, real_labels, fake_videos_sample, fake_videos_downsample, z_class):
        # self.dt_loss = dt_loss
        return self.ode_step(real_videos, real_labels, fake_videos_sample, fake_videos_downsample, z_class)

    def g_step(self,fake_videos_sample, fake_videos_downsample, z_class):
        gloss1 = self.g_loss(fake_videos_sample, fake_videos_downsample, z_class)
        gloss = gloss1.item()
        # v_phi
        g_grad1 = torch.autograd.grad(gloss1, self.g_params)
        if self.medthod == 'euler':
            # update parameter
            with torch.no_grad():
                # update G
                for (param, grad) in zip(self.g_params, g_grad1):
                    param.sub_(self.lr * grad)
        elif self.method == 'rk2':
            # x1~ = x1 + h*grad1
            with torch.no_grad():
                # phi tilde
                for (param, grad) in zip(self.g_params, g_grad1):
                    param.add_(-self.lr * grad)

            g_grad2 = torch.autograd.grad(self.g_loss(fake_videos_sample, fake_videos_downsample, z_class), self.g_params)
           
            # x1~ = x1 + h*grad1
            # x2 = x1 + h/2(grad1+grad2) = x1~ - h*grad1 + h/2(grad1 + grad2) = x1~ + h/2(-grad1 + grad2)
            with torch.no_grad():
                # update G
                for (param, g1, g2) in zip(self.g_params, g_grad1, g_grad2):
                    param.add_(-0.5 * self.lr * (g2-g1))
        else:
            #x_k2 =  x_k1 - h/2 * grad1
            with torch.no_grad():
                # phi tilde
                for (param, grad) in zip(self.g_params, g_grad1):
                    param.add_(-self.lr / 2 * grad)
            
            g_grad2 = torch.autograd.grad(self.g_loss(fake_videos_sample, fake_videos_downsample, z_class), self.g_params)

            #x_k3 = x_k1 - h/2 * grad2 
            #     = x_k2 + h/2 * grad1 - h/2* grad2
            #     = x_k2 - h/2 * (-grad1 + grad2)
            with torch.no_grad():
                # phi tilde
                for (param, g1, g2) in zip(self.g_params, g_grad1, g_grad2):
                    param.add_(-self.lr / 2 * (- g1 + g2))

            g_grad3 = torch.autograd.grad(self.g_loss(fake_videos_sample, fake_videos_downsample, z_class), self.g_params)

            #x_k4 = x_k1 - h * grad3
            #     = x_k2 + h/2 *grad1 - h*grad3
            #     = x_k3 + h/2(-grad1 + grad2) + h/2 *grad1 - h*grad3
            #     = x_k3 - h * (-grad2/2 + grad3)

            with torch.no_grad():
                # phi tilde
                for (param, g2, g3) in zip(self.g_params, g_grad2, g_grad3):
                    param.add_(-self.lr * (- g2 / 2 + g3))

            g_grad4 = torch.autograd.grad(self.g_loss(fake_videos_sample, fake_videos_downsample, z_class), self.g_params)

            #x_{k+1} = x_k1 - h/6 *(grad1 + 2 * grad2 + 2 * grad3 + grad4) 
            #        = x_k2 + h/2 * grad1 - h/6 *(grad1 + 2 * grad2 + 2 * grad3 + grad4) 
            #        = x_k3 + h/2(-grad1 + grad2) + h/2*grad1 - h/6 *(grad1 + 2 * grad2 + 2 * grad3 + grad4) 
            #        = x_k4 + h * (-grad2/2 + grad3) + h/2(-grad1 + grad2) + h/2*grad1 - h/6 *(grad1 + 2 * grad2 + 2 * grad3 + grad4) 
            #        = x_k4 - h * (grad2/2  - grad3 + grad1/2 - grad2/2 - grad1/2 + grad1/6 + grad2/3 + grad3/3 + grad4/6)
            #        = x_k4 - h * (grad1/6 + grad2/3 -2*grad3/3 + grad4/6)

            with torch.no_grad():
                # update G
                for (param, g1, g2, g3, g4) in zip(self.g_params, g_grad1, g_grad2, g_grad3, g_grad4):
                    param.add_(-self.lr * (g1/6 + g2/3 - 2*g3/3 + g4/6))
        return gloss


    def d_step(self,real_videos, real_labels, fake_videos_sample, fake_videos_downsample, z_class):
        dsloss1 = self.ds_loss(real_videos,real_labels,fake_videos_sample,z_class)
        dtloss1 = self.dt_loss(real_videos,real_labels,fake_videos_downsample,z_class)
        gloss1 = self.g_loss(fake_videos_sample, fake_videos_downsample, z_class)
        dsloss = dsloss1.item()
        dtloss = dtloss1.item()
        # v_theta
        ds_grad1 = torch.autograd.grad(dsloss1, self.ds_params)
        dt_grad1 = torch.autograd.grad(dtloss1, self.dt_params)
        # v_phi
        g_grad1 = torch.autograd.grad(gloss1, self.g_params, create_graph=self.penalty)
        if self.penalty:
            g_grad_magnitude = sum(g.square().sum() for g in g_grad1)
            ds_penalty = torch.autograd.grad(g_grad_magnitude, self.ds_params)
            dt_penalty = torch.autograd.grad(g_grad_magnitude, self.dt_params)
            for g in g_grad1:
                g.detach()
            # free memory
            del g_grad_magnitude
        if self.method == 'euler':
            # update parameter
            with torch.no_grad():
                # update D
                if self.penalty:
                    for (param, grad, gp) in zip(self.ds_params, ds_grad1, ds_penalty):
                        param.add_(self.lr * (-grad) - self.reg * gp)
                    for (param, grad, gp) in zip(self.dt_params, dt_grad1, dt_penalty):
                        param.add_(self.lr * (-grad) - self.reg * gp)
                else:
                    for (param, grad) in zip(self.ds_params, ds_grad1):
                        param.add_(-self.lr * grad)
                    for (param, grad) in zip(self.dt_params, dt_grad1):
                        param.add_(-self.lr * grad)
        elif self.mehod == 'rk2':
            # x1~ = x1 + h*grad1
            with torch.no_grad():
                # theta tilde
                for (param, grad) in zip(self.ds_params, ds_grad1):
                    param.add_(-self.lr * grad)
                for (param, grad) in zip(self.dt_params, dt_grad1):
                    param.add_(-self.lr * grad)

            ds_grad2 = torch.autograd.grad(self.ds_loss(real_videos,real_labels,fake_videos_sample,z_class), self.ds_params)
            dt_grad2 = torch.autograd.grad(self.dt_loss(real_videos,real_labels,fake_videos_downsample,z_class), self.dt_params)
            
            # x1~ = x1 + h*grad1
            # x2 = x1 + h/2(grad1+grad2) = x1~ - h*grad1 + h/2(grad1 + grad2) = x1~ + h/2(-grad1 + grad2)
            with torch.no_grad():
                # update D
                # D get additional -reg*lr*penalty for gradient regularization
                if self.penalty:
                    for (param, g1, g2, gp) in zip(self.ds_params, ds_grad1, ds_grad2, ds_penalty):
                        param.add_(-0.5 * self.lr * (g2-g1) - self.reg * self.lr * gp)
                    for (param, g1, g2, gp) in zip(self.dt_params, dt_grad1, dt_grad2, dt_penalty):
                        param.add_(-0.5 * self.lr * (g2-g1) - self.reg * self.lr * gp)
                else:
                    for (param, g1, g2) in zip(self.ds_params, ds_grad1, ds_grad2):
                        param.add_(-0.5 * self.lr * (g2-g1))
                    for (param, g1, g2) in zip(self.dt_params, dt_grad1, dt_grad2):
                        param.add_(-0.5 * self.lr * (g2-g1))
        else:
            #x_k2 =  x_k1 - h/2 * grad1
            with torch.no_grad():
                # theta tilde
                for (param, grad) in zip(self.ds_params, ds_grad1):
                    param.add_(-self.lr / 2 * grad)
                for (param, grad) in zip(self.dt_params, dt_grad1):
                    param.add_(-self.lr / 2 * grad)
            
            ds_grad2 = torch.autograd.grad(self.ds_loss(real_videos,real_labels,fake_videos_sample,z_class), self.ds_params)
            dt_grad2 = torch.autograd.grad(self.dt_loss(real_videos,real_labels,fake_videos_downsample,z_class), self.dt_params)

            #x_k3 = x_k1 - h/2 * grad2 
            #     = x_k2 + h/2 * grad1 - h/2* grad2
            #     = x_k2 - h/2 * (-grad1 + grad2)
            with torch.no_grad():
                # theta tilde
                for (param, g1, g2) in zip(self.ds_params, ds_grad1, ds_grad2):
                    param.add_(-self.lr / 2 * (- g1 + g2))
                for (param, g1, g2) in zip(self.dt_params, dt_grad1, dt_grad2):
                    param.add_(-self.lr / 2 * (- g1 + g2))

            ds_grad3 = torch.autograd.grad(self.ds_loss(real_videos,real_labels,fake_videos_sample,z_class), self.ds_params)
            dt_grad3 = torch.autograd.grad(self.dt_loss(real_videos,real_labels,fake_videos_downsample,z_class), self.dt_params)
            
            #x_k4 = x_k1 - h * grad3
            #     = x_k2 + h/2 *grad1 - h*grad3
            #     = x_k3 + h/2(-grad1 + grad2) + h/2 *grad1 - h*grad3
            #     = x_k3 - h * (-grad2/2 + grad3)

            with torch.no_grad():
                # theta tilde
                for (param, g2, g3) in zip(self.ds_params, ds_grad2, ds_grad3):
                    param.add_(-self.lr * (- g2 / 2 + g3))
                for (param, g2, g3) in zip(self.dt_params, dt_grad2, dt_grad3):
                    param.add_(-self.lr * (- g2 / 2 + g3))

            ds_grad4 = torch.autograd.grad(self.ds_loss(real_videos,real_labels,fake_videos_sample,z_class), self.ds_params)
            dt_grad4 = torch.autograd.grad(self.dt_loss(real_videos,real_labels,fake_videos_downsample,z_class), self.dt_params)

            #x_{k+1} = x_k1 - h/6 *(grad1 + 2 * grad2 + 2 * grad3 + grad4) 
            #        = x_k2 + h/2 * grad1 - h/6 *(grad1 + 2 * grad2 + 2 * grad3 + grad4) 
            #        = x_k3 + h/2(-grad1 + grad2) + h/2*grad1 - h/6 *(grad1 + 2 * grad2 + 2 * grad3 + grad4) 
            #        = x_k4 + h * (-grad2/2 + grad3) + h/2(-grad1 + grad2) + h/2*grad1 - h/6 *(grad1 + 2 * grad2 + 2 * grad3 + grad4) 
            #        = x_k4 - h * (grad2/2  - grad3 + grad1/2 - grad2/2 - grad1/2 + grad1/6 + grad2/3 + grad3/3 + grad4/6)
            #        = x_k4 - h * (grad1/6 + grad2/3 -2*grad3/3 + grad4/6)

            with torch.no_grad():
                # update D
                # D get additional -reg*lr*penalty for gradient regularization
                if self.penalty:
                    for (param, g1, g2, g3, g4, gp) in zip(self.ds_params, ds_grad1, ds_grad2, ds_grad3, ds_grad4, ds_penalty):
                        param.add_(-self.lr * (g1/6 + g2/3 - 2*g3/3 + g4/6) - self.reg * self.lr * gp)
                    for (param, g1, g2, g3, g4, gp) in zip(self.dt_params, dt_grad1, dt_grad2, dt_grad3, dt_grad4, dt_penalty):
                        param.add_(-self.lr * (g1/6 + g2/3 - 2*g3/3 + g4/6) - self.reg * self.lr * gp)
                else:
                    for (param, g1, g2, g3, g4) in zip(self.ds_params, ds_grad1, ds_grad2, ds_grad3, ds_grad4):
                        param.add_(-self.lr * (g1/6 + g2/3 - 2*g3/3 + g4/6))
                    for (param, g1, g2, g3, g4) in zip(self.dt_params, dt_grad1, dt_grad2, dt_grad3, dt_grad4):
                        param.add_(-self.lr * (g1/6 + g2/3 - 2*g3/3 + g4/6))
        return dsloss, dtloss

    def euler(self, real_videos, real_labels, fake_videos_sample, fake_videos_downsample, z_class):
        """ Euler Method
        """
        dsloss1 = self.ds_loss(real_videos,real_labels,fake_videos_sample,z_class)
        dtloss1 = self.dt_loss(real_videos,real_labels,fake_videos_downsample,z_class)
        gloss1 = self.g_loss(fake_videos_sample, fake_videos_downsample, z_class)
        dsloss = dsloss1.item()
        dtloss = dtloss1.item()
        gloss = gloss1.item()
        # v_theta
        ds_grad1 = torch.autograd.grad(dsloss1, self.ds_params)
        dt_grad1 = torch.autograd.grad(dtloss1, self.dt_params)
        # v_phi
        g_grad1 = torch.autograd.grad(gloss1, self.g_params, create_graph=self.penalty)

        if self.penalty:
            g_grad_magnitude = sum(g.square().sum() for g in g_grad1)
            ds_penalty = torch.autograd.grad(g_grad_magnitude, self.ds_params)
            dt_penalty = torch.autograd.grad(g_grad_magnitude, self.dt_params)
            for g in g_grad1:
                g.detach()
            # free memory
            del g_grad_magnitude
        
        # update parameter
        with torch.no_grad():
            # update G
            for (param, grad) in zip(self.g_params, g_grad1):
                param.sub_(self.lr * grad)
            # update D
            if self.penalty:
                for (param, grad, gp) in zip(self.ds_params, ds_grad1, ds_penalty):
                    param.add_(self.lr * (-grad) - self.reg * gp)
                for (param, grad, gp) in zip(self.dt_params, dt_grad1, dt_penalty):
                    param.add_(self.lr * (-grad) - self.reg * gp)
            else:
                for (param, grad) in zip(self.ds_params, ds_grad1):
                    param.add_(-self.lr * grad)
                for (param, grad) in zip(self.dt_params, dt_grad1):
                    param.add_(-self.lr * grad)
        return gloss, dsloss, dtloss


    def rk2(self,real_videos, real_labels, fake_videos_sample, fake_videos_downsample, z_class):
        """ Heun's Method
        """
        dsloss1 = self.ds_loss(real_videos,real_labels,fake_videos_sample,z_class)
        dtloss1 = self.dt_loss(real_videos,real_labels,fake_videos_downsample,z_class)
        gloss1 = self.g_loss(fake_videos_sample, fake_videos_downsample, z_class)
        dsloss = dsloss1.item()
        dtloss = dtloss1.item()
        gloss = gloss1.item()
        # v_theta
        ds_grad1 = torch.autograd.grad(dsloss1, self.ds_params)
        dt_grad1 = torch.autograd.grad(dtloss1, self.dt_params)
        # v_phi
        g_grad1 = torch.autograd.grad(gloss1, self.g_params, create_graph=self.penalty)
        

        if self.penalty:
            g_grad_magnitude = sum(g.square().sum() for g in g_grad1)
            ds_penalty = torch.autograd.grad(g_grad_magnitude, self.ds_params, retain_graph=True)
            dt_penalty = torch.autograd.grad(g_grad_magnitude, self.dt_params)
            for g in g_grad1:
                g.detach()
            # free memory
            del g_grad_magnitude
        
        # x1~ = x1 + h*grad1
        with torch.no_grad():
            # phi tilde
            for (param, grad) in zip(self.g_params, g_grad1):
                param.add_(-self.lr * grad)
            # theta tilde
            for (param, grad) in zip(self.ds_params, ds_grad1):
                param.add_(-self.lr * grad)
            for (param, grad) in zip(self.dt_params, dt_grad1):
                param.add_(-self.lr * grad)
        

        g_grad2 = torch.autograd.grad(self.g_loss(fake_videos_sample, fake_videos_downsample, z_class), self.g_params)
        ds_grad2 = torch.autograd.grad(self.ds_loss(real_videos,real_labels,fake_videos_sample,z_class), self.ds_params)
        dt_grad2 = torch.autograd.grad(self.dt_loss(real_videos,real_labels,fake_videos_downsample,z_class), self.dt_params)
        
        # x1~ = x1 + h*grad1
        # x2 = x1 + h/2(grad1+grad2) = x1~ - h*grad1 + h/2(grad1 + grad2) = x1~ + h/2(-grad1 + grad2)
        with torch.no_grad():
            # update G
            for (param, g1, g2) in zip(self.g_params, g_grad1, g_grad2):
                param.add_(-0.5 * self.lr * (g2-g1))
            # update D
            # D get additional -reg*lr*penalty for gradient regularization
            if self.penalty:
                for (param, g1, g2, gp) in zip(self.ds_params, ds_grad1, ds_grad2, ds_penalty):
                    param.add_(-0.5 * self.lr * (g2-g1) - self.reg * self.lr * gp)
                for (param, g1, g2, gp) in zip(self.dt_params, dt_grad1, dt_grad2, dt_penalty):
                    param.add_(-0.5 * self.lr * (g2-g1) - self.reg * self.lr * gp)
            else:
                for (param, g1, g2) in zip(self.ds_params, ds_grad1, ds_grad2):
                    param.add_(-0.5 * self.lr * (g2-g1))
                for (param, g1, g2) in zip(self.dt_params, dt_grad1, dt_grad2):
                    param.add_(-0.5 * self.lr * (g2-g1))
        return gloss, dsloss, dtloss

    def rk4(self,real_videos, real_labels, fake_videos_sample, fake_videos_downsample, z_class):
        """ Runge Kutta 4
        """
        dsloss1 = self.ds_loss(real_videos,real_labels,fake_videos_sample,z_class)
        dtloss1 = self.dt_loss(real_videos,real_labels,fake_videos_downsample,z_class)
        gloss1 = self.g_loss(fake_videos_sample, fake_videos_downsample, z_class)
        dsloss = dsloss1.item()
        dtloss = dtloss1.item()
        gloss = gloss1.item()
        # v_theta
        ds_grad1 = torch.autograd.grad(dsloss1, self.ds_params)
        dt_grad1 = torch.autograd.grad(dtloss1, self.dt_params)
        # v_phi
        g_grad1 = torch.autograd.grad(gloss1, self.g_params, create_graph=self.penalty)

        if self.penalty:
            g_grad_magnitude = sum(g.square().sum() for g in g_grad1)
            ds_penalty = torch.autograd.grad(g_grad_magnitude, self.ds_params, retain_graph=True)
            dt_penalty = torch.autograd.grad(g_grad_magnitude, self.dt_params)
            for g in g_grad1:
                g.detach()
            # free memory
            del g_grad_magnitude

        #x_k2 =  x_k1 - h/2 * grad1
        with torch.no_grad():
            # phi tilde
            for (param, grad) in zip(self.g_params, g_grad1):
                param.add_(-self.lr / 2 * grad)
            # theta tilde
            for (param, grad) in zip(self.ds_params, ds_grad1):
                param.add_(-self.lr / 2 * grad)
            for (param, grad) in zip(self.dt_params, dt_grad1):
                param.add_(-self.lr / 2 * grad)
        
        g_grad2 = torch.autograd.grad(self.g_loss(fake_videos_sample, fake_videos_downsample, z_class), self.g_params)
        ds_grad2 = torch.autograd.grad(self.ds_loss(real_videos,real_labels,fake_videos_sample,z_class), self.ds_params)
        dt_grad2 = torch.autograd.grad(self.dt_loss(real_videos,real_labels,fake_videos_downsample,z_class), self.dt_params)

        #x_k3 = x_k1 - h/2 * grad2 
        #     = x_k2 + h/2 * grad1 - h/2* grad2
        #     = x_k2 - h/2 * (-grad1 + grad2)
        with torch.no_grad():
            # phi tilde
            for (param, g1, g2) in zip(self.g_params, g_grad1, g_grad2):
                param.add_(-self.lr / 2 * (- g1 + g2))
            # theta tilde
            for (param, g1, g2) in zip(self.ds_params, ds_grad1, ds_grad2):
                param.add_(-self.lr / 2 * (- g1 + g2))
            for (param, g1, g2) in zip(self.dt_params, dt_grad1, dt_grad2):
                param.add_(-self.lr / 2 * (- g1 + g2))

        g_grad3 = torch.autograd.grad(self.g_loss(fake_videos_sample, fake_videos_downsample, z_class), self.g_params)
        ds_grad3 = torch.autograd.grad(self.ds_loss(real_videos,real_labels,fake_videos_sample,z_class), self.ds_params)
        dt_grad3 = torch.autograd.grad(self.dt_loss(real_videos,real_labels,fake_videos_downsample,z_class), self.dt_params)
        
        #x_k4 = x_k1 - h * grad3
        #     = x_k2 + h/2 *grad1 - h*grad3
        #     = x_k3 + h/2(-grad1 + grad2) + h/2 *grad1 - h*grad3
        #     = x_k3 - h * (-grad2/2 + grad3)

        with torch.no_grad():
            # phi tilde
            for (param, g2, g3) in zip(self.g_params, g_grad2, g_grad3):
                param.add_(-self.lr * (- g2 / 2 + g3))
            # theta tilde
            for (param, g2, g3) in zip(self.ds_params, ds_grad2, ds_grad3):
                param.add_(-self.lr * (- g2 / 2 + g3))
            for (param, g2, g3) in zip(self.dt_params, dt_grad2, dt_grad3):
                param.add_(-self.lr * (- g2 / 2 + g3))

        g_grad4 = torch.autograd.grad(self.g_loss(fake_videos_sample, fake_videos_downsample, z_class), self.g_params)
        ds_grad4 = torch.autograd.grad(self.ds_loss(real_videos,real_labels,fake_videos_sample,z_class), self.ds_params)
        dt_grad4 = torch.autograd.grad(self.dt_loss(real_videos,real_labels,fake_videos_downsample,z_class), self.dt_params)

        #x_{k+1} = x_k1 - h/6 *(grad1 + 2 * grad2 + 2 * grad3 + grad4) 
        #        = x_k2 + h/2 * grad1 - h/6 *(grad1 + 2 * grad2 + 2 * grad3 + grad4) 
        #        = x_k3 + h/2(-grad1 + grad2) + h/2*grad1 - h/6 *(grad1 + 2 * grad2 + 2 * grad3 + grad4) 
        #        = x_k4 + h * (-grad2/2 + grad3) + h/2(-grad1 + grad2) + h/2*grad1 - h/6 *(grad1 + 2 * grad2 + 2 * grad3 + grad4) 
        #        = x_k4 - h * (grad2/2  - grad3 + grad1/2 - grad2/2 - grad1/2 + grad1/6 + grad2/3 + grad3/3 + grad4/6)
        #        = x_k4 - h * (grad1/6 + grad2/3 -2*grad3/3 + grad4/6)

        with torch.no_grad():
            # update G
            for (param, g1, g2, g3, g4) in zip(self.g_params, g_grad1, g_grad2, g_grad3, g_grad4):
                param.add_(-self.lr * (g1/6 + g2/3 - 2*g3/3 + g4/6))
            # update D
            # D get additional -reg*lr*penalty for gradient regularization
            if self.penalty:
                for (param, g1, g2, g3, g4, gp) in zip(self.ds_params, ds_grad1, ds_grad2, ds_grad3, ds_grad4, ds_penalty):
                    param.add_(-self.lr * (g1/6 + g2/3 - 2*g3/3 + g4/6) - self.reg * self.lr * gp)
                for (param, g1, g2, g3, g4, gp) in zip(self.dt_params, dt_grad1, dt_grad2, dt_grad3, dt_grad4, dt_penalty):
                    param.add_(-self.lr * (g1/6 + g2/3 - 2*g3/3 + g4/6) - self.reg * self.lr * gp)
            else:
                for (param, g1, g2, g3, g4) in zip(self.ds_params, ds_grad1, ds_grad2, ds_grad3, ds_grad4):
                    param.add_(-self.lr * (g1/6 + g2/3 - 2*g3/3 + g4/6))
                for (param, g1, g2, g3, g4) in zip(self.dt_params, dt_grad1, dt_grad2, dt_grad3, dt_grad4):
                    param.add_(-self.lr * (g1/6 + g2/3 - 2*g3/3 + g4/6))
        return gloss, dsloss, dtloss