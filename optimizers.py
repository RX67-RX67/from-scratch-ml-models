import numpy as np

# for cce
def to_one_hot(y, num_classes):
    one_hot = np.zeros((y.shape[0], num_classes))
    one_hot[np.arange(y.shape[0]), y] = 1
    return one_hot

class Optimizer:
    def __init__(self, model, loss_function = 'mse', learning_rate = 0.01, regularization = None, regularization_lambda = 0.0):
        self.model = model
        self.loss = loss_function
        self.lr = learning_rate
        self.reg = regularization
        self.reg_lambda = regularization_lambda
    
    def loss_functions(self, y_true, y_pred):
        if self.loss == 'mse':
            mse = np.mean((y_true - y_pred)**2)
            return mse
        elif self.loss == 'bce':
            bce = -np.mean(y_true*np.log(y_pred)+(1-y_true)*np.log(1-y_pred))
            return bce
        elif self.loss == 'cce':
            if y_true.ndim == 1:  
                y_true = to_one_hot(y_true, y_pred.shape[1])
            cce = -np.mean(np.sum(y_true * np.log(y_pred + 1e-9), axis=1))
            return cce
        else:
            raise ValueError('Unsupported loss functions. Choose within mse/bce/cce')
    
    def compute_gradient(self, x, y_true, eps = 1e-5):

        original_parameters = self.model.flatten_parameters()
        print("---flatten parameters---")
        print(original_parameters)
        grad = np.zeros_like(original_parameters)
        

        for i in range(len(original_parameters)):

            perturb = np.zeros_like(original_parameters)
            perturb[i] = eps

            self.model.update_parameters(original_parameters + perturb)
            y_pred_add = self.model.forward(x)
            loss_add = self.loss_functions(y_true, y_pred_add)

            self.model.update_parameters(original_parameters - perturb)
            y_pred_minus = self.model.forward(x)
            loss_minus = self.loss_functions(y_true, y_pred_minus)

            grad[i] = (loss_add - loss_minus) / (2*eps)
        
        print('---the gradient is comupted---')
        print(grad)

        self.model.update_parameters(original_parameters)

        return grad
    
    def gradient_descent(self, x_train, y_train, x_val, y_val, epochs = 1000, patience = 10):

        best_loss = float('inf')
        patience_counter = 0

        num_epoch = []
        val_loss_lst = []
        train_loss_lst = []

        for epoch in range(epochs):

            grad = self.compute_gradient(x_train, y_train)
            params = self.model.flatten_parameters()

            if self.reg is not None and self.reg_lambda > 0:
                if self.reg == 'l2':
                    grad += self.reg_lambda * params
                elif self.reg == 'l1':
                    grad += self.reg_lambda * np.sign(params)


            new_params = params - self.lr * grad
            self.model.update_parameters(new_params)

            y_val_pred = self.model.forward(x_val)
            y_train_pred = self.model.forward(x_train)
            val_loss = self.loss_functions(y_val, y_val_pred)
            train_loss = self.loss_functions(y_train, y_train_pred)

            num_epoch.append(epoch)
            val_loss_lst.append(val_loss)
            train_loss_lst.append(train_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'early stopping at epoch {epoch}')
                    break
        
        return num_epoch, val_loss_lst, train_loss_lst
        
                





    