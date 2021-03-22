""""###Trainer class """
import torch
import matplotlib.pyplot as plt
import os

def plot_metrics(metrics: dict, task_no = -1, title="Losses for given task"):
    """
    Plots training/validation losses.
    :param metrics: dictionar
    """
    plt.figure()
    plt.plot(metrics['train_losses'][task_no], c='r', label='Train loss') #RIGHT NOW ONLY PLOTS LOSSES ON THE TASK THAT WAS TRAINED - that's what the -1 does.
    plt.plot(metrics['test_losses'][task_no], c='g', label='Test loss')
    plt.plot(metrics['train_acc'][task_no], c='b', label='Train acc')
    plt.plot(metrics['test_acc'][task_no], c='y', label='Test acc')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.legend()
    plt.show(block=False)


class Trainer:
    def __init__(self, hyperparams: dict, model = None):
        self.device = hyperparams['device']
        self.model = model
        self.learning_algo = hyperparams['learning_algo']
        self.lr = hyperparams['learning_rate']
        self.weight_decay = hyperparams['weight_decay']
        if self.model is not None:
            self.model = self.model.to(self.device)
            self.set_optimizer()
        self.num_epochs = hyperparams['num_epochs']
        self.lr = hyperparams['learning_rate']
        self.batch_size = hyperparams['batch_size']
        self.basepath = hyperparams['basepath']
        self.save_appendix = ""
        self.task_no = 0
        self.save_interm = True

    def set_optimizer(self):
        if self.learning_algo == 'adam':
            self.optimizer = torch.optim.Adam(params = self.model.parameters(),
                                              lr = self.lr, weight_decay = self.weight_decay)
        else:
            self.optimizer = torch.optim.SGD(params=self.model.parameters(),
                                             lr = self.lr, weight_decay = self.weight_decay)
    def save(self):
        self.save_interm = True
        return

    def toggle_save(self):
        self.save_interm = not self.save_interm
        return

    def nosave(self):
        self.save_interm = False

    def set_saveloc(self, save_appendix):
        self.save_appendix = save_appendix
        return

    def set_task(self, task_no):
        self.task_no = task_no

    def save_metrics(self, metrics, epoch):
        import pickle
        with open(os.path.join(self.basepath, 'savedump',
                               f"{self.model.__class__.__name__}_{epoch}_epochs_metrics_task_{str(self.task_no) + self.save_appendix}"),
                  'wb') as filehandle:
            pickle.dump(metrics, filehandle)
        return

    def save_model(self, epoch):
        torch.save(self.model.state_dict(),
                   os.path.join(self.basepath, 'savedump',
                                f"{self.model.__class__.__name__}_{epoch}_epochs_model_after_task{str(self.task_no)}{self.save_appendix}"))
        return

    def set_model(self, model_type, cl_dset, load = False, load_attr = {}):
        """
        load_attributes: dict containing as keys: basepath, path, suffix, num_epochs, task
        """
        from auxiliary import get_model
        self.model = get_model(model_type = model_type, cl_dset = cl_dset, device = self.device)
        if load:
            assert {"basepath", "path", "suffix"}.issubset(load_attr.keys()), "Incorrect load parameters provided"
            self.model.load_state_dict(torch.load(os.path.join(load_attr["basepath"], load_attr["path"],
                                                  f"{model.__class__.__name__}_{load_attr['num_epochs']}_epochs_model_after_task{load_attr['task']}{load_attr['suffix']}")))
        self.model = self.model.to(self.device)
        self.set_optimizer()
        return

    def train_epoch(self, train_loader, epoch_num: int, prev_fisher = None, prev_params = None, compute_new_fisher = False):
        """
            Compute the loss on the training set
            :param
            epoch_num: number  of current epoch
        """
        self.model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        loss = torch.nn.CrossEntropyLoss()
        for batch_num, (batch_imgs, batch_labels) in enumerate(train_loader):
            params = None
            batch_imgs, batch_labels = batch_imgs.to(self.device), batch_labels.to(self.device)
            # reset gradients

            self.optimizer.zero_grad()
            batch_logits=self.model(batch_imgs)
            batch_loss = loss(batch_logits, batch_labels)

            batch_loss_corrected = batch_loss

            #update batch loss if we use fisher (if we have it)
            if prev_fisher is not None and prev_params is not None:
                params = None
                for param in self.model.parameters():
                    if params is None:
                        params = param.view([1, -1])
                    else:
                        params = torch.cat((params, param.view([1, -1])), dim=1)  # will add to columns
                param_square_dif = torch.pow(params-prev_params, 2)
                if batch_num == 0:
                    print(f"prev fisher has shape {prev_fisher.shape}, prev_params have shape {prev_params.shape} and current params have shape {params.shape} (their difference having shape {param_square_dif.shape}")
                batch_loss_corrected = batch_loss + torch.dot(torch.squeeze(prev_fisher), torch.squeeze(param_square_dif) )
                print(f"Regular batch loss is {batch_loss} and the corrected one is {batch_loss_corrected}")

            epoch_loss += batch_loss.item()
            batch_loss_corrected.backward()

            # update parameters
            self.optimizer.step()

            with torch.no_grad():
              batch_acc = self.get_accuracy(batch_logits, batch_labels)
              epoch_acc += batch_acc.item()

            if batch_num % 100 == 0:
                print(f"(epoch: {epoch_num}, batch: {batch_num + 1}), batch loss = {batch_loss.item()}")

        epoch_loss /= (batch_num + 1)
        epoch_acc /= (batch_num + 1)
        print(f"epoch {epoch_num} has overall loss {epoch_loss}")

        fisher_diag = None
        """
        COMPUTING FISHER DIAGONAL.
        """
        if compute_new_fisher:
            print(f"Calculating Fisher for task {self.task_no} (at end of epoch {epoch_num}). \n")
            fisher_list = None
            for batch_num, (batch_imgs, _) in enumerate(train_loader):
                self.optimizer.zero_grad()
                batch_logits = self.model(batch_imgs.to(self.device))
                batch_probs = torch.log(torch.nn.functional.softmax(batch_logits, dim=1))
                for sample_idx in range(batch_probs.shape[0]):
                    for label_idx in range(batch_probs.shape[-1]):
                        grads = torch.autograd.grad(batch_probs[sample_idx, label_idx], self.model.parameters(),
                                                    create_graph=True)
                        # print(f"When calculating new fisher matrix, batch_grad of prob for sample {sample_idx} label {label_idx} is {grads[0]}")
                        if fisher_list is None:
                            fisher_list = list()
                            ###ADD A DETACH
                            for idx in range(len(grads)):
                                fisher_list.append(torch.pow(grads[idx].detach(), 2))
                        else:
                            for idx in range(len(grads)):
                                fisher_list[idx] += torch.pow(grads[idx].detach(), 2)
                    for idx in range(len(fisher_list)):
                        fisher_list[idx] /= batch_probs.shape[-1]
                for idx in range(len(fisher_list)):
                    fisher_list[idx] /= batch_probs.shape[0]
            for idx in range(len(fisher_list)):
                fisher_list[idx] /= batch_num
            """
            probs_grads is a tuple of gradients of each network parameter; 
            we need to take each element of this tuple and concatenate them in a loong tensor
            """
            for grad in fisher_list:
                if fisher_diag is None:
                    fisher_diag = grad.view([1, -1])
                else:
                    fisher_diag = torch.cat((fisher_diag, grad.view([1, -1])), dim=1)  # will add to columns
            fisher_diag.requires_grad = False
        return epoch_loss, epoch_acc, fisher_diag

    def test_epoch(self, test_loader) -> (float,float):
        """ Compute the loss on the test set
        :param
        epoch_num: number of current epoch
        """
        epoch_loss = 0.0
        epoch_acc = 0.0
        loss = torch.nn.CrossEntropyLoss()
        self.model.eval()
        with torch.no_grad():
            for batch_num, (batch_imgs, batch_labels) in enumerate(test_loader):
                batch_imgs, batch_labels = batch_imgs.to(self.device), batch_labels.to(self.device)
                batch_logits = self.model(batch_imgs)
                batch_loss = loss(batch_logits, batch_labels)
                batch_acc = self.get_accuracy(batch_logits, batch_labels)
                epoch_loss += batch_loss.item()
                epoch_acc += batch_acc.item()
            epoch_loss /= (batch_num + 1)
            epoch_acc /= (batch_num + 1)
        return epoch_loss, epoch_acc

    def get_accuracy(self, predictions, labels):
        """
        predictions are of the shape no_imgs x no_classes, representing probability of each img being in each class
        labels are of the same shape only binary
        average = 'weighted' is used in f1_score because it accounts for possible class imbalances
        """
        predictions = torch.argmax(predictions, dim=1).type(torch.FloatTensor)
        assert predictions.shape == labels.shape, "Predictions and labels don't have same shape: "+{predictions.shape}+" vs. "+{labels.shape}+" respectively."
        from sklearn.metrics import f1_score
        score=f1_score(y_true=labels.cpu(), y_pred=predictions.cpu(), average='weighted')
        return(score)

    def train(self, train_loaders, test_loaders, prev_fisher = None, prev_params = None) -> dict:
        """
        Expects one or multiple train loaders corresponding to specific tasks in a list
        Assumes the last train loader & test loader correspond to current task and that we've done the training on previous tasks before.
        At the end of each epoch of training on current task also retests perfomance on each previous task.
        """
        train_loader, test_loader = train_loaders[-1], test_loaders[-1]
        train_losses, test_losses, train_acc, test_acc = [[] for i in range(len(train_loaders))], [[] for i in range(len(train_loaders))], [[] for i in range(len(train_loaders))], [[] for i in range(len(train_loaders))]

        if True:
            for epoch in range(self.num_epochs):
                print(f"learning rate at epoch {epoch}: {[g['lr'] for g in self.optimizer.param_groups]}")
                epoch_train_loss, epoch_train_acc, fisher_diag = \
                    self.train_epoch(train_loader, epoch, compute_new_fisher = True if epoch == self.num_epochs-1 else False, prev_fisher = prev_fisher, prev_params = prev_params)
                epoch_test_loss, epoch_test_acc = self.test_epoch(test_loader)
                train_losses[-1].append(epoch_train_loss)
                test_losses[-1].append(epoch_test_loss)
                train_acc[-1].append(epoch_train_acc)
                test_acc[-1].append(epoch_test_acc)

                #INCREASE LR IF DIFFERENCE BETWEEN TWO EPOCHS GETS TOO SMALL; DECREASE IF ACC DECREASES SIGNIFICANTLY
                if (epoch > 0): #have trained at least one epoch
                  if (abs(test_acc[-1][-1] - test_acc[-1][-2])<5e-3): #acc increased less than 0.5%
                    for g in self.optimizer.param_groups:
                      g['lr'] = g['lr']*2
                  elif (test_acc[-1][-2] - test_acc[-1][-1]> 5e-2): #acc decreased more than 5%
                    for g in self.optimizer.param_groups:
                      g['lr'] = g['lr']/2


                #REVISIT PREVIOUS TASKS
                if len(train_loaders) > 1:
                  for prev_task_no in range(len(train_loaders)-1):
                    train_loader, test_loader = train_loaders[prev_task_no], test_loaders[prev_task_no]
                    prev_epoch_train_loss, prev_epoch_train_acc = self.test_epoch(train_loader)
                    prev_epoch_test_loss, prev_epoch_test_acc = self.test_epoch(test_loader)
                    train_losses[prev_task_no].append(prev_epoch_train_loss)
                    test_losses[prev_task_no].append(prev_epoch_test_loss)
                    train_acc[prev_task_no].append(prev_epoch_train_acc)
                    test_acc[prev_task_no].append(prev_epoch_test_acc)


                #SAVE
                if self.save_interm:
                    metrics_save = {"train_losses": train_losses,
                                        "test_losses": test_losses,
                                        "train_acc": train_acc,
                                        "test_acc": test_acc}
                    self.save_metrics(metrics_save, epoch)
                    self.save_model(epoch)

                #DO PRINTS AND PLOTS
                if False:
                    if epoch % 3 == 0 and epoch > 0:
                      interm_plot_metrics = {"train_losses": train_losses,
                          "test_losses": test_losses,
                          "train_acc": train_acc,
                          "test_acc": test_acc}
                      plot_metrics(metrics = interm_plot_metrics, task_no = -1, title = f"Metrics after {epoch} epochs with learning rate {self.lr} for model {self.model.__class__.__name__}.")
                      if len(train_loaders) > 1:
                        for task_no in range(len(train_loaders)-1):
                          plot_metrics(metrics = interm_plot_metrics, task_no = task_no, title = f"Same as before, revisiting task {task_no}.")

        return {"train_losses": train_losses,
                "test_losses": test_losses,
                "train_acc": train_acc,
                "test_acc": test_acc}, fisher_diag
