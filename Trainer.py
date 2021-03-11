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
        load_attributes: dict containing as keys: load_bpath, path, suffix, num_epochs, task
        """
        from auxiliary import get_model
        self.model = get_model(model_type = model_type, cl_dset = cl_dset, device = self.device)
        if load:
            assert {"load_bpath", "path", "suffix"}.issubset(load_attr.keys()), "Incorrect load parameters provided"
            self.model.load_state_dict(torch.load(os.path.join(load_attr["load_bpath"], load_attr["path"],
                                                  f"{self.model.__class__.__name__}_{load_attr['num_epochs']}_epochs_model_after_task{load_attr['task']}{load_attr['suffix']}"),
                                                  map_location = self.device)) #map location ensures it's mapped to the proper device
            self.task_no = load_attr['task']
            self.save_appendix = load_attr['suffix']
            self.num_epochs = load_attr['num_epochs']
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
                setprint = False if batch_num % 40 == 0 else False
                if setprint:
                    print(f"-------------------------------(Task {self.task_no}, epoch {epoch_num}, batch {batch_num})---------------------------------------")
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        if setprint:
                            # print(f"param {name}:")
                            # print(f"-----prev_fisher: {prev_fisher[name][:1]}")
                            # print(f"-----param: {param[:1]}")
                            # print(f"-----prev_param: {prev_params[name][:1]}")
                            pass
                        param_loss = prev_fisher[name] * (param - prev_params[name])**2 # '*' is elementwise multiplication and '**' is eltwise exponentiation. this is ok, no need for dot product (this is the dot product)
                        if setprint:
                            # print(f"param {name} loss: {param_loss}")
                            pass
                        if setprint:
                            print(f"before adding correction of param {name} we have batch loss {batch_loss_corrected}")
                        batch_loss_corrected += param_loss.sum()
                        if setprint:
                            print(f"after correction of param {name} we have batch loss {batch_loss_corrected}")
                if batch_num == 0:
                    print(f"len of dict (trainable) prev_params: {len(prev_params.keys())}; len of prev_fisher: {len(prev_fisher.keys())}; "
                          f"len of current dict of (trainable) model params: {len([1 for _,p in self.model.named_parameters() if p.requires_grad])}")
                if setprint:
                    """
                    This is useless because the assignment batch_loss_corrected = batch_loss does nothing: 
                    batch_loss will end up referencing the same value as batch_loss_corrected; 
                    if we want to do this, batch_loss needs to be detached when adding it to batch_loss_corrected, but that might mess up the computational graph, need to make sure the graph is retained!!
                    """
                    # print(f"(Batch {batch_num}, epoch {epoch_num}): regular batch loss {batch_loss}, corrected batch loss {batch_loss_corrected}. \n\n") #USELESS (AS IS NOW)

            epoch_loss += batch_loss.item()
            batch_loss_corrected.backward()

            # update parameters
            self.optimizer.step()

            with torch.no_grad():
              batch_acc, _ = self.get_accuracy(batch_logits, batch_labels)
              epoch_acc += batch_acc.item()

            if batch_num % 100 == 0:
                print(f"(epoch: {epoch_num}, batch: {batch_num + 1}), batch loss = {batch_loss.item()}")

        epoch_loss /= (batch_num + 1)
        epoch_acc /= (batch_num + 1)
        print(f"epoch {epoch_num} has overall loss {epoch_loss}")

        fisher_diag = None
        """
        COMPUTING FISHER DIAGONAL
        """
        if compute_new_fisher:
            print(f"Calculating Fisher for task {self.task_no} (at end of epoch {epoch_num}). \n")
            for batch_num, (batch_imgs, _) in enumerate(train_loader):
                batch_logits = self.model(batch_imgs.to(self.device))
                batch_probs = torch.log(torch.nn.functional.softmax(batch_logits, dim=1))
                for sample_idx in range(batch_probs.shape[0]):
                    for label_idx in range(batch_probs.shape[-1]):
                        self.optimizer.zero_grad()
                        torch.autograd.backward(tensors = batch_probs[sample_idx, label_idx], create_graph = True)
                        # print(f"When calculating new fisher matrix, batch_grad of prob for sample {sample_idx} label {label_idx} is {grads[0]}")
                        if fisher_diag is None:
                            fisher_diag = dict()
                            ###ADD A DETACH
                            for name, param in self.model.named_parameters():
                                if param.requires_grad:
                                    fisher_diag[name] = torch.pow(param.grad.detach().clone(), 2)
                                    fisher_diag[name].requires_grad = False
                        else:
                            for name, param in self.model.named_parameters():
                                if param.requires_grad:
                                    fisher_diag[name] += torch.pow(param.grad.detach().clone(), 2)
                    for name in fisher_diag.keys():
                        fisher_diag[name] /= batch_probs.shape[-1]
                for name in fisher_diag.keys():
                    fisher_diag[name] /= batch_probs.shape[0]
            for name in fisher_diag.keys():
                fisher_diag[name] /= batch_num

        return epoch_loss, epoch_acc, fisher_diag

    def test_epoch(self, test_loader):
        """ Compute the loss on the test set
        :param
        epoch_num: number of current epoch
        """
        epoch_loss = 0.0
        epoch_acc = 0.0
        epoch_confusion = None
        all_ground_labels = None
        all_pred_probs = None
        loss = torch.nn.CrossEntropyLoss()
        self.model.eval()
        with torch.no_grad():
            for batch_num, (batch_imgs, batch_labels) in enumerate(test_loader):
                batch_imgs, batch_labels = batch_imgs.to(self.device), batch_labels.to(self.device)

                #saving probabilities for entropy
                if all_ground_labels is None:
                    all_ground_labels = batch_labels
                else:
                    all_ground_labels = torch.cat((all_ground_labels, batch_labels), dim = 0)
                batch_logits = self.model(batch_imgs)
                batch_probs = torch.nn.functional.softmax(input = batch_logits, dim = -1)
                if all_pred_probs is None:
                    all_pred_probs = batch_probs
                else:
                    all_pred_probs = torch.cat((all_pred_probs, batch_probs), dim = 0)
                batch_loss = loss(batch_logits, batch_labels)

                batch_acc, batch_confusion = self.get_accuracy(batch_logits, batch_labels)
                epoch_loss += batch_loss.item()
                epoch_acc += batch_acc.item()
                if epoch_confusion is None:
                    epoch_confusion = batch_confusion
                else:
                    epoch_confusion += batch_confusion
            epoch_loss /= (batch_num + 1)
            epoch_acc /= (batch_num + 1)
            #dealing with entropy; want entropy matrix, e[i,j] = average confidence of class i when it was classified as class j
            entr_matrix = [[ [] for _ in range(all_pred_probs.shape[-1])] for _ in range(all_pred_probs.shape[-1])]
            from scipy.stats import entropy
            for i, j in zip(range(all_pred_probs.shape[-1]), range(all_pred_probs.shape[-1]) ):
                entr_matrix[i][j] = [all_pred_probs[[k for k in range(all_pred_probs.shape[0]) if torch.argmax(all_pred_probs[k, :])==j], i].cpu()] #probs for all samples on class i
        # print(f"an entropy matrix {entr_matrix}") #PROBLEM
        return epoch_loss, epoch_acc, epoch_confusion, entr_matrix

    def get_accuracy(self, predictions, labels):
        """
        predictions are of the shape no_imgs x no_classes, representing probability of each img being in each class
        labels are of the same shape only binary
        average = 'weighted' is used in f1_score because it accounts for possible class imbalances
        """
        argmax_preds = torch.argmax(predictions, dim=1).type(torch.FloatTensor)
        assert argmax_preds.shape == labels.shape, "Predictions and labels don't have same shape: "+{argmax_preds.shape}+" vs. "+{labels.shape}+" respectively."
        from sklearn.metrics import f1_score
        score = f1_score(y_true=labels.cpu(), y_pred=argmax_preds.cpu(), average='weighted')
        from sklearn.metrics import confusion_matrix
        conf_matr = confusion_matrix(y_true = labels.cpu(), y_pred = argmax_preds.cpu(), labels = range(predictions.shape[-1]))
        return score, conf_matr

    def get_all_task_averages(self, taskwise_averages, test_loaders):
        """
        :param taskwise_averages: a list of lists; each list corresponds to a task (starting from 0);
                        each element of a list corresp to a task is accuracy on that task at epoch given by index (in the nested list) when training on latest task
        :param test_loaders:
        :return: a list of the length the number of current epochs, with average of accuracies at each epoch on all tasks that have been seen (weighted average with weights given by the number of images in each loader)
        """
        import numpy as np
        weights = [len(loader) for loader in test_loaders]
        no_epochs = len(taskwise_averages[0])
        all_task_avgs = [np.average([taskwise_averages[task][epoch] for task in range(len(taskwise_averages))], weights = weights) for epoch in range(no_epochs)]
        return all_task_avgs

    def train(self, train_loaders, test_loaders, ewc = False, prev_fisher = None, prev_params = None):
        """
        Expects one or multiple train loaders corresponding to specific tasks in a list
        Assumes the last train loader & test loader correspond to current task and that we've done the training on previous tasks before.
        At the end of each epoch of training on current task also retests perfomance on each previous task.
        """
        train_loader, test_loader = train_loaders[-1], test_loaders[-1]
        train_losses, test_losses, train_acc, test_acc = [[] for i in range(len(train_loaders))], [[] for i in range(len(train_loaders))], [[] for i in range(len(train_loaders))], [[] for i in range(len(train_loaders))]
        test_confusions = [[] for _ in range(len(train_loaders))]
        test_entropies = [[] for _ in range(len(train_loaders))]
        if True:
            for epoch in range(self.num_epochs):
                print(f"learning rate at epoch {epoch}: {[g['lr'] for g in self.optimizer.param_groups]}")
                epoch_train_loss, epoch_train_acc, fisher_diag = \
                    self.train_epoch(train_loader, epoch, compute_new_fisher = True if epoch == self.num_epochs-1 and ewc else False, prev_fisher = prev_fisher, prev_params = prev_params)
                epoch_test_loss, epoch_test_acc, epoch_test_confusion, epoch_test_entropy = self.test_epoch(test_loader)
                train_losses[-1].append(epoch_train_loss)
                test_losses[-1].append(epoch_test_loss)
                train_acc[-1].append(epoch_train_acc)
                test_acc[-1].append(epoch_test_acc)
                test_confusions[-1].append(epoch_test_confusion)
                test_entropies[-1].append(epoch_test_entropy)

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
                    prev_epoch_train_loss, prev_epoch_train_acc, _, _ = self.test_epoch(train_loader)
                    prev_epoch_test_loss, prev_epoch_test_acc, prev_epoch_test_confusion, prev_epoch_test_entropy = self.test_epoch(test_loader)
                    train_losses[prev_task_no].append(prev_epoch_train_loss)
                    test_losses[prev_task_no].append(prev_epoch_test_loss)
                    train_acc[prev_task_no].append(prev_epoch_train_acc)
                    test_acc[prev_task_no].append(prev_epoch_test_acc)
                    test_confusions[prev_task_no].append(prev_epoch_test_confusion)
                    test_entropies[prev_task_no].append(prev_epoch_test_entropy)

                print(f"At end of (task {self.task_no}, epoch {epoch}) accuracy average on all tasks is: {self.get_all_task_averages(test_acc, test_loaders)}")
                #SAVE
                if self.save_interm:
                    metrics_save = {"train_losses": train_losses,
                                        "test_losses": test_losses,
                                        "train_acc": train_acc,
                                        "test_acc": test_acc,
                                        "all_task_averages": self.get_all_task_averages(test_acc, test_loaders),
                                        "test_confusions": test_confusions,
                                        "test_entropies": test_entropies}
                    self.save_metrics(metrics_save, epoch)
                    if epoch == self.num_epochs-1:
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
                "test_acc": test_acc,
                "all_task_averages": self.get_all_task_averages(test_acc, test_loaders),
                "test_confusion": test_confusions,
                "test_entropies": test_entropies}, fisher_diag