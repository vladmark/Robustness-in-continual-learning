""""###Trainer class """
import torch
import matplotlib.pyplot as plt

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
    plt.show()


class Trainer:
    def __init__(self, model: torch.nn.Module,
                 hyperparams: dict):
        self.device = hyperparams['device']
        self.model = model.to(self.device)
        if hyperparams['learning_algo'] == 'adam':
            self.optimizer = torch.optim.Adam(params = self.model.parameters(),
                                  lr = hyperparams['learning_rate'], weight_decay = hyperparams['weight_decay'])
        else:
            self.optimizer = torch.optim.SGD(params = self.model.parameters(),
                                 lr = hyperparams['learning_rate'], weight_decay = hyperparams['weight_decay'])
        self.num_epochs = hyperparams['num_epochs']
        self.lr = hyperparams['learning_rate']
        self.batch_size = hyperparams['batch_size']

    def train_epoch(self, train_loader, epoch_num: int) -> float:
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
            batch_imgs, batch_labels = batch_imgs.to(self.device), batch_labels.to(self.device)
            # reset gradients
            self.optimizer.zero_grad()
            batch_logits=self.model(batch_imgs)
            # if batch_num % 100 == 0:
            #   print(f"batch {batch_num} first 5 logits: {batch_logits[:5]}")
            #   print(f"batch {batch_num} first 5 labels: {batch_labels[:5]}")
            # print(f"predicted labels {torch.argmax(batch_logits, dim=1)}")
            batch_loss = loss(batch_logits, batch_labels)
            epoch_loss += batch_loss.item()
            batch_loss.backward()

            # update parameters
            self.optimizer.step()

            with torch.no_grad():
              batch_acc = self.get_accuracy(batch_logits, batch_labels)
              epoch_acc += batch_acc.item()

            if batch_num % 100 == 0:
                print(f"(epoch: {epoch_num}, batch: {batch_num + 1}), batch loss = {batch_loss.item()}")
        print(f"epoch {epoch_num} has overall loss {epoch_loss}")
        epoch_loss /= (batch_num + 1)
        epoch_acc /= (batch_num + 1)
        return epoch_loss, epoch_acc

    def test_epoch(self, test_loader) -> float:
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

    def train(self, train_loaders, test_loaders) -> dict:
        """
        Expects one or multiple train loaders corresponding to specific tasks in a list
        Assumes the last train loader & test loader correspond to current task and that we've done the training on previous tasks before.
        At the end of each epoch of training on current task also retests perfomance on each previous task.
        """
        train_loader, test_loader = train_loaders[-1], test_loaders[-1]
        train_losses, test_losses, train_acc, test_acc = [[] for i in range(len(train_loaders))], [[] for i in range(len(train_loaders))], [[] for i in range(len(train_loaders))], [[] for i in range(len(train_loaders))]

        #TEST PLOTTING TEST PLOTTING
        # for task_no in range(len(train_loaders)):
        #     fig = plt.figure()
        #     axes = fig.subplots(2, 5)
        #     it = iter(train_loaders[task_no])
        #     imgs, labels = next(it)
        #     for i in range(10):
        #         axes[i // 5, i % 5].set_title(f"label: {labels[i]}")
        #         axes[i // 5, i % 5].imshow(imgs[i].permute(1,2,0))
        #     fig.set_figheight(5)
        #     fig.set_figwidth(5)
        #     fig.suptitle(f"Train loader of task {task_no} corresponding to training on task {len(train_loaders)}")
        #     fig.tight_layout()
        #     fig.show()
        # TEST PLOTTING TEST PLOTTING
        if True:
            for epoch in range(self.num_epochs):
                print(f"learning rate at epoch {epoch}: {[g['lr'] for g in self.optimizer.param_groups]}")
                epoch_train_loss, epoch_train_acc = self.train_epoch(train_loader, epoch)
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
                  for task_no in range(len(train_loaders)-1):
                    train_loader, test_loader = train_loaders[task_no], test_loaders[task_no]
                    epoch_train_loss, epoch_train_acc = self.test_epoch(train_loader)
                    epoch_test_loss, epoch_test_acc = self.test_epoch(test_loader)
                    train_losses[task_no].append(epoch_train_loss)
                    test_losses[task_no].append(epoch_test_loss)
                    train_acc[task_no].append(epoch_train_acc)
                    test_acc[task_no].append(epoch_test_acc)

                #DO PRINTS AND PLOTS
                if epoch % 3 == 0 and epoch > 0:
                  interm_plot_metrics = {"train_losses": train_losses,
                      "test_losses": test_losses,
                      "train_acc": train_acc,
                      "test_acc": test_acc}
                  plot_metrics(metrics = interm_plot_metrics, task_no = -1, title = f"Metrics after {epoch} epochs with learning rate {self.lr} for model {self.model.__class__.__name__}.")
                  if len(train_loaders) > 1:
                    for task_no in range(len(train_loaders)-1):
                      plot_metrics(metrics = interm_plot_metrics, task_no = task_no, title = f"Same as before, revisiting task {task_no}.")
        print(f"On training task {len(train_loaders)} task 0 was retested on {len(train_losses[0])}")
        return {"train_losses": train_losses,
                "test_losses": test_losses,
                "train_acc": train_acc,
                "test_acc": test_acc}
