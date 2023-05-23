import os.path

import torch


class Autoencoder(object):
    """
    ?
    """

    def __init__(
            self, model,
            checkpoint,
            device,
            optimizer,
            criterion,
            lr_scheduler=None
    ):

        self.model = model.to(device)
        if checkpoint is not None:
            checkpoint_path = os.path.join('learners', checkpoint)
            self.model.load_state_dict(torch.load(checkpoint_path))

        # x = torch.load('../pytorch-AE/data.pt')
        # y = torch.load('../pytorch-AE/label.pt')
        # rep = self.model.encode(x)
        # recon = self.model(x)
        # batch_size = x.size(0)
        # x_recon = model(x)
        # criterion = torch.nn.BCELoss(reduction='none')
        # recon_loss = criterion(x_recon, x.view(batch_size, -1)).sum(dim=1)

        self.device = device
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.criterion = criterion
        self.model_dim = int(self.get_param_tensor().shape[0])


    def optimizer_step(self):
        """
         perform one optimizer step, requires the gradients to be already computed.
        """
        self.optimizer.step()
        if self.lr_scheduler:
            self.lr_scheduler.step()

    def get_param_tensor(self):
        """
        get `model` parameters as a unique flattened tensor

        :return: torch.tensor

        """
        param_list = []

        for param in self.model.parameters():
            param_list.append(param.data.view(-1, ))

        return torch.cat(param_list)

    def get_grad_tensor(self):
        """
        get `model` gradients as a unique flattened tensor

        :return: torch.tensor

        """
        grad_list = []

        for param in self.model.parameters():
            if param.grad is not None:
                grad_list.append(param.grad.data.view(-1, ))

        return torch.cat(grad_list)

    def free_memory(self):
        """
        free the memory allocated by the model weights

        """
        del self.optimizer
        del self.model

    def free_gradients(self):
        """
        free memory allocated by gradients

        """
        self.optimizer.zero_grad(set_to_none=True)

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad = True
