import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import torch
import os
import utils
import networks
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision
import torchvision.transforms as TF
from tensorboardX import SummaryWriter
from losses import SupervisedLoss

class Model():
    def __init__(self, config_path):
        self.config_path = config_path
        config = utils.load_yaml(config_path)
        self.config = config
        self.device = torch.device("cuda" if config["use_cuda"] else "cpu")
        self.model = self.build_model()


    def build_model(self):
        """
        Sets model to be whatever neural network we want with specified
        num_classes and whether or not we want to use pretrained imagenet weights
        """

        network_name = self.config["network_name"]
        num_classes = self.config["num_classes"]
        pretrained = self.config["pretrained_imagenet"]

        if "resnet" in network_name:
            num_layers = int(network_name.split("resnet")[1])
            self.model = networks.resnet(num_layers, num_classes, pretrained)
        else:
            raise RuntimeError("%s is an invalid network_name" % network_name)

        model = self.model.to(self.device)
        return model

    def build_dataloaders(self):
        dataset_name = self.config["dataset_name"]
        dataset_path = self.config["dataset_path"]
        batch_size = self.config["batch_size"]
        num_workers = self.config["num_workers"]

        dataset_dict = {
            "cifar10": torchvision.datasets.CIFAR10
        }

        print("Building Dataloaders...")
        train_dataset = dataset_dict[dataset_name](root=dataset_path, train=True,
                                                   transform=TF.ToTensor(), download=True)
        val_dataset = dataset_dict[dataset_name](root=dataset_path, train=False,
                                                   transform=TF.ToTensor(), download=True)
        class_names = train_dataset.classes
        train_loader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  pin_memory=True)
        val_loader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=num_workers,
                                pin_memory=True)
        return train_loader, val_loader, class_names
        

    def set_mode(self, mode):
        if mode == "train":
            self.is_train = True
        elif mode == "eval":
            self.is_train = False
        else:
            raise RuntimeError("Invalid mode %s" % mode)
        if mode == "train":
            self.model.train()
        else:
            self.model.eval()

    def log_time(self, duration, loss):
        """Print a logging statement to the terminal
        """
        batch_size = self.config["batch_size"]
        samples_per_sec = batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, self.batch_idx, samples_per_sec, loss,
                                  utils.sec_to_dhm_str(time_sofar),
                                  utils.sec_to_dhm_str(training_time_left)))

    def log(self, images, y_hat, y, loss, mode):
        """
        Log to tensorbard
        """
        writer = self.writers[mode]        

        writer.add_scalar("Loss", loss, self.step)

        # Get predictions and gt class names as strings
        preds = self.logit2class(y_hat, is_gt=False)
        targets = self.logit2class(y, is_gt=True)

        for i in range(min(4, images.shape[0])):
            image = images[i]
            pred = preds[i]
            target = targets[i]
            image = image.permute(1,2,0).detach().cpu().numpy()
            image *= 255
            image = image.astype(np.uint8)
            # Need to do this conversion, otherwise can't use cv2.putText
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            height,width = image.shape[0],image.shape[1]
            # Need to enlarge the image; otherwise we can't fit text on image
            image = cv2.resize(image, (width*5,height*5))
            height,width = image.shape[0],image.shape[1]
            text = "pred: {} | gt: {}".format(pred, target)
            y_loc = int(0.05*height) + 1
            x_loc = int(0.05*width) + 1
            cv2.putText(image, text, org=(x_loc, y_loc), fontFace=1, fontScale=0.5, color=(0,255,0), thickness=1)
            # Don't forget to change back to RGB so image is displayed properly on tensorboard
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = torch.from_numpy(image).permute(2,0,1)
            writer.add_image("{}/color_{}".format(mode, i), image, self.step)

    def logit2class(self, y, is_gt=False):
        # Class with largest positive logit value is chosen as predicted class
        if not is_gt:
            indexes = torch.argmax(y, dim=1)
        else:
            indexes = y
        class_names = [self.class_names[index.item()] for index in indexes]
        return class_names


    def run_epoch(self):

        for batch_idx, (image, y) in enumerate(self.train_loader): 
            image = image.to(self.device)
            y = y.to(self.device)

            start_time = time.time()
            self.batch_idx = batch_idx

            # Feed-forward image through neural net
            y_hat = self.model.forward(image)
            # Compute the loss
            loss = self.loss_fn.forward(y_hat, y)
            # Perform optimization step; backpropagation
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            duration = time.time() - start_time

            log_frequency = self.config["log_frequency"]
            batch_size = self.config["batch_size"]

            # How many steps to wait before switching from frequent tensorboard
            # logging to less frequent tensorboard logging
            # set to how many ever steps are in one epoch
            step_cutoff = len(self.train_loader)

            early_phase = self.step % log_frequency == 0 and self.step < step_cutoff
            # Once past first epoch, only log every epoch
            late_phase = self.step % step_cutoff == 0

            if early_phase or late_phase:
                # Log time, tensorboard, and evaluate on validation set
                self.log_time(duration, loss)
                self.log(image, y_hat, y, loss, mode="train")
                self.val()

            self.step += 1


    def train(self):
        self.set_mode("train")
        # if self.config["load_weights_folder"] is not None:
        #     self.load_model()


        log_dir = self.config["log_dir"]
        dataset_name = self.config["dataset_name"]
        model_name = self.config["model_name"]
        batch_size = self.config["batch_size"]

        self.log_path = os.path.join(log_dir, dataset_name, model_name)
        print("log path: ", self.log_path)
        if os.path.exists(self.log_path):
            print("A model already exists at this path %s\n. Refusing to overwrite "
            "previously trained model. Exiting program..." % self.log_path)
            exit(1)
        models_dir = os.path.join(self.log_path, "models")
        os.makedirs(models_dir)
        # Save the yaml configuration file
        os.system("cp %s %s" % (self.config_path, models_dir))

        # Initialize tensorboard writers
        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        self.train_loader, self.val_loader, self.class_names = self.build_dataloaders()
        self.val_iter = iter(self.val_loader)
        print("Number of training examples: ", len(self.train_loader)*batch_size)
        print("Number of validation examples: ", len(self.val_loader)*batch_size)
        print("Number of steps per epoch: ", len(self.train_loader))

        weight_decay = self.config["weight_decay"]
        momentum = self.config["momentum"]
        lr = self.config["learning_rate"]
        scheduler_step_size = self.config["scheduler_step_size"]
        self.optim = torch.optim.SGD(self.model.parameters(), lr, momentum=momentum,
                                     weight_decay=weight_decay)
        # Use gamma as 0.1 as specified in paper
        # TODO: Do not know what scheduler step size the paper used...
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optim, scheduler_step_size, gamma=0.1)

        loss_dict = {
            "supervised": SupervisedLoss
        }

        # Define our loss function
        loss_type = self.config["loss_type"]
        self.loss_fn = loss_dict[loss_type](self.config)

        batch_size = self.config["batch_size"]
        num_epochs = self.config["num_epochs"]
        self.num_steps = batch_size * num_epochs

        # Keep track of number of training iterations we take
        self.step = 0

        self.start_time = time.time()

        for epoch in range(num_epochs):
            self.epoch = epoch
            self.run_epoch()
            self.lr_scheduler.step()


    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_mode("eval")
        try:
            image, y = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            image, y = self.val_iter.next()
        image = image.to(self.device)
        y = y.to(self.device)

        with torch.no_grad():
            y_hat = self.model.forward(image)
            loss = self.loss_fn.forward(y_hat, y)
            self.log(image, y_hat, y, loss, mode="val")

        self.set_mode("train")

    def inference(self):
        pass


if __name__ == "__main__":
    yaml_path = "configs/config.yaml"
    model = Model(yaml_path)
    model.train()