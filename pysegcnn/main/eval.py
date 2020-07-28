# builtins
from __future__ import absolute_import
import os

# externals
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
import torch.nn.functional as F

# locals
from pysegcnn.core.dataset import StandardEoDataset
from pysegcnn.core.trainer import NetworkTrainer
from pysegcnn.core.utils import reconstruct_scene
from pysegcnn.core.graphics import (plot_confusion_matrix, plot_loss,
                                    plot_sample)
from pysegcnn.main.config import config


if __name__ == '__main__':

    # instanciate the NetworkTrainer class
    trainer = NetworkTrainer(config)
    trainer

    # load pretrained model
    state = trainer.model.load(trainer.state_file,
                               trainer.optimizer,
                               trainer.state_path)
    trainer.model.eval()

    if trainer.plot_cm:
        # predict each batch in the validation/test set
        cm, accuracy, loss = trainer.predict(pretrained=True, confusion=True)

        # plot confusion matrix: labels of the dataset
        labels = [label['label'] for label in trainer.dataset.labels.values()]
        plot_confusion_matrix(cm, labels, state=trainer.state_file)

    # plot loss and accuracy
    plot_loss(trainer.loss_state)

    # whether to plot the samples of the validation/test dataset
    if trainer.plot_samples:

        # base filename for each sample
        fname = trainer.state_file.split('.pt')[0]

        # set random seed for reproducibility
        np.random.seed(trainer.seed)

        # plot samples from the validation or test dataset
        dataset = trainer.test_ds if trainer.test else trainer.valid_ds
        dname = 'test' if trainer.test else 'val'

        # draw a number of samples from the validation/test set
        samples = np.arange(0, len(dataset))
        if trainer.nsamples > 0:
            samples = np.random.randint(len(dataset),
                                        size=min(trainer.nsamples,
                                                 len(dataset)))

        # iterate over the samples and plot inputs, ground truth and
        # model predictions
        for sample in samples:
            # a sample from the validation/test set
            inputs, labels = dataset[sample]

            # convert to net input shape
            net_inputs = torch.tensor(np.expand_dims(inputs, axis=0))

            # compute model predictions
            with torch.no_grad():
                y_pred = F.softmax(trainer.model(net_inputs),
                                   dim=1).argmax(dim=1).squeeze()

            # plot inputs, ground truth and model predictions
            sname = fname + '_{}_sample_{}.pt'.format(dname, sample)
            fig, ax = plot_sample(inputs.numpy().clip(0, 1),
                                  labels,
                                  trainer.dataset.use_bands,
                                  trainer.dataset.labels,
                                  y_pred=y_pred,
                                  bands=trainer.plot_bands,
                                  state=sname,
                                  stretch=True,
                                  alpha=5)

    # whether to plot the reconstructed scenes
    if trainer.plot_scenes:

        # base filename for each scene
        fname = trainer.state_file.split('.pt')[0]

        # only supported if the dataset is a StandardEoDataset
        if not isinstance(trainer.dataset, StandardEoDataset):
            print('Reconstruction of entire scenes only supported for '
                  'datasets of type {}. Aborting ...'
                  .format(StandardEoDataset.__name__))

        # get the names of the scenes
        scene_ids = [s for s in os.listdir(trainer.dataset.root) if
                     trainer.dataset.parse_scene_id(s) is not None]

        # spatial size of scene
        scene_size = (trainer.dataset.height, trainer.dataset.width)

        # iterate over the scenes
        for sid in scene_ids:

            # filename for the current scene
            sname = fname + '_' + sid + '.pt'

            # get the tiles of the scene
            tiles = trainer._get_scene_tiles(sid)
            tiles.sort(key=lambda k: k['idx'])

            # create a subset of the dataset
            scene_ds = Subset(trainer.dataset,
                              indices=[t['idx'] for t in tiles])

            # create the dataloader
            scene_dl = DataLoader(scene_ds, batch_size=len(scene_ds),
                                  shuffle=False)

            # predict the current scene
            for i, (inp, lab) in enumerate(scene_dl):
                print('Predicting scene: {}'.format(sid))

                # apply forward pass: model prediction
                with torch.no_grad():
                    prd = F.softmax(trainer.model(inp),
                                    dim=1).argmax(dim=1).squeeze()

            # reconstruct the entire scene
            inputs = reconstruct_scene(inp, scene_size, nbands=inp.shape[1])
            labels = reconstruct_scene(lab, scene_size, nbands=1)
            prdtcn = reconstruct_scene(prd, scene_size, nbands=1)

            # plot current scene
            fig, ax = plot_sample(inputs.clip(0, 1),
                                  labels,
                                  trainer.dataset.use_bands,
                                  trainer.dataset.labels,
                                  y_pred=prdtcn,
                                  bands=trainer.plot_bands,
                                  state=sname,
                                  stretch=True,
                                  alpha=5)
