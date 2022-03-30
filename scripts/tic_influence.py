import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mfe.from_txt import msi_from_txt
from mfe.peak_picking import de_flatten

if __name__ == "__main__":
    spectra0 = msi_from_txt("./raw_data/SBB0-5cm_mz375-525.txt")
    tic0 = dict()
    for key in spectra0.keys():
        tic0[key] = spectra0[key].tic
    tic0 = pd.DataFrame.from_dict(tic0, orient='index')
    tic0 = tic0.reset_index()
    tic0['x'] = tic0['index'].apply(lambda x: x[0])
    tic0['y'] = tic0['index'].apply(lambda x: x[1])
    tic0 = tic0.drop(columns='index')

    spectra1 = msi_from_txt("./raw_data/SBB5-10cm_mz375-525.txt")
    tic1 = dict()
    for key in spectra1.keys():
        tic1[key] = spectra1[key].tic
    tic1 = pd.DataFrame.from_dict(tic1, orient='index')
    tic1 = tic1.reset_index()
    tic1['x'] = tic1['index'].apply(lambda x: x[0])
    tic1['y'] = tic1['index'].apply(lambda x: x[1])
    tic1 = tic1.drop(columns='index')


    im0 = de_flatten(tic0[['x', 'y']].abs().to_numpy(),
                     tic0.drop(columns=['x', 'y']).to_numpy().flatten(),
                     stretch_contrast=True,
                     interpolation='None'
                     )
    im1 = de_flatten(tic1[['x', 'y']].abs().to_numpy(),
                     tic1.drop(columns=['x', 'y']).to_numpy().flatten(),
                     stretch_contrast=True,
                     interpolation='None'
                     )
    if im0.shape[1] < im1.shape[1]:
        missing_width = im1.shape[1] - im0.shape[1]
        im0 = np.c_[im0, np.zeros([im0.shape[0], missing_width])]
    elif im0.shape[1] > im1.shape[1]:
        missing_width = im0.shape[1] - im1.shape[1]
        im1 = np.c_[im1, np.zeros([im1.shape[0], missing_width])]
    im1 = np.fliplr(im1)
    im = np.r_[im0, im1]
    im = im.T
    fig, ax = plt.subplots()
    ax.imshow(im)
    ax.axis('off')
    plt.savefig('b_tic.svg', format='svg')
    plt.show()