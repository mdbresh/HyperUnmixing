from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from .img_util import avg_spectra

def normalize(data):
    """ Normalizes data by shifting origin to mean"""
    orig_mean = np.mean(data, axis=0)
    norm_data = data - orig_mean
    
    return norm_data

def get_PC(im, show_plots=True, top_n=3, PC_n=1, top_load_n=1, figsize=(8,9)):
    
    """
    get_PC(im)

    Returns numpy.ndarray of loading scores for each PC (row) and each feature (column)
    Also, returns the scree values (Variance shares) for each PC.
    Principal Component Analysis (PCA) gives the significant features for dimentionality reduction

    Parameters
    ----------
    im : image passed as numpy array

    Returns
    -------
    out : tuple
        A tuple of loading scores, scree values and the original mean of data points.
        This mean of data points is a mean spectrum.

    """
    
    #For PCA, each row should be a data point, columns are features
    data = np.reshape(im, (im.shape[0]*im.shape[1], im.shape[2])) #reshaping image - independent pixels
    data = normalize(data)

    pca = PCA() #define PCA object
    _ = pca.fit(data) #fit PCA

    scree_values = np.round(pca.explained_variance_ratio_, decimals=5) #Gives scree values array for PCs
    loading_scores = pca.components_ #Loading scores for each feature and each PC

    return (loading_scores, scree_values)


def plot_PCs(ax, loading_scores, scree_values, top_n=3, PC_n=1, top_load_n=1):
    
    """
    Updates plt.fig.axes() objects with PCA plots.
    ax must be an array of 2 axes objects.
    
    Parameters
    -----------------
    ax : array of 2 plt.fig.axes() objects
    loading_scores : An array of loading scores (rows are loading scores of each PC and columns are PCs)
    top_n : Number of top PCs to plot
    PC_n : nth PC to show the loading scores
    top_load_n : Top loading scores of PC_n th PC to be shown in analysis
    
    Returns
    ----------------
    out : array
        Updated array of axes
    
    """
    
    
    feat_arr = np.arange(750, 750+scree_values.shape[0], 1) #array of features
    
    #Getting top top_load_n number of features in PC_n
    top_inds = np.argsort(loading_scores[PC_n - 1])[-top_load_n:]
    top_feat, top_scores = feat_arr[top_inds], loading_scores[PC_n - 1, top_inds]
    #For plots : labelling PCs
    PC_names = ["PC-"+ str(i) for i in np.arange(1,scree_values.shape[0]+1,1)]

    #SCREE PLOT : Explained Var./Unexplained Var. ------------------------------------------
    ax[0].bar(np.arange(1,top_n+1,1), scree_values[:top_n])
    ax[0].set_xticks(np.arange(1,top_n+1,1))
    ax[0].set_xticklabels(PC_names[:top_n])
    ax[0].set_title('Variance/Total Explained Variance')

    txt = [str(i) for i in scree_values[:top_n]] #Annotate bar plots
    for i, txt_i in enumerate(txt):
        ax[0].text(i+0.85, float(txt_i), txt_i, fontsize = 10, color = 'black')

    #Single PC analysis : plotting loading scores ------------------------------------------
    #Change PC_n to get the corresponding PCs loading scores plots.
    ax[1].set_title('Abs(Loading scores) in PC-{}'.format(PC_n))
    ax[1].plot(feat_arr, np.abs(loading_scores[PC_n - 1]))
    ax[1].grid(color='gray')
    
    for x,y in zip(top_feat, top_scores):
        ax[1].plot([x,x], [0,y], linestyle='dashed')
        ax[1].scatter(x,y, marker='o', c='yellow', s=200, edgecolors='red')

    for i, txt_i in enumerate(top_scores): #Annotate the top features
        ax[1].text(top_feat[i], txt_i, str(round(txt_i,1)), fontsize = 8.5, color = 'black')
    
    return ax
    


def make_PC_images(im_x, loading_scores, PC_num=[1]):
    """
    Makes single feature using loading scores of PC_num^th PC, by linear combination of features in im_x

    Parameters
    ----------
    im_x : image passed as numpy array
    loading_scores : numpy array with ith row should have loading scores of ith PC.
    PC_num : if PC_num = n, then nth PC's loading scores will be used to calculate the new feature

    Returns
    -------
    out : ndarray
        A new x array, with PC as feature in a single column

    """
    mean_spectra = avg_spectra(im_x)
    new_im_x = np.reshape(np.dot(im_x-mean_spectra, loading_scores[PC_num[0]-1]),(-1,1))
    if len(PC_num)>1:
        for PC in PC_num[1:]:
            new_im_x = np.hstack([new_im_x, np.reshape(np.dot(im_x-mean_spectra, loading_scores[PC-1]),(-1,1))])

    return np.reshape( new_im_x, (im_x.shape[0], im_x.shape[1], len(PC_num)) )