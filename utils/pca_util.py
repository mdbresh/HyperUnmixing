from sklearn.decomposition import PCA

def get_PC(im, show_plots=True, top_n=3, PC_n=1, top_load_n=1, figsize=(8,9)):
    
    """
    get_PC(im)

    Returns numpy.ndarray of loading scores for each PC (row) and each feature (column)
    Principal Component Analysis (PCA) gives the significant features for dimentionality reduction

    Parameters
    ----------
    im : image passed as numpy array
    show_plots : True shows plots for analysis, otherwise only loading scores returned

    Arguements if show_plots = True:
    _
    top_n : Number of top PCs to plot
    PC_n : nth PC to show the loading scores
    top_load_n : Top loading scores of PC_n th PC to be shown in analysis
    figsize : figsize for plots

    Returns
    -------
    out : ndarray
        An array object satisfying the specified requirements.

    """
    
    #For PCA, each row should be a data point, columns are features
    data = np.reshape(im, (im.shape[0]*im.shape[1], im.shape[2])) #reshaping image - independent pixels
    feat_arr = np.arange(750, 750+im.shape[2], 1) #array of features

    pca = PCA() #define PCA object
    _ = pca.fit(data) #fit PCA

    scree_values = np.round(pca.explained_variance_ratio_, decimals=5) #Gives scree values array for PCs
    loading_scores = pca.components_ #Loading scores for each feature and each PC

    #Getting top top_load_n number of features in PC_n
    top_inds = np.argsort(loading_scores[PC_n - 1])[-top_load_n:]
    top_feat, top_scores = feat_arr[top_inds], loading_scores[PC_n - 1, top_inds]
    #For plots : labelling PCs
    PC_names = ["PC-"+ str(i) for i in np.arange(1,scree_values.shape[0]+1,1)]

    #SCREE PLOT : Explained Var./Unexplained Var. ------------------------------------------
    fig, ax = plt.subplots(nrows = 2, figsize=figsize)
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

    return loading_scores