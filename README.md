# HyperUnmixing
The goal of this project is to perform unmixing of hyperspectral photo-induced force microscopy images of composite or mixed polymer films with distinct chemical signatures. The major inspiration for this is [Kong *et al.*](https://pubs.acs.org/doi/10.1021/acs.jpclett.8b01003)

We plan to explore the following techniques to apply the unmixing algorithm to other polymer datasets:
  - Convultional Neural Networks
    |**Pros** |**Cons**  |
    |---|:---:|
    |Spatially correlated | Complex/new |
    |Other applications explored | Scary! |
  - Non-negative Matrix Factorization*


  *There are many varieties of NMF including, but not limited to:
  - Graph regularized NMF
  - Constrained NMF
  - Minimum volume constrained NMF*
  
Goal 1 : Replicate results from paper
  
  Try Principle Component Analysis to get important wavelengths
  
  ![PCA_sample_image](https://github.com/mdbresh/HyperUnmixing/blob/master/data/sample_pca.png)
  
  Each pixel has features like this :

  ![SAMPLE_SPECTRA](https://github.com/mdbresh/HyperUnmixing/blob/master/data/sample_spectra.png)
  
## Google Colab

Here is the our google colab page [HyperUnmixing](https://colab.research.google.com/drive/1mzOykuRUsAjyIfyvOFmxHKzTt-8aI0dI)
