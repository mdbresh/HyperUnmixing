# HyperUnmixing

Photo-induced force microscopy is a tool often used to visualize a variety of materials and how those materials' topographies, absorptions, and emissions behave under different excitation wavelengths.

Analysis of PiFM images relies on major assumptions:

- The ratio of the prepared solutions or samples is maintained post-preparation and during measurement
- Spectra of individual components can be parsed out of a spectrum of the mixed sample

A variety of techniques have been used to probe these assumptions, including Principal Component Analaysis and Principal Component Regression. These techniques allow us to better understand if intended ratios are maintained, how well mixed samples behave, and how sample preparation or sample interactions might affect spectra.

The goal of this project is to perform unmixing of hyperspectral photo-induced force microscopy images of composite or mixed polymer films with distinct chemical signatures. The major inspiration for this is [Kong *et al.*](https://pubs.acs.org/doi/10.1021/acs.jpclett.8b01003)

We implemented Non-negative Matrix Factorization (NMF) to accomplish this goal. NMF decomposes the data cube into its weighted coefficient matrices and component spectra. Given a data cube or matrix X, NMF supposes that:

X = WH

Given the W and H matrices (the weighted coefficients and component spectra, respectively), we have developed several functions and tools to aid in the processing, analysis, and visualization of how NMF spectrally unmixes PiFM data cubes.

In order to use these tools, feel free to explore the given Jupyter notebooks (including a final workthrough notebook entitled 20200602_final_presentation.ipynb). We hope to develop these modules into Classes with methods at a later date.
