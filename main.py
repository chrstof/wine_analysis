__author__ = 'christof'

"""

Based on the wine data set from:

  P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
  Modeling wine preferences by data mining from physicochemical properties.
  In Decision Support Systems>, Elsevier, 47(4):547-553. ISSN: 0167-9236.

  Available at: [@Elsevier] http://dx.doi.org/10.1016/j.dss.2009.05.016
                [Pre-press (pdf)] http://www3.dsi.uminho.pt/pcortez/winequality09.pdf
                [bib] http://www3.dsi.uminho.pt/pcortez/dss09.bib
"""

#######################################################################
#############IMPORT DATA###############################################
#######################################################################

import pandas as pd
import numpy as np
import os.path
import io
import requests

d_name = 'winequality-white.csv'

# check if data is available localy, if not, download from github
if os.path.isfile('Data/' + d_name):
    print('Use local data.')
    df = pd.read_csv('Data/' + d_name, delimiter=';')
else:
    print('Use remote data.')
    url = 'https://raw.githubusercontent.com/chrstof/wine_analysis/master/Data/winequality-white.csv'
    s=requests.get(url).content
    df=pd.read_csv(io.StringIO(s.decode('utf-8')), delimiter=';')

#rename "sulfur dioxide" to "SO2"
df = df.rename(columns={'free sulfur dioxide': 'free SO2', 'total sulfur dioxide': 'total SO2'})


# sort the data acording to the quality
dfs = df.sort_values('quality', ascending=False)
quality_s = dfs.quality

# normalize data: take off mean, devide by standard deviation
dfs_norm = dfs.apply(lambda x: (x - np.mean(x)) / (np.std(x)))
dfs_norm = dfs_norm.drop('quality', 1)




#######################################################################
# ------------HISTOGRAMS-----------------------------------------------#
#######################################################################


import seaborn as sns
import matplotlib.pyplot as plt
from pylab import savefig

#sns.set()
#sns.pairplot(dfs, hue="quality")

g = sns.PairGrid(dfs_norm)
g.map_diag(sns.kdeplot)

g.map_offdiag(plt.scatter, c="k", s=1)
g.map_upper(sns.kdeplot, cmap="Blues_d", n_levels=10);
#g.map_upper(plt.hexbin)


g.set(xticks=[ -2, -1, 0,1,2], yticks=[-2, -1, 0,1,2]);
g.set(ylim=(-3, 3))
g.set(xlim=(-3, 3))

savefig('Pics/kde_scatterplot.png', bbox_inches='tight')


#######################################################################
# ------------CORRELATIONS--------------------------------------------#
#######################################################################


from string import letters
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="white")


# Compute the correlation matrix
corr = dfs.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
g = sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.1,square=True,
            linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)

locs, labels = plt.xticks()
plt.setp(labels, rotation=45)

g.set_yticklabels(rotation=30)
g.set_xticklabels(rotation=30)
plt.yticks(rotation=45)
plt.yticks(rotation=0)


#######################################################################
# ------------PCA-----------------#
#######################################################################

from sklearn.decomposition import PCA
X = dfs_norm.values

pca = PCA(n_components=11) # take first n components
#pca = PCA(n_components='mle')# take all significant components
#pca = PCA(n_components=0.9) # explains this amount of variance

pca.fit(X)
f = pca.fit_transform(X)

print(pca.explained_variance_ratio_)

pca.components_.shape
pca.explained_variance_ratio_


import numpy as np
U, s, V = np.linalg.svd(X, full_matrices=True)
U.shape, V.shape, s.shape

S = np.zeros((4898, 11))
S[:11, :11] = np.diag(s)
np.allclose(X, np.dot(U, np.dot(S, V)))







