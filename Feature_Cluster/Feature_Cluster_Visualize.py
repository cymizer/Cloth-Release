import numpy as np
import matplotlib.pyplot as plt
import configure as cfg
from sklearn.cluster import KMeans
import joblib
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.manifold import TSNE
import seaborn as sns


def feature_cluster(features, clusters=cfg.Clusters, save=True):
    """
    concatenate three types features with PCA  as  inputs
    inputs : features
    outputs : cluster module
    """

    kmeans = KMeans(n_clusters=clusters, random_state=2, n_jobs=-1, n_init=30)
    kmeans.fit(features)
    if save:
        joblib.dump(kmeans, f"{cfg.Model_Root}/{clusters}-means.pkl")
    return kmeans


def feature_visualize(features, kmeans,draw =True):
    

    # Silhouette analysis

    # Get silhouette samples
    silhouette_vals = silhouette_samples(features, kmeans.labels_)
    # Silhouette plot
    fig2 = plt.figure()
    ax3 = fig2.add_subplot(121)
    y_lower, y_upper = 0, 0
    for i, cluster in enumerate(np.unique(kmeans.labels_)):
        cluster_silhouette_vals = silhouette_vals[kmeans.labels_ == cluster]
        cluster_silhouette_vals.sort()
        y_upper += len(cluster_silhouette_vals)
        ax3.barh(range(y_lower, y_upper), cluster_silhouette_vals,
                 edgecolor='none', height=1, color=sns.xkcd_rgb[cfg.Color[cluster]])
        ax3.text(-0.03, (y_lower + y_upper) / 2, str(i + 1))
        y_lower += len(cluster_silhouette_vals)
    # Get the average silhouette score and plot it
    avg_score = np.mean(silhouette_vals)
    ax3.axvline(avg_score, linestyle='--', linewidth=2, color='red')
    ax3.set_yticks([])
    ax3.set_xlim([-0.1, 1])
    ax3.set_xlabel('Silhouette coefficient values')
    ax3.set_ylabel('Cluster labels')
    ax3.set_title('Silhouette plot for the various clusters', y=1.02)
    print('Average silhouette score = %f' % avg_score)

    tmp = np.concatenate((kmeans.cluster_centers_, features), axis=0)
    del features
    tsne_fv = TSNE(n_components=2).fit_transform(tmp)
    del tmp
    tsne_fv_cen = tsne_fv[0:cfg.Clusters, :]
    tsne_fv = tsne_fv[cfg.Clusters:, :]
    ax4 = fig2.add_subplot(122)
    ax4.set_title('fisher vector clustering visualization')
    for i in range(0, cfg.Clusters):
        index = np.argwhere(kmeans.labels_ == i)
        ax4.scatter(tsne_fv[index[:], 0], tsne_fv[index[:], 1],
                    c=sns.xkcd_rgb[cfg.Color[i]], s=20, marker='.', label=i)
        ax4.scatter(tsne_fv_cen[i, 0], tsne_fv_cen[i, 1], c=sns.xkcd_rgb[cfg.Color[i]],
                    s=100, alpha=0.5, marker='*', label=i, edgecolors='black')
    ax4.legend(loc=(1, 0), ncol=3, fontsize=8)
    fig2.set_size_inches(18.5, 10.5)
    fig2.savefig(f'{cfg.Result_save}/{cfg.Clusters}_clusters_result.png',format='png')
    if draw:
        plt.show()


if __name__ == "__main__":
    m = np.load(
        'Z:/Users/cymb103u/Desktop/WorkSpace/Hipster/feature/all_pca_m.npy')
    t = np.load(
        'Z:/Users/cymb103u/Desktop/WorkSpace/Hipster/feature/all_pca_t.npy')
    c = np.load(
        'Z:/Users/cymb103u/Desktop/WorkSpace/Hipster/feature/all_pca_c.npy')
    features = np.concatenate((m, t, c), axis=1)

    KMS = feature_cluster(features, clusters=3)
    label = KMS.labels_
    print(KMS.labels_)
