import torch
import numpy as np
from maxflow.fastmin import aexpansion_grid
from sklearn.cluster import KMeans, DBSCAN
from sklearn import metrics
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE

def data_term(content_feature, cluster_centers):
    c = content_feature.permute(1, 2, 0)
    d = torch.matmul(c, cluster_centers)
    c_norm = torch.norm(c, dim=2, keepdim=True)
    s_norm = torch.norm(cluster_centers, dim=0, keepdim=True)
    norm = torch.matmul(c_norm, s_norm)
    d = 1 - d.div(norm)
    return d


def pairwise_term(cluster_centers, lam):
    _, k = cluster_centers.shape
    v = torch.ones((k, k)) - torch.eye(k)
    v = lam * v.to(cluster_centers.device)
    return v


def labeled_whiten_and_color(f_c, f_s, alpha, label):
    try:
        c, h, w = f_c.shape
        cf = (f_c * label).reshape(c, -1)
        c_mean = torch.mean(cf, 1).reshape(c, 1, 1) * label

        cf = cf.reshape(c, h, w) - c_mean
        cf = cf.reshape(c, -1)
        c_cov = torch.mm(cf, cf.t()).div(torch.sum(label).item() / c - 1)
        c_u, c_e, c_v = torch.svd(c_cov)

        # if necessary, use k-th largest eig-value
        k_c = c
        # for i in range(c):
        #     if c_e[i] < 0.00001:
        #         k_c = i
        #         break
        c_d = c_e[:k_c].pow(-0.5)

        w_step1 = torch.mm(c_v[:, :k_c], torch.diag(c_d))
        w_step2 = torch.mm(w_step1, (c_v[:, :k_c].t()))
        whitened = torch.mm(w_step2, cf)

        sf = f_s.t()
        c, k = sf.shape
        s_mean = torch.mean(sf, 1, keepdim=True)
        sf = sf - s_mean
        s_cov = torch.mm(sf, sf.t()).div(k - 1)
        s_u, s_e, s_v = torch.svd(s_cov)

        # if necessary, use k-th largest eig-value
        k_s = c
        # for i in range(c):
        #     if s_e[i] < 0.00001:
        #         k_s = i
        #         break
        s_d = s_e[:k_s].pow(0.5)

        c_step1 = torch.mm(s_v[:, :k_s], torch.diag(s_d))
        c_step2 = torch.mm(c_step1, s_v[:, :k_s].t())
        colored = torch.mm(c_step2, whitened).reshape(c, h, w)
        s_mean = s_mean.reshape(c, 1, 1) * label
        colored = colored + s_mean
        colored_feature = alpha * colored + (1 - alpha) * (f_c * label)
    except:
        # Need fix
        # RuntimeError: MAGMA gesdd : the updating process of SBDSDC did not converge
        colored_feature = f_c * label

    return colored_feature


class MultimodalStyleTransfer:
    def __init__(self, n_cluster, alpha, device='cpu', lam=0.1, max_cycles=None, print_tsne=True , print_cluster_criterium=True):
        self.k = n_cluster
        self.k_means_estimator = KMeans(n_cluster)
        if (type(alpha) is int or type(alpha) is float) and 0 <= alpha <= 1:
            self.alpha = [alpha] * n_cluster
        elif type(alpha) is list and len(alpha) == n_cluster:
            self.alpha = alpha
        else:
            raise ValueError('Error for alpha')

        self.device = device
        self.lam = lam
        self.max_cycles = max_cycles
        self.print_tsne = print_tsne
        self.print_cluster_criterium = print_cluster_criterium 

    def style_feature_clustering(self, style_feature):
        C, _, _ = style_feature.shape
        s = style_feature.reshape(C, -1).transpose(0, 1)
        if self.print_tsne=="kmeans" : 
            ##############################
            ###     T -SNE display     ###
            ##############################

            # Dimensionality reduction using t-SNE
            tsne = TSNE(n_components=3, random_state=42)
            s_tsne = tsne.fit_transform(s.to('cpu'))
           
            ##Kmeans 
            kmeans = KMeans(n_clusters=self.k, random_state=42)
            labels = kmeans.fit_predict(s_tsne)

            # Reorganise clusters
            clusters = []
            for i in range(self.k):
                cluster_points = s[labels == i]
                clusters.append(cluster_points)
            
            # Convertir a tensores y enviar a dispositivo
            cluster_centers = torch.Tensor(kmeans.cluster_centers_).to(self.device)
            clusters = [torch.Tensor(cluster).to(self.device) for cluster in clusters]

            # Plot the PCA-transformed data in 3D using Kmeans
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            for cluster_label in range(self.k):
                cluster_points = s_tsne[labels == cluster_label]
                ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], label=f'Cluster {cluster_label}')
    
            ax.set_xlabel('t-SNE Component 1')
            ax.set_ylabel('t-SNE Component 2')
            ax.set_zlabel('t-SNE Component 3')
            ax.set_title('t-SNE of Style Feature (3D). Clustering using Kmeans.')
            ax.legend()

            # Save the plot to a file
            plt.savefig('pca_kmeans_3d_plot.png')
            
            # Show the plot
            plt.show()
        
        if self.print_tsne=="dbscan" : 
            ##############################
            ###    DBSCAN clustering   ###
            ##############################
            tsne = TSNE(n_components=3, random_state=42)
            s_tsne = tsne.fit_transform(s.to('cpu'))

            dbscan = DBSCAN(eps=3, min_samples=15)
            labels = dbscan.fit_predict(s_tsne)
            #Number of clusters in labels, ignoring noise if present.
            nclusters = len(set(labels)) - (1 if -1 in labels else 0)
            nnoise = list(labels).count(-1)
            #Plot the DBSCAN clustering in 3D
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            #Black color for noise points
            unique_labels = set(labels)
            #Define a list of typical colors
            typical_colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'brown', 'pink', 'gray', 'olive']
            colors = typical_colors[:len(unique_labels)]
            for k, col in zip(unique_labels, colors):
                if k == -1:
                    col = [0, 0, 0, 1]
                class_member_mask = (labels == k)

                xy = s_tsne[class_member_mask]

                ax.scatter(xy[:, 0], xy[:, 1], xy[:, 2], c=[col],
                marker='o', label='Cluster %d' % k)
                ax.set_xlabel('t-SNE Component 1')
                ax.set_ylabel('t-SNE Component 2')
                ax.set_zlabel('t-SNE Component 3')
                ax.set_title('t-SNE of Style Feature (3D). Clustering using DBSCAN.')
                ax.legend()
            #Save the plot to a file
            plt.savefig('dbscan_3d_plot.png')
            #Show the plot
            plt.show()

        self.k_means_estimator.fit(s.to('cpu'))
        # you could replace kmeans by dbscan here 

        if self.print_cluster_criterium : 
            ######################################
            ###     Cluster number display     ###
            ######################################
            # Elbow Method
            inertia = []
            silhouette_scores = []
            for k in range(1, 11):
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans.fit(s_tsne)
                inertia.append(kmeans.inertia_)
                if k > 1: 
                    silhouette_scores.append(metrics.silhouette_score(s_tsne, kmeans.labels_))

            print("silouhette score" , silhouette_scores)
            print("cluster number", np.argmax(silhouette_scores) + 2)
            # Calculate the rate of change in inertia
            rate_of_change = [inertia[i] - inertia[i+1] for i in range(len(inertia)-1)]
            
            # Find the index where rate of change is maximized
            elbow_index = rate_of_change.index(max(rate_of_change))
            plt.plot([1, 2, 3, 4, 5, 6, 7, 8, 9 , 10], inertia)
            plt.show()

            # Return the optimal number of clusters
            print("elbow method", elbow_index + 1) 
            print("============================")

        labels = torch.Tensor(self.k_means_estimator.labels_).to(self.device)
        cluster_centers = torch.Tensor(self.k_means_estimator.cluster_centers_).to(self.device).transpose(0, 1)

        s = s.to(self.device)
        clusters = [s[labels == i] for i in range(self.k)]

        return cluster_centers, clusters

    def graph_based_style_matching(self, content_feature, style_feature):
        cluster_centers, s_clusters = self.style_feature_clustering(style_feature)

        D = data_term(content_feature, cluster_centers).to('cpu').numpy().astype(np.double)
        V = pairwise_term(cluster_centers, lam=self.lam).to('cpu').numpy().astype(np.double)
        labels = torch.Tensor(aexpansion_grid(D, V, max_cycles=self.max_cycles)).to(self.device)
        return labels, s_clusters

    def transfer(self, content_feature, style_feature):
        labels, s_clusters = self.graph_based_style_matching(content_feature, style_feature)
        f_cs = torch.zeros_like(content_feature)
        for f_s, a, k in zip(s_clusters, self.alpha, range(self.k)):
            label = (labels == k).unsqueeze(dim=0).expand_as(content_feature)
            if (label > 0).any():
                label = label.to(torch.float)
                f_cs += labeled_whiten_and_color(content_feature, f_s, a, label)

        return f_cs