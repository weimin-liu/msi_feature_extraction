from mfe.src.vis.cluster import show_kmeans



def remove_abnormal_areas(spot, features):
    for _ in range(100):
        n_clusters = int(input('how many clusters to be generated '))
        kmeans = show_kmeans(spot, features, n_clusters)
        re_run = input('Re-run kmeans clustering? y/N')
        if (re_run == 'N') or (re_run == 'n'):
            label = int(input('label to be removed:'))
            spot = spot[kmeans.labels_ != label]
            features = features[kmeans.labels_ != label]
            return spot, features

