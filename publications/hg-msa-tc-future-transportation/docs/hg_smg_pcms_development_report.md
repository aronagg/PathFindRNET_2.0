# HG-SMG PCMS Development Report

| scene | method | n_clusters | params_json | interval_lower | interval_upper | interval_distance |
| --- | --- | --- | --- | --- | --- | --- |
| bellevue_116th_ne12th | kmeans | 10 | {"max_iter": 300, "n_clusters": 10, "n_init": 10} | 8 | 13 | 0 |
| bellevue_116th_ne12th | hdbscan | 4 | {"min_cluster_size": 80, "min_samples": 10} | 8 | 13 | 4 |
| bellevue_116th_ne12th | optics | 5 | {"max_eps": 0.31873943664454396, "min_samples": 40, "xi": 0.05} | 8 | 13 | 3 |
| bellevue_150th_newport | kmeans | 7 | {"max_iter": 300, "n_clusters": 7, "n_init": 10} | 7 | 13 | 0 |
| bellevue_150th_newport | hdbscan | 6 | {"min_cluster_size": 80, "min_samples": 10} | 7 | 13 | 1 |
| bellevue_150th_newport | optics | 7 | {"max_eps": 0.036343618402565836, "min_samples": 40, "xi": 0.07} | 7 | 13 | 0 |
| bellevue_150th_eastgate | kmeans | 11 | {"max_iter": 300, "n_clusters": 11, "n_init": 10} | 7 | 11 | 0 |
| bellevue_150th_eastgate | hdbscan | 10 | {"min_cluster_size": 320, "min_samples": 20} | 7 | 11 | 0 |
| bellevue_150th_eastgate | optics | 11 | {"max_eps": 0.022171253230168272, "min_samples": 80, "xi": 0.05} | 7 | 11 | 0 |
| bellevue_150th_se38th | kmeans | 12 | {"max_iter": 300, "n_clusters": 12, "n_init": 10} | 8 | 20 | 0 |
| bellevue_150th_se38th | hdbscan | 7 | {"min_cluster_size": 80, "min_samples": 10} | 8 | 20 | 1 |
| bellevue_150th_se38th | optics | 12 | {"max_eps": 0.09623894321033927, "min_samples": 40, "xi": 0.05} | 8 | 20 | 0 |
| bellevue_ne8th | kmeans | 10 | {"max_iter": 300, "n_clusters": 10, "n_init": 10} | 9 | 10 | 0 |
| bellevue_ne8th | hdbscan | 9 | {"min_cluster_size": 160, "min_samples": 10} | 9 | 10 | 0 |
| bellevue_ne8th | optics | 10 | {"max_eps": 0.008851381172013503, "min_samples": 80, "xi": 0.05} | 9 | 10 | 0 |

Only the frozen model-selection candidate grid was read. EMAS_HG and semantic/reference labels were not used in selection.
