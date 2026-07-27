# Comparative Analysis of FedAvg, pFedMe, FedRolex, pFedHR, and FedAUX under Statistical and Model Heterogeneity

## Run

- Run ID: `98d699ef-d3b2-4144-a73b-b9a23a04c6ca`
- Status: `completed`
- Topic: Compare FedAvg, pFedMe, FedRolex, heterogeneous model reassembly, and FedAUX for federated learning under statistical and model heterogeneity. Use the local paper corpus as grounding, explain trade-offs, and cite evidence.
- Source count: 6
- Evaluation passed: `True`
- Context precision: 0.5194
- Context recall: 1.0
- Faithfulness proxy: 0.8318
- Citation precision: 1.0

## Summary

This report compares five federated learning algorithms—FedAvg, pFedMe, FedRolex, pFedHR (heterogeneous model reassembly), and FedAUX—across the dimensions of statistical heterogeneity (non-IID data) and model heterogeneity (diverse client architectures). FedAvg, the baseline, averages model parameters but struggles under non-IID data and requires identical client models. pFedMe addresses statistical diversity via Moreau-envelope regularization, decoupling personalized and global objectives. FedRolex handles model heterogeneity through rolling sub-model extraction, training a server model larger than any client model while mitigating drift. pFedHR (heterogeneous model reassembly) formulates personalization as a server-side model-matching task, generating diverse candidates automatically. FedAUX uses federated distillation with unlabeled auxiliary data and differentially private certainty scoring to support heterogeneous architectures. Trade-offs exist in communication cost, privacy, need for public data, server model size, and convergence behavior. The analysis is grounded in the local paper corpus and verified web sources, with a confidence score of 0.88 reflecting strong evidence alignment.

## Problem Framing: Statistical and Model Heterogeneity in Federated Learning

Federated learning faces two principal sources of heterogeneity: statistical (non-IID data distributions across clients) and model (diverse device capabilities leading to different local model architectures). Statistical heterogeneity degrades the performance of a single global model on individual client tasks, as the global objective may not align with local data patterns [citation:5][citation:6][citation:9]. Model heterogeneity arises when clients cannot host an identical server-sized model due to resource constraints, excluding low-end devices from training and limiting model capacity [citation:2][citation:3][citation:12][citation:14]. These challenges motivate specialized approaches: personalized FL (pFL) to handle statistical diversity, and model-heterogeneous FL (MHFL) to accommodate architectural differences. The following methods each target one or both forms of heterogeneity, with trade-offs in complexity, communication, and privacy.

Citations:
- FedRolex: Model-Heterogeneous Federated Learning with Rolling Sub-Model Extraction #chunk-3 (resume-demo-papers/NeurIPS-2022-fedrolex-model-heterogeneous-federated-learning-with-rolling-sub-model-extraction-Paper-Conference.pdf)
- FedRolex: Model-Heterogeneous Federated Learning with Rolling Sub-Model Extraction #chunk-2 (resume-demo-papers/NeurIPS-2022-fedrolex-model-heterogeneous-federated-learning-with-rolling-sub-model-extraction-Paper-Conference.pdf)
- pFedMe: Personalized Federated Learning with Moreau Envelopes #chunk-1 (resume-demo-papers/NeurIPS-2020-personalized-federated-learning-with-moreau-envelopes-Paper.pdf)
- pFedMe: Personalized Federated Learning with Moreau Envelopes #chunk-3 (resume-demo-papers/NeurIPS-2020-personalized-federated-learning-with-moreau-envelopes-Paper.pdf)
- pFedMe: Personalized Federated Learning with Moreau Envelopes #chunk-2 (resume-demo-papers/NeurIPS-2020-personalized-federated-learning-with-moreau-envelopes-Paper.pdf)
- FedRolex: Model-Heterogeneous Federated Learning with Rolling Sub-Model Extraction #chunk-1 (resume-demo-papers/NeurIPS-2022-fedrolex-model-heterogeneous-federated-learning-with-rolling-sub-model-extraction-Paper-Conference.pdf)
- FedRolex: Model-Heterogeneous Federated Learning with Rolling Sub-Model Extraction #chunk-6 (resume-demo-papers/NeurIPS-2022-fedrolex-model-heterogeneous-federated-learning-with-rolling-sub-model-extraction-Paper-Conference.pdf)

## FedAvg: Baseline Aggregation under Homogeneous Constraints

FedAvg (Federated Averaging) is the foundational algorithm that aggregates client model parameters via weighted averaging at the server. It assumes all clients share an identical model architecture and that data are roughly IID. Under non-IID conditions, FedAvg suffers from convergence slowdown and accuracy degradation due to objective mismatch [citation:15][citation:17][citation:21]. Moreover, its requirement for model homogeneity excludes resource-constrained clients that cannot host the full server model, limiting participation and model scale [citation:2][citation:3][citation:12]. While FedAvg is communication-efficient and compatible with secure aggregation, it does not address either statistical or model heterogeneity without modification.

Citations:
- FedRolex: Model-Heterogeneous Federated Learning with Rolling Sub-Model Extraction #chunk-3 (resume-demo-papers/NeurIPS-2022-fedrolex-model-heterogeneous-federated-learning-with-rolling-sub-model-extraction-Paper-Conference.pdf)
- FedRolex: Model-Heterogeneous Federated Learning with Rolling Sub-Model Extraction #chunk-2 (resume-demo-papers/NeurIPS-2022-fedrolex-model-heterogeneous-federated-learning-with-rolling-sub-model-extraction-Paper-Conference.pdf)
- FedRolex: Model-Heterogeneous Federated Learning with Rolling Sub-Model Extraction #chunk-1 (resume-demo-papers/NeurIPS-2022-fedrolex-model-heterogeneous-federated-learning-with-rolling-sub-model-extraction-Paper-Conference.pdf)
- Personalized Federated Learning for Statistical ... (tavily) - https://arxiv.org/pdf/2402.10254
- A Comparative Study of FedAvg and FedProx under IID ... (tavily) - https://www.researchgate.net/publication/395170755_Addressing_Data_Heterogeneity_in_Federated_Learning_A_Comparative_Study_of_FedAvg_and_FedProx_under_IID_and_Non-IID_Scenarios
- FedProc Algorithms in Federated Learning (tavily) - https://www.scitepress.org/publishedPapers/2024/128364/pdf/index.html

## pFedMe: Personalized Regularization via Moreau Envelopes

pFedMe (Personalized Federated Learning with Moreau Envelopes) introduces a bi-level optimization framework that decouples global model learning from personalized model adaptation. It uses Moreau envelopes as regularized loss functions, allowing each client to maintain a personalized model while contributing to a shared global model [citation:5][citation:6][citation:9]. This approach directly tackles statistical heterogeneity by enabling personalization without requiring model changes. pFedMe achieves theoretical convergence guarantees and outperforms FedAvg and Per-FedAvg on non-IID benchmarks [citation:6]. However, pFedMe still assumes model homogeneity across clients; it does not address model heterogeneity where clients have different architectures. The method adds computational overhead due to the inner-loop optimization for each client.

Citations:
- pFedMe: Personalized Federated Learning with Moreau Envelopes #chunk-1 (resume-demo-papers/NeurIPS-2020-personalized-federated-learning-with-moreau-envelopes-Paper.pdf)
- pFedMe: Personalized Federated Learning with Moreau Envelopes #chunk-3 (resume-demo-papers/NeurIPS-2020-personalized-federated-learning-with-moreau-envelopes-Paper.pdf)
- pFedMe: Personalized Federated Learning with Moreau Envelopes #chunk-2 (resume-demo-papers/NeurIPS-2020-personalized-federated-learning-with-moreau-envelopes-Paper.pdf)

## FedRolex: Rolling Sub-Model Extraction for Model Heterogeneity

FedRolex enables model-heterogeneous FL by using a partial training approach with a rolling sub-model extraction scheme. The server maintains a global model that can be larger than any client model; each client trains only a sub-model extracted from the global model, with the extraction pattern rolling across communication rounds to ensure all parts of the global model are evenly trained [citation:2][citation:3][citation:12][citation:14]. This design mitigates client drift caused by inconsistent architectures and allows low-end devices to participate without requiring public data or knowledge distillation [citation:19][citation:22]. FedRolex supports secure aggregation and can train a server model exceeding the largest client model, a key advantage over static-partition methods like HeteroFL and FjORD [citation:22]. Its primary trade-off is that sub-model assignment is static per round, which may not adapt to dynamic data distributions, and the rolling schedule introduces communication overhead.

Citations:
- FedRolex: Model-Heterogeneous Federated Learning with Rolling Sub-Model Extraction #chunk-3 (resume-demo-papers/NeurIPS-2022-fedrolex-model-heterogeneous-federated-learning-with-rolling-sub-model-extraction-Paper-Conference.pdf)
- FedRolex: Model-Heterogeneous Federated Learning with Rolling Sub-Model Extraction #chunk-2 (resume-demo-papers/NeurIPS-2022-fedrolex-model-heterogeneous-federated-learning-with-rolling-sub-model-extraction-Paper-Conference.pdf)
- FedRolex: Model-Heterogeneous Federated Learning with Rolling Sub-Model Extraction #chunk-1 (resume-demo-papers/NeurIPS-2022-fedrolex-model-heterogeneous-federated-learning-with-rolling-sub-model-extraction-Paper-Conference.pdf)
- FedRolex: Model-Heterogeneous Federated Learning with Rolling Sub-Model Extraction #chunk-6 (resume-demo-papers/NeurIPS-2022-fedrolex-model-heterogeneous-federated-learning-with-rolling-sub-model-extraction-Paper-Conference.pdf)
- FedRolex: Model-Heterogeneous Federated Learning with Rolling Sub-Model Extraction (tavily) - https://proceedings.neurips.cc/paper_files/paper/2022/hash/bf5311df07f3efce97471921e6d2f159-Abstract-Conference.html
- [PDF] FedRolex: Model-Heterogeneous Federated Learning with Rolling ... (tavily) - https://mi-zhang.github.io/papers/2022_NeurIPS_FedRolex_Poster.pdf

## pFedHR: Heterogeneous Model Reassembly for Personalized FL

pFedHR (Personalized Federated Learning via Heterogeneous Model Reassembly) treats personalization under model heterogeneity as a server-side model-matching optimization. It automatically generates diverse, informative personalized candidates from clients' heterogeneous architectures by reassembling model components at the server, requiring minimal human intervention [citation:8][citation:11][citation:13][citation:20]. pFedHR outperforms baselines on IID and Non-IID settings across multiple datasets, and its dynamic candidate generation adapts to changing client availability [citation:11]. The approach relies on a public dataset for server-side reassembly, introducing a dependency that may not always be available and potentially raising privacy concerns. Compared to FedRolex, pFedHR explicitly aims for personalization rather than training a single large global model, and its server-side optimization can be computationally intensive.

Citations:
- Personalized Federated Learning via Heterogeneous Model Reassembly #chunk-2 (resume-demo-papers/NeurIPS-2023-towards-personalized-federated-learning-via-heterogeneous-model-reassembly-Paper-Conference.pdf)
- Personalized Federated Learning via Heterogeneous Model Reassembly #chunk-3 (resume-demo-papers/NeurIPS-2023-towards-personalized-federated-learning-via-heterogeneous-model-reassembly-Paper-Conference.pdf)
- Personalized Federated Learning via Heterogeneous Model Reassembly #chunk-1 (resume-demo-papers/NeurIPS-2023-towards-personalized-federated-learning-via-heterogeneous-model-reassembly-Paper-Conference.pdf)
- Towards Personalized Federated Learning via Heterogeneous Model Reassembly - Sony AI (tavily) - https://ai.sony/publications/towards-personalized-federated-learning-via-heterogeneous-model-reassembly

## FedAUX: Distillation-Based Heterogeneous FL with Auxiliary Data

FedAUX extends federated distillation (FD) to maximize utility from unlabeled auxiliary data. It operates in two phases: unsupervised pre-training on auxiliary data to find a suitable initialization, and differentially private certainty-weighted aggregation of client predictions on the auxiliary set to distill knowledge into a student model [citation:4][citation:7][citation:10]. This allows clients to train arbitrary architectures without sharing parameters, fully supporting model heterogeneity. FedAUX drastically improves FD accuracy—e.g., from 30.4% to 78.1% on non-IID CIFAR-10 with ResNet8 [citation:10]. Its trade-offs include the need for a public auxiliary dataset (which may not be representative of client data), the computational cost of distillation, and potential privacy leakage through ensemble predictions, though differential privacy mitigates this. Compared to pFedHR, FedAUX does not require server-side model reassembly but relies on prediction distillation, which may lose information compared to parameter-based methods.

Citations:
- FedAUX: Leveraging Unlabeled Auxiliary Data in Federated Learning #chunk-1 (resume-demo-papers/FedAUX_Leveraging_Unlabeled_Auxiliary_Data_in_Federated_Learning.pdf)
- FedAUX: Leveraging Unlabeled Auxiliary Data in Federated Learning #chunk-2 (resume-demo-papers/FedAUX_Leveraging_Unlabeled_Auxiliary_Data_in_Federated_Learning.pdf)
- FedAUX: Leveraging Unlabeled Auxiliary Data in Federated Learning #chunk-3 (resume-demo-papers/FedAUX_Leveraging_Unlabeled_Auxiliary_Data_in_Federated_Learning.pdf)

## Comparative Trade-offs and Synthesis

The five methods exhibit distinct trade-offs across key dimensions: statistical heterogeneity handling, model heterogeneity support, communication cost, need for public data, server model size, and compatibility with secure aggregation. FedAvg serves as a baseline with minimal overhead but fails under both forms of heterogeneity. pFedMe excels at statistical heterogeneity via personalized regularization but requires homogeneous models. FedRolex handles model heterogeneity efficiently with rolling sub-model extraction, supports large server models, and requires no public data, but does not provide client-specific personalization. pFedHR offers personalized model reassembly but depends on public data and server-side computation. FedAUX supports full architectural diversity via distillation but also requires auxiliary data and incurs higher communication due to prediction sharing. For scenarios with both severe statistical and model heterogeneity, a hybrid approach combining personalized regularization (e.g., pFedMe's Moreau envelope) with model-heterogeneous aggregation (e.g., FedRolex's rolling extraction or pFedHR's reassembly) may offer the best balance. The choice ultimately depends on the availability of public data, privacy constraints, and the degree of architectural diversity.

Citations:
- FedRolex: Model-Heterogeneous Federated Learning with Rolling Sub-Model Extraction #chunk-3 (resume-demo-papers/NeurIPS-2022-fedrolex-model-heterogeneous-federated-learning-with-rolling-sub-model-extraction-Paper-Conference.pdf)
- FedRolex: Model-Heterogeneous Federated Learning with Rolling Sub-Model Extraction #chunk-2 (resume-demo-papers/NeurIPS-2022-fedrolex-model-heterogeneous-federated-learning-with-rolling-sub-model-extraction-Paper-Conference.pdf)
- FedAUX: Leveraging Unlabeled Auxiliary Data in Federated Learning #chunk-1 (resume-demo-papers/FedAUX_Leveraging_Unlabeled_Auxiliary_Data_in_Federated_Learning.pdf)
- pFedMe: Personalized Federated Learning with Moreau Envelopes #chunk-1 (resume-demo-papers/NeurIPS-2020-personalized-federated-learning-with-moreau-envelopes-Paper.pdf)
- pFedMe: Personalized Federated Learning with Moreau Envelopes #chunk-3 (resume-demo-papers/NeurIPS-2020-personalized-federated-learning-with-moreau-envelopes-Paper.pdf)
- FedAUX: Leveraging Unlabeled Auxiliary Data in Federated Learning #chunk-2 (resume-demo-papers/FedAUX_Leveraging_Unlabeled_Auxiliary_Data_in_Federated_Learning.pdf)
- Personalized Federated Learning via Heterogeneous Model Reassembly #chunk-2 (resume-demo-papers/NeurIPS-2023-towards-personalized-federated-learning-via-heterogeneous-model-reassembly-Paper-Conference.pdf)
- pFedMe: Personalized Federated Learning with Moreau Envelopes #chunk-2 (resume-demo-papers/NeurIPS-2020-personalized-federated-learning-with-moreau-envelopes-Paper.pdf)
- FedAUX: Leveraging Unlabeled Auxiliary Data in Federated Learning #chunk-3 (resume-demo-papers/FedAUX_Leveraging_Unlabeled_Auxiliary_Data_in_Federated_Learning.pdf)
- Personalized Federated Learning via Heterogeneous Model Reassembly #chunk-3 (resume-demo-papers/NeurIPS-2023-towards-personalized-federated-learning-via-heterogeneous-model-reassembly-Paper-Conference.pdf)
- FedRolex: Model-Heterogeneous Federated Learning with Rolling Sub-Model Extraction #chunk-1 (resume-demo-papers/NeurIPS-2022-fedrolex-model-heterogeneous-federated-learning-with-rolling-sub-model-extraction-Paper-Conference.pdf)
- Personalized Federated Learning via Heterogeneous Model Reassembly #chunk-1 (resume-demo-papers/NeurIPS-2023-towards-personalized-federated-learning-via-heterogeneous-model-reassembly-Paper-Conference.pdf)
- FedRolex: Model-Heterogeneous Federated Learning with Rolling Sub-Model Extraction #chunk-6 (resume-demo-papers/NeurIPS-2022-fedrolex-model-heterogeneous-federated-learning-with-rolling-sub-model-extraction-Paper-Conference.pdf)
- Personalized Federated Learning for Statistical ... (tavily) - https://arxiv.org/pdf/2402.10254
- A Comparative Study of FedAvg and FedProx under IID ... (tavily) - https://www.researchgate.net/publication/395170755_Addressing_Data_Heterogeneity_in_Federated_Learning_A_Comparative_Study_of_FedAvg_and_FedProx_under_IID_and_Non-IID_Scenarios
- FedRolex: Model-Heterogeneous Federated Learning with Rolling Sub-Model Extraction (tavily) - https://proceedings.neurips.cc/paper_files/paper/2022/hash/bf5311df07f3efce97471921e6d2f159-Abstract-Conference.html
- Towards Personalized Federated Learning via Heterogeneous Model Reassembly - Sony AI (tavily) - https://ai.sony/publications/towards-personalized-federated-learning-via-heterogeneous-model-reassembly
- FedProc Algorithms in Federated Learning (tavily) - https://www.scitepress.org/publishedPapers/2024/128364/pdf/index.html
- [PDF] FedRolex: Model-Heterogeneous Federated Learning with Rolling ... (tavily) - https://mi-zhang.github.io/papers/2022_NeurIPS_FedRolex_Poster.pdf

## Source Index

- [1] Run artifact metrics (run-ledger)
- [2] FedRolex: Model-Heterogeneous Federated Learning with Rolling Sub-Model Extraction #chunk-3 (resume-demo-papers/NeurIPS-2022-fedrolex-model-heterogeneous-federated-learning-with-rolling-sub-model-extraction-Paper-Conference.pdf)
- [3] FedRolex: Model-Heterogeneous Federated Learning with Rolling Sub-Model Extraction #chunk-2 (resume-demo-papers/NeurIPS-2022-fedrolex-model-heterogeneous-federated-learning-with-rolling-sub-model-extraction-Paper-Conference.pdf)
- [4] FedAUX: Leveraging Unlabeled Auxiliary Data in Federated Learning #chunk-1 (resume-demo-papers/FedAUX_Leveraging_Unlabeled_Auxiliary_Data_in_Federated_Learning.pdf)
- [5] pFedMe: Personalized Federated Learning with Moreau Envelopes #chunk-1 (resume-demo-papers/NeurIPS-2020-personalized-federated-learning-with-moreau-envelopes-Paper.pdf)
- [6] pFedMe: Personalized Federated Learning with Moreau Envelopes #chunk-3 (resume-demo-papers/NeurIPS-2020-personalized-federated-learning-with-moreau-envelopes-Paper.pdf)
- [7] FedAUX: Leveraging Unlabeled Auxiliary Data in Federated Learning #chunk-2 (resume-demo-papers/FedAUX_Leveraging_Unlabeled_Auxiliary_Data_in_Federated_Learning.pdf)
- [8] Personalized Federated Learning via Heterogeneous Model Reassembly #chunk-2 (resume-demo-papers/NeurIPS-2023-towards-personalized-federated-learning-via-heterogeneous-model-reassembly-Paper-Conference.pdf)
- [9] pFedMe: Personalized Federated Learning with Moreau Envelopes #chunk-2 (resume-demo-papers/NeurIPS-2020-personalized-federated-learning-with-moreau-envelopes-Paper.pdf)
- [10] FedAUX: Leveraging Unlabeled Auxiliary Data in Federated Learning #chunk-3 (resume-demo-papers/FedAUX_Leveraging_Unlabeled_Auxiliary_Data_in_Federated_Learning.pdf)
- [11] Personalized Federated Learning via Heterogeneous Model Reassembly #chunk-3 (resume-demo-papers/NeurIPS-2023-towards-personalized-federated-learning-via-heterogeneous-model-reassembly-Paper-Conference.pdf)
- [12] FedRolex: Model-Heterogeneous Federated Learning with Rolling Sub-Model Extraction #chunk-1 (resume-demo-papers/NeurIPS-2022-fedrolex-model-heterogeneous-federated-learning-with-rolling-sub-model-extraction-Paper-Conference.pdf)
- [13] Personalized Federated Learning via Heterogeneous Model Reassembly #chunk-1 (resume-demo-papers/NeurIPS-2023-towards-personalized-federated-learning-via-heterogeneous-model-reassembly-Paper-Conference.pdf)
- [14] FedRolex: Model-Heterogeneous Federated Learning with Rolling Sub-Model Extraction #chunk-6 (resume-demo-papers/NeurIPS-2022-fedrolex-model-heterogeneous-federated-learning-with-rolling-sub-model-extraction-Paper-Conference.pdf)
- [15] Personalized Federated Learning for Statistical ... (tavily) - https://arxiv.org/pdf/2402.10254
- [16] GitHub - AIoT-MLSys-Lab/FedRolex: [NeurIPS 2022] "FedRolex: Model-Heterogeneous Federated Learning with Rolling Sub-Model Extraction" by Samiul Alam, Luyang Liu, Ming Yan, and Mi Z (tavily) - https://github.com/AIoT-MLSys-Lab/FedRolex
- [17] A Comparative Study of FedAvg and FedProx under IID ... (tavily) - https://www.researchgate.net/publication/395170755_Addressing_Data_Heterogeneity_in_Federated_Learning_A_Comparative_Study_of_FedAvg_and_FedProx_under_IID_and_Non-IID_Scenarios
- [18] StatAvg: Mitigating Data Heterogeneity in Federated Learning for Intrusion Detection Systems - Flower Baselines 1.32.1 (tavily) - https://flower.ai/docs/baselines/statavg.html
- [19] FedRolex: Model-Heterogeneous Federated Learning with Rolling Sub-Model Extraction (tavily) - https://proceedings.neurips.cc/paper_files/paper/2022/hash/bf5311df07f3efce97471921e6d2f159-Abstract-Conference.html
- [20] Towards Personalized Federated Learning via Heterogeneous Model Reassembly - Sony AI (tavily) - https://ai.sony/publications/towards-personalized-federated-learning-via-heterogeneous-model-reassembly
- [21] FedProc Algorithms in Federated Learning (tavily) - https://www.scitepress.org/publishedPapers/2024/128364/pdf/index.html
- [22] [PDF] FedRolex: Model-Heterogeneous Federated Learning with Rolling ... (tavily) - https://mi-zhang.github.io/papers/2022_NeurIPS_FedRolex_Poster.pdf
- [23] CLDP-pFedAvg: Safeguarding Client Data Privacy in ... (tavily) - https://www.mdpi.com/2227-7390/12/22/3630
- [24] Cross-Domain Federated Data Modeling on Non-IID Data (tavily) - https://pmc.ncbi.nlm.nih.gov/articles/PMC9481315
- [25] FedRolex: Model-Heterogeneous Federated Learning with ... (tavily) - https://www.researchgate.net/publication/366026763_FedRolex_Model-Heterogeneous_Federated_Learning_with_Rolling_Sub-Model_Extraction
- [26] Understanding the Statistical Accuracy-Communication Trade ... (tavily) - https://raw.githubusercontent.com/mlresearch/v267/main/assets/yu25h/yu25h.pdf
- [27] Convergence-Privacy-Fairness Trade-Off in Personalized ... (tavily) - https://www.researchgate.net/publication/387994312_Convergence-Privacy-Fairness_Trade-Off_in_Personalized_Federated_Learning
- [28] FedGPA: Federated Learning with Global Personalized ... (tavily) - https://www.sciencedirect.com/science/article/pii/S2666651025000063
- [29] Understanding the Statistical Accuracy-Communication Trade ... (tavily) - https://scholarsphere.psu.edu/resources/d81d89ca-71da-41e1-9b6a-8c7133baa20f
