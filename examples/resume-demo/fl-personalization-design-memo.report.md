# Design Memo: Selecting a Federated Learning Approach for Non-IID Data and Heterogeneous Client Capacity

## Run

- Run ID: `0404a715-df93-4639-a32b-9fec9477c708`
- Status: `completed`
- Topic: Using the local federated learning corpus, write a design memo for choosing between pFedMe, FedRolex, heterogeneous model reassembly, and FedAUX when clients have non-IID data and heterogeneous model capacity. Cite local evidence and note trade-offs.
- Source count: 6
- Evaluation passed: `True`
- Context precision: 0.5597
- Context recall: 1.0
- Faithfulness proxy: 0.8459
- Citation precision: 1.0

## Summary

This design memo compares four federated learning approaches—pFedMe, FedRolex, heterogeneous model reassembly (pFedHR), and FedAUX—for scenarios with non-IID client data and heterogeneous model capacity. pFedMe uses Moreau envelopes to personalize per-client models but assumes a homogeneous model architecture. FedRolex employs rolling sub-model extraction to train a large global model across devices with different capacities without requiring public data. pFedHR reassembles heterogeneous client models into a shared knowledge base for personalized fine-tuning. FedAUX leverages unlabeled auxiliary data via federated distillation, enabling heterogeneous architectures but requiring a public dataset. The memo recommends FedRolex as the primary choice for its balance of inclusiveness, no public data requirement, and strong performance under model heterogeneity, with pFedHR as a strong alternative when personalized model architectures are a priority. FedAUX is suitable when public unlabeled data is available, and pFedMe remains a baseline for homogeneous-model personalization.

## Problem framing

Federated learning (FL) in cross-device settings faces two fundamental challenges: (1) statistical heterogeneity, where client data distributions are non-IID, and (2) system heterogeneity, where clients have diverse computational and memory capacities. Traditional model-homogeneous FL approaches force all clients to use identical model architectures, which excludes low-end devices and limits the ability to train large models [evidence index 2, 4]. pFedMe addresses non-IID data via Moreau-envelope regularization but assumes a shared global model architecture, limiting its applicability when clients have heterogeneous capacities [evidence index 5, 8]. FedRolex enables model-heterogeneous FL through rolling sub-model extraction, allowing clients to train sub-models of varying sizes without requiring public data, thus including low-end devices while still training a large server model [evidence index 2, 4, 16]. Heterogeneous model reassembly (pFedHR) allows each client to maintain a unique network structure and achieves personalization by reassembling knowledge from other clients' models, but it requires more complex coordination [evidence index 3, 9]. FedAUX extends federated distillation to leverage unlabeled auxiliary data, supporting heterogeneous client architectures but depending on the availability of a public auxiliary dataset [evidence index 6, 7]. The key trade-offs revolve around the need for public data, the degree of model heterogeneity supported, the complexity of aggregation, and the impact on personalization under non-IID conditions.

Citations:
- FedRolex: Model-Heterogeneous Federated Learning with Rolling Sub-Model Extraction #chunk-1 (resume-demo-papers/NeurIPS-2022-fedrolex-model-heterogeneous-federated-learning-with-rolling-sub-model-extraction-Paper-Conference.pdf)
- Personalized Federated Learning via Heterogeneous Model Reassembly #chunk-1 (resume-demo-papers/NeurIPS-2023-towards-personalized-federated-learning-via-heterogeneous-model-reassembly-Paper-Conference.pdf)
- FedRolex: Model-Heterogeneous Federated Learning with Rolling Sub-Model Extraction #chunk-2 (resume-demo-papers/NeurIPS-2022-fedrolex-model-heterogeneous-federated-learning-with-rolling-sub-model-extraction-Paper-Conference.pdf)
- pFedMe: Personalized Federated Learning with Moreau Envelopes #chunk-2 (resume-demo-papers/NeurIPS-2020-personalized-federated-learning-with-moreau-envelopes-Paper.pdf)
- FedAUX: Leveraging Unlabeled Auxiliary Data in Federated Learning #chunk-1 (resume-demo-papers/FedAUX_Leveraging_Unlabeled_Auxiliary_Data_in_Federated_Learning.pdf)
- FedAUX: Leveraging Unlabeled Auxiliary Data in Federated Learning #chunk-2 (resume-demo-papers/FedAUX_Leveraging_Unlabeled_Auxiliary_Data_in_Federated_Learning.pdf)
- pFedMe: Personalized Federated Learning with Moreau Envelopes #chunk-6 (resume-demo-papers/NeurIPS-2020-personalized-federated-learning-with-moreau-envelopes-Paper.pdf)
- Personalized Federated Learning via Heterogeneous Model Reassembly #chunk-4 (resume-demo-papers/NeurIPS-2023-towards-personalized-federated-learning-via-heterogeneous-model-reassembly-Paper-Conference.pdf)
- FedRolex: Model-Heterogeneous Federated Learning with Rolling Sub-Model Extraction (tavily) - https://proceedings.neurips.cc/paper_files/paper/2022/hash/bf5311df07f3efce97471921e6d2f159-Abstract-Conference.html

## Approach comparison and trade-offs

pFedMe: Uses Moreau envelopes to decouple global and personalized objectives, effectively handling non-IID data by allowing each client to optimize a personalized model close to the global model. However, it requires all clients to have the same model architecture, making it unsuitable for scenarios with heterogeneous client capacities. It does not require public data and has relatively low communication overhead, but the assumption of homogeneous model capacity is a significant limitation in practice [evidence index 5, 8].

FedRolex: A partial-training approach where each client trains a rolling sub-model extracted from the global server model. The rolling mechanism ensures that over time, every part of the global model is updated by clients with sufficient capacity, while smaller clients contribute to a smaller sub-model. This method does not require any public data, supports full model heterogeneity, and can train a server model larger than any individual client model. Its main trade-off is potential performance degradation under extreme non-IID settings because the sub-model extraction is not personalized; the server model is shared, and personalization is limited to the sub-model training process. The approach is compatible with secure aggregation and has shown to reduce the gap between model-heterogeneous and model-homogeneous FL, especially under large-model regimes [evidence index 2, 4, 16].

Heterogeneous Model Reassembly (pFedHR): Allows each client to have a completely different network structure. Clients upload their models to the server, which reassembles knowledge into a public knowledge bank; clients then download relevant modules to fine-tune personalized models. This method excels at personalization and handles extreme model heterogeneity, but it requires more complex server-side reassembly and higher communication costs. It also does not require public data but may incur higher overhead than simpler extraction methods [evidence index 3, 9].

FedAUX: Based on federated distillation, FedAUX uses unlabeled auxiliary data to distill knowledge from client models into a student model. It supports heterogeneous client architectures and can improve performance by leveraging public unlabeled data. However, its dependence on a public dataset is a major limitation; if such data is not available or does not match the client data distribution, performance may degrade. It also introduces additional training steps for distillation and certainty scoring [evidence index 6, 7].

Citations:
- FedRolex: Model-Heterogeneous Federated Learning with Rolling Sub-Model Extraction #chunk-1 (resume-demo-papers/NeurIPS-2022-fedrolex-model-heterogeneous-federated-learning-with-rolling-sub-model-extraction-Paper-Conference.pdf)
- Personalized Federated Learning via Heterogeneous Model Reassembly #chunk-1 (resume-demo-papers/NeurIPS-2023-towards-personalized-federated-learning-via-heterogeneous-model-reassembly-Paper-Conference.pdf)
- FedRolex: Model-Heterogeneous Federated Learning with Rolling Sub-Model Extraction #chunk-2 (resume-demo-papers/NeurIPS-2022-fedrolex-model-heterogeneous-federated-learning-with-rolling-sub-model-extraction-Paper-Conference.pdf)
- pFedMe: Personalized Federated Learning with Moreau Envelopes #chunk-2 (resume-demo-papers/NeurIPS-2020-personalized-federated-learning-with-moreau-envelopes-Paper.pdf)
- FedAUX: Leveraging Unlabeled Auxiliary Data in Federated Learning #chunk-1 (resume-demo-papers/FedAUX_Leveraging_Unlabeled_Auxiliary_Data_in_Federated_Learning.pdf)
- FedAUX: Leveraging Unlabeled Auxiliary Data in Federated Learning #chunk-2 (resume-demo-papers/FedAUX_Leveraging_Unlabeled_Auxiliary_Data_in_Federated_Learning.pdf)
- pFedMe: Personalized Federated Learning with Moreau Envelopes #chunk-6 (resume-demo-papers/NeurIPS-2020-personalized-federated-learning-with-moreau-envelopes-Paper.pdf)
- Personalized Federated Learning via Heterogeneous Model Reassembly #chunk-4 (resume-demo-papers/NeurIPS-2023-towards-personalized-federated-learning-via-heterogeneous-model-reassembly-Paper-Conference.pdf)
- FedRolex: Model-Heterogeneous Federated Learning with Rolling Sub-Model Extraction (tavily) - https://proceedings.neurips.cc/paper_files/paper/2022/hash/bf5311df07f3efce97471921e6d2f159-Abstract-Conference.html

## Recommendation and justification

Based on the trade-offs and evidence, FedRolex is the recommended approach for most cross-device FL deployments with non-IID data and heterogeneous client capacity. It directly addresses the exclusion of low-end devices, does not require public data, and has been shown to perform well under large-model and large-dataset conditions. Its rolling sub-model extraction is computationally efficient and compatible with secure aggregation, making it practical for real-world systems. The primary risk is reduced personalization under extreme non-IID skew, but for many applications, the global model trained via FedRolex can still be effectively fine-tuned locally. If personalization is the top priority and clients have very diverse architectures, heterogeneous model reassembly (pFedHR) is a strong alternative, though with higher complexity. FedAUX is recommended only when a high-quality public unlabeled dataset is available, as its performance heavily depends on this external data. pFedMe remains a baseline for homogeneous-model settings and is less suitable when clients have varying capacities.

Citations:
- FedRolex: Model-Heterogeneous Federated Learning with Rolling Sub-Model Extraction #chunk-1 (resume-demo-papers/NeurIPS-2022-fedrolex-model-heterogeneous-federated-learning-with-rolling-sub-model-extraction-Paper-Conference.pdf)
- Personalized Federated Learning via Heterogeneous Model Reassembly #chunk-1 (resume-demo-papers/NeurIPS-2023-towards-personalized-federated-learning-via-heterogeneous-model-reassembly-Paper-Conference.pdf)
- FedRolex: Model-Heterogeneous Federated Learning with Rolling Sub-Model Extraction #chunk-2 (resume-demo-papers/NeurIPS-2022-fedrolex-model-heterogeneous-federated-learning-with-rolling-sub-model-extraction-Paper-Conference.pdf)
- pFedMe: Personalized Federated Learning with Moreau Envelopes #chunk-2 (resume-demo-papers/NeurIPS-2020-personalized-federated-learning-with-moreau-envelopes-Paper.pdf)
- FedAUX: Leveraging Unlabeled Auxiliary Data in Federated Learning #chunk-1 (resume-demo-papers/FedAUX_Leveraging_Unlabeled_Auxiliary_Data_in_Federated_Learning.pdf)
- FedAUX: Leveraging Unlabeled Auxiliary Data in Federated Learning #chunk-2 (resume-demo-papers/FedAUX_Leveraging_Unlabeled_Auxiliary_Data_in_Federated_Learning.pdf)
- pFedMe: Personalized Federated Learning with Moreau Envelopes #chunk-6 (resume-demo-papers/NeurIPS-2020-personalized-federated-learning-with-moreau-envelopes-Paper.pdf)
- Personalized Federated Learning via Heterogeneous Model Reassembly #chunk-4 (resume-demo-papers/NeurIPS-2023-towards-personalized-federated-learning-via-heterogeneous-model-reassembly-Paper-Conference.pdf)
- FedRolex: Model-Heterogeneous Federated Learning with Rolling Sub-Model Extraction (tavily) - https://proceedings.neurips.cc/paper_files/paper/2022/hash/bf5311df07f3efce97471921e6d2f159-Abstract-Conference.html

## Source Index

- [1] Run artifact metrics (run-ledger)
- [2] FedRolex: Model-Heterogeneous Federated Learning with Rolling Sub-Model Extraction #chunk-1 (resume-demo-papers/NeurIPS-2022-fedrolex-model-heterogeneous-federated-learning-with-rolling-sub-model-extraction-Paper-Conference.pdf)
- [3] Personalized Federated Learning via Heterogeneous Model Reassembly #chunk-1 (resume-demo-papers/NeurIPS-2023-towards-personalized-federated-learning-via-heterogeneous-model-reassembly-Paper-Conference.pdf)
- [4] FedRolex: Model-Heterogeneous Federated Learning with Rolling Sub-Model Extraction #chunk-2 (resume-demo-papers/NeurIPS-2022-fedrolex-model-heterogeneous-federated-learning-with-rolling-sub-model-extraction-Paper-Conference.pdf)
- [5] pFedMe: Personalized Federated Learning with Moreau Envelopes #chunk-2 (resume-demo-papers/NeurIPS-2020-personalized-federated-learning-with-moreau-envelopes-Paper.pdf)
- [6] FedAUX: Leveraging Unlabeled Auxiliary Data in Federated Learning #chunk-1 (resume-demo-papers/FedAUX_Leveraging_Unlabeled_Auxiliary_Data_in_Federated_Learning.pdf)
- [7] FedAUX: Leveraging Unlabeled Auxiliary Data in Federated Learning #chunk-2 (resume-demo-papers/FedAUX_Leveraging_Unlabeled_Auxiliary_Data_in_Federated_Learning.pdf)
- [8] pFedMe: Personalized Federated Learning with Moreau Envelopes #chunk-6 (resume-demo-papers/NeurIPS-2020-personalized-federated-learning-with-moreau-envelopes-Paper.pdf)
- [9] Personalized Federated Learning via Heterogeneous Model Reassembly #chunk-4 (resume-demo-papers/NeurIPS-2023-towards-personalized-federated-learning-via-heterogeneous-model-reassembly-Paper-Conference.pdf)
- [10] [2212.01548] FedRolex: Model-Heterogeneous Federated ... (tavily) - https://arxiv.org/abs/2212.01548
- [11] FedRolex: Model-Heterogeneous Federated Learning with Rolling ... (tavily) - https://github.com/AIoT-MLSys-Lab/FedRolex
- [12] Personalized Federated Learning through Local Memorization (tavily) - https://proceedings.mlr.press/v162/marfoq22a.html
- [13] FedRolex: Model-Heterogeneous Federated Learning with ... (tavily) - https://www.researchgate.net/publication/366026763_FedRolex_Model-Heterogeneous_Federated_Learning_with_Rolling_Sub-Model_Extraction
- [14] Design a federated learning system in seven steps – OpenMined (tavily) - https://openmined.org/blog/design-a-federated-learning-system-in-seven-steps
- [15] FedRolex: Model-Heterogeneous Federated Learning with Rolling ... (tavily) - https://www.researchgate.net/publication/401455933_FedRolex_Model-Heterogeneous_Federated_Learning_with_Rolling_Sub-Model_Extraction
- [16] FedRolex: Model-Heterogeneous Federated Learning with Rolling Sub-Model Extraction (tavily) - https://proceedings.neurips.cc/paper_files/paper/2022/hash/bf5311df07f3efce97471921e6d2f159-Abstract-Conference.html
- [17] FedRolex: Model-Heterogeneous Federated Learning with ... (tavily) - https://mi-zhang.github.io/papers/2022_NeurIPS_FedRolex_Poster.pdf
- [18] Personalized Federated Learning through Local ... (tavily) - https://hal.science/hal-03697969v1/document
- [19] FedRolex: Model-Heterogeneous Federated Learning with ... (tavily) - https://proceedings.neurips.cc/paper_files/paper/2022/file/bf5311df07f3efce97471921e6d2f159-Paper-Conference.pdf
