## Usage
To run `pipeline.py`:
1. Unzip processed_dataset.
2. Run `claim.ipynb`, `evidence.ipynb`, `stance.ipynb`, and `relation.ipynb` in the components folder to generate the models. The outputted models should be saved in the models folder.
3. Run `pipeline.py`

## Structure
- The components folder contains files for training the individual components
- The data_preprocessing folder contains files to transform the original datasets to datasets suitable for the corresponding components
- The ibm_claim_evidence_dataset folder contains the original datasets for the claim detection component and the evidence detection component. It is obtained from [1]
- The relation_dataset folder contains the original datasets for the relation prediction component. It is obtained from [2]
- The stance_dataset folder contains the original datasets for the stance detection component. It is obtained from [3] and [4]

## Bibliography
[1] Ruty Rinott, L. Dankin, C. A. Perez, M. M. Khapra, E. Aharoni, and N. Slonim, “Show Me Your Evidence – an Automatic Method for Context Dependent Evidence Detection,” Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing, pp. 440–450, Sep. 2015, doi: https://doi.org/10.18653/v1/d15-1050.

[2] C. Stab, T. Miller, B. Schiller, P. Rai, and I. Gurevych, “Cross-topic Argument Mining from Heterogeneous Sources,” TUbilio (Technical University of Darmstadt), Jan. 2018, doi: https://doi.org/10.18653/v1/d18-1402.

[3] R. Bar-Haim, I. Bhattacharya, F. Dinuzzo, A. Saha, and N. Slonim, “Stance Classification of Context-Dependent Claims,” Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics: Volume 1, Long Papers, pp. 251–261, Jan. 2017, doi: https://doi.org/10.18653/v1/e17-1024.

[4] L. Chen, L. Bing, R. He, Q. Yu, Y. Zhang, and L. Si, “IAM: A comprehensive and large-scale dataset for integrated argument mining tasks,” Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 2277–2287, 2022. doi:10.18653/v1/2022.acl-long.162
