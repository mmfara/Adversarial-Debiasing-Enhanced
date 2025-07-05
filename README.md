# 🧠 Enhanced Adversarial Debiasing (AD⁺)

This repository provides an enhanced implementation of the **Adversarial Debiasing** algorithm from [AIF360](https://github.com/IBM/AIF360), extended to support **multi-class protected attributes**, **intersectional fairness via label encoding**, and **robust training techniques** including gradient clipping and input validation.

---

## ✨ Features

- **Multi-Class Protected Attribute Support**  
  Debias against sensitive attributes with more than two categories (e.g., race = {White, Black, Asian, Hispanic}).

- **Intersectional Group Debiasing**  
  Supports fairness over combinations like race × gender by encoding intersectional groups into a single categorical attribute.

- **Validation & Early Stopping**  
  Monitors validation loss to stop training before overfitting.

- **Reproducible Dropout and Random Seeding**  
  Ensures consistent training runs.

- **Stable Training**  
  Gradient clipping and input validation improve convergence and robustness on real-world data.

- **Compatible with AIF360**  
  Drop-in replacement for `AdversarialDebiasing`, works with `BinaryLabelDataset`.

---

## 📦 Requirements

- Python 3.6+
- TensorFlow 1.x via `tensorflow.compat.v1`
- AIF360 (`pip install aif360`)

> ⚠️ This implementation disables eager execution using `tf.disable_v2_behavior()` and must run under TensorFlow 1.x compatibility mode.

---

## 🚀 Getting Started

### 1. Install Dependencies

bash
pip install aif360 tensorflow==1.15


```
### 2. Example Usage

import numpy as np
import tensorflow.compat.v1 as tf
from aif360.datasets import AdultDataset
from enhanced_adversarial_debiasing import AdversarialDebiasing

# Disable eager execution (required for TF 1.x style code)
tf.disable_v2_behavior()

# Load dataset
dataset = AdultDataset()

# Split into training and test sets
train, test = dataset.split([0.7], shuffle=True)

# STEP 1: Extract individual protected attributes
race = dataset.protected_attributes[:, dataset.protected_attribute_names.index('race')]
gender = dataset.protected_attributes[:, dataset.protected_attribute_names.index('sex')]

# STEP 2: Encode intersectional group label
num_gender_values = len(np.unique(gender))
intersectional = race * num_gender_values + gender  # e.g., 0–3 for binary race/gender

# STEP 3: Replace protected attributes in dataset with the encoded intersectional group
dataset.protected_attributes = intersectional.reshape(-1, 1)
dataset.protected_attribute_names = ['race_gender']

# Repeat for train/test (since they're subsets of the original)
train.protected_attributes = dataset.protected_attributes[:len(train)]
test.protected_attributes = dataset.protected_attributes[len(train):]
train.protected_attribute_names = ['race_gender']
test.protected_attribute_names = ['race_gender']

# STEP 4: Setup TensorFlow session
sess = tf.Session()

# STEP 5: Train Enhanced Adversarial Debiasing model
ad = AdversarialDebiasing(
    privileged_groups=[{'race_gender': 0}],  # You can customize this
    unprivileged_groups=[{'race_gender': 1}, {'race_gender': 2}, {'race_gender': 3}],
    scope_name='adv_debiasing',
    sess=sess,
    num_epochs=50,
    batch_size=128,
    seed=42,
    validation_dataset=test,
    early_stopping_patience=5,
    verbose=True,
    debias=True
)

# STEP 6: Fit and evaluate
ad.fit(train)
preds = ad.predict(test)

# Now `preds` is a debiased BinaryLabelDataset
```
---

## 🆚 Comparison with AIF360 Default

| Feature                     | Enhanced Version | AIF360 Original |
| --------------------------- | ---------------- | --------------- |
| Multi-attribute Support     | ✅                | ❌               |
| Intersectional Fairness     | ✅                | ❌               |
| Validation & Early Stopping | ✅                | ❌               |
| Dropout Control & Seeding   | ✅                | ❌               |
| Logging & Verbosity         | ✅                | Limited         |

---

## 📂 Files

* `enhanced_adversarial_debiasing.py`: Main implementation with intersectionality and validation.
* `original_adversarial_debiasing.py`: Reference baseline implementation from AIF360 for comparison.

---

## 📚 Reference

Zhang, B. H., Lemoine, B., & Mitchell, M. (2018).
**Mitigating Unwanted Biases with Adversarial Learning.**
*Proceedings of the 2018 AAAI/ACM Conference on AI, Ethics, and Society.*

[https://doi.org/10.1145/3278721.3278779](https://doi.org/10.1145/3278721.3278779)

---

## 📖 License

This project uses and extends open-source components from IBM's [AIF360](https://github.com/IBM/AIF360) and the [BlackBoxAuditing](https://github.com/algofairness/BlackBoxAuditing) toolkit..

---

## 🙌 Acknowledgments

Based on the original [AIF360 Adversarial Debiasing](https://aif360.readthedocs.io/en/stable/modules/generated/aif360.algorithms.inprocessing.AdversarialDebiasing.html) implementation by IBM Research. This fork includes enhancements to better support research on intersectional fairness in machine learning.

## 📖 Citation

If you use this code in your research or applications, **please cite the following paper**:

> Farayola, Michael Mayowa, Malika Bendechache, Takfarinas Saber, Regina Connolly, and Irina Tal.  
> *Enhancing Algorithmic Fairness: Integrative Approaches and Multi-Objective Optimization Application in Recidivism Models*.  
> In **Proceedings of the 19th International Conference on Availability, Reliability and Security (ARES 2024)**, pages 1–10, ACM, 2024.  
> [https://doi.org/10.1145/3664476.3669978](https://doi.org/10.1145/3664476.3669978)

BibTeX:
```bibtex
@inproceedings{farayola2024enhancing,
  title={Enhancing Algorithmic Fairness: Integrative Approaches and Multi-Objective Optimization Application in Recidivism Models},
  author={Farayola, Michael Mayowa and Bendechache, Malika and Saber, Takfarinas and Connolly, Regina and Tal, Irina},
  booktitle={Proceedings of the 19th International Conference on Availability, Reliability and Security},
  pages={1--10},
  year={2024}
}

