# Enhanced Adversarial Debiasing with Intersectionality and Validation Support

This repository contains an enhanced implementation of the **Adversarial Debiasing** algorithm from [AIF360](https://github.com/IBM/AIF360), with additional support for intersectional fairness, validation-based early stopping, and improved training robustness.

## ✨ Features

- ✅ **Intersectional Debiasing**  
  Supports multiple protected attributes, including intersectional combinations (e.g., race+gender).

- ✅ **Validation & Early Stopping**  
  Monitors validation loss and stops training early to prevent overfitting.

- ✅ **Dropout API with Reproducibility**  
  Controlled dropout with explicit seeds for consistent results.

- ✅ **Verbose Logging**  
  Optional progress logs to monitor training and validation loss.

- ✅ **Compatible with AIF360 BinaryLabelDataset**  
  Seamless integration with existing fairness datasets and evaluation tools.

---

## 📦 Requirements

- Python 3.6+
- TensorFlow 1.x (`tensorflow.compat.v1`)
- AIF360 (`pip install aif360`)

**Note**: This implementation uses TensorFlow 1.x via `tensorflow.compat.v1` and disables eager execution explicitly.

---

## 🚀 Getting Started

### 1. Install Dependencies

```bash
pip install aif360 tensorflow==1.15
````

### 2. Example Usage

```python
from aif360.datasets import AdultDataset
from enhanced_adversarial_debiasing import AdversarialDebiasing
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
sess = tf.Session()

dataset = AdultDataset()
train, test = dataset.split([0.7], shuffle=True)

ad = AdversarialDebiasing(
    privileged_groups=[{'sex': 1}],
    unprivileged_groups=[{'sex': 0}],
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

ad.fit(train)
preds = ad.predict(test)
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

---

## 📖 License

MIT License. See `LICENSE` for details.

---

## 🙌 Acknowledgments

Based on the original [AIF360 Adversarial Debiasing](https://aif360.readthedocs.io/en/stable/modules/generated/aif360.algorithms.inprocessing.AdversarialDebiasing.html) implementation by IBM Research. This fork includes enhancements to better support research on intersectional fairness in machine learning.
