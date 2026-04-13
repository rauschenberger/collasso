
![tests](https://github.com/rauschenberger/collasso/actions/workflows/tests.yml/badge.svg)
[![codecov](https://codecov.io/gh/rauschenberger/collasso/graph/badge.svg?token=7E6GSQ32F5)](https://codecov.io/gh/rauschenberger/collasso)
![mypy](https://github.com/rauschenberger/collasso/actions/workflows/mypy.yml/badge.svg)
![ruff](https://github.com/rauschenberger/collasso/actions/workflows/ruff.yml/badge.svg)
![pylint](https://github.com/rauschenberger/collasso/actions/workflows/pylint.yml/badge.svg)
![docs](https://github.com/rauschenberger/collasso/actions/workflows/pages/pages-build-deployment/badge.svg)


# Sparse Linear Multi-Task Regression

This personal GitHub repository is mirrored at two institutional GitLab instances (see [LIH](https://git.lih.lu/arauschenberger/collasso) and [LCSB](https://gitlab.com/uniluxembourg/Personalfolders/armin.rauschenberger/collasso)). It contains a Python library that implements "Sparse linear multi-task regression with correlation-based information sharing". 

## Usage

Install `collasso` from GitHub:

```bash
pip install git+https://github.com/rauschenberger/collasso.git
```

Use the function `CoopLassoCV` to model a multivariate target ($n \times q$ matrix $\boldsymbol{Y}$) based on high-dimensional features ($n \times q$ matrix $\boldsymbol{X}$).

```python
from collasso import CoopLassoCV
model = CoopLassoCV()
model.fit(X_train,y_train)
model.predict(y_test)
```

## Reference

Armin Rauschenberger
[![AR](https://info.orcid.org/wp-content/uploads/2019/11/orcid_16x16.png)](https://orcid.org/0000-0001-6498-4801)
(2026).
"Sparse linear multi-task regression with correlation-based information sharing". 
_Manuscript in preparation._

## Disclaimer

The Python library `collasso` implements sparse linear multi-task regression with correlation-based information sharing (Rauschenberger, 2026).

**Copyright** &copy; 2026 Armin Rauschenberger; Luxembourg Institute of Health (LIH), Department of Medical Informatics (DMI), Bioinformatics and Artificial Intelligence (BioAI); University of Luxembourg, Luxembourg Centre for Systems Biomedicine (LCSB), Biomedical Data Science (BDS). **All Rights Reserved.** (NB: The Python library will have an open-source license at a later stage.)
