
![pytest](https://github.com/rauschenberger/collasso/actions/workflows/pytest.yaml/badge.svg)
![pytest](https://img.shields.io/github/actions/workflow/status/rauschenberger/collasso/pytest.yaml?label=pytest)
[![codecov](https://codecov.io/gh/rauschenberger/collasso/graph/badge.svg?token=7E6GSQ32F5)](https://codecov.io/gh/rauschenberger/collasso)
![mypy](https://github.com/rauschenberger/collasso/actions/workflows/mypy.yaml/badge.svg)
![ruff](https://github.com/rauschenberger/collasso/actions/workflows/ruff.yaml/badge.svg)
![pylint](https://github.com/rauschenberger/collasso/actions/workflows/pylint.yaml/badge.svg)
![docstr-coverage](https://github.com/rauschenberger/collasso/actions/workflows/docstr-coverage.yaml/badge.svg)
![pydocstyle](https://github.com/rauschenberger/collasso/actions/workflows/pydocstyle.yaml/badge.svg)
![pydoclint](https://github.com/rauschenberger/collasso/actions/workflows/pydoclint.yaml/badge.svg)
![numpydoc](https://github.com/rauschenberger/collasso/actions/workflows/numpydoc.yaml/badge.svg)
![pip-audit](https://github.com/rauschenberger/collasso/actions/workflows/pip-audit.yaml/badge.svg)
![pip-licenses](https://github.com/rauschenberger/collasso/actions/workflows/pip-licenses.yaml/badge.svg)
![sphinx](https://github.com/rauschenberger/collasso/actions/workflows/sphinx.yaml/badge.svg)

<img src="collasso-logo.png" alt="collasso-logo" style="width:25%; height:auto;">

<img src="https://raw.githubusercontent.com/rauschenberger/collasso/main/collasso-logo.png" alt="collasso-logo" width="25%">

# Sparse Linear Multi-Task Regression

## Scope

The Python package `collasso` implements sparse linear multi-task regression with correlation-based information sharing (*Rauschenberger*, 2026). In contrast to `MultiTaskLassoCV` from `scikit-learn`, it supports target-specific feature selection, target-specific feature matrices, and privileged information. 

## Installation

Install the latest release from PyPI or conda-forge:

```bash
pip install -U collasso
conda install -c conda-forge collasso
```

Alternatively, install the development version from GitHub or TestPyPI:

```bash
pip install -U git+https://github.com/rauschenberger/collasso.git
pip install -i https://test.pypi.org/simple/ collasso
```

## Usage

Use the class `CoopLassoCV` to model a multivariate target (_n_ × _q_ matrix **Y**) based on high-dimensional features (_n_ × _p_ matrix **X**).

```python
from collasso import CoopLassoCV
model = CoopLassoCV()
model.fit(X_train, y_train)
model.coef_ # estimated coefficients
model.predict(X_test) # out-of-sample predictions
```

Please find the full documentation on the [website](https://rauschenberger.github.io/collasso/). The [vignette](https://github.com/rauschenberger/collasso/blob/main/scripts/vignette.py) contains examples on multi-task regression with a common feature matrix, multi-task regression with specific feature matrices, and multi-task regression with privileged information. This repository also contains the scripts for a [simulation](https://github.com/rauschenberger/collasso/blob/main/scripts/simulation.py) and an [application](https://github.com/rauschenberger/collasso/blob/main/scripts/application.py).

## Reference

Armin Rauschenberger
[![AR](https://info.orcid.org/wp-content/uploads/2019/11/orcid_16x16.png)](https://orcid.org/0000-0001-6498-4801)
(2026).
"Sparse linear multi-task regression with correlation-based information sharing". 
_Manuscript in preparation._

## Repository

The source code of this Python package is available on [GitHub](https://github.com/rauschenberger/collasso). This personal GitHub repository is mirrored at two institutional GitLab instances (see [LIH](https://git.lih.lu/arauschenberger/collasso) and [LCSB](https://gitlab.com/uniluxembourg/Personalfolders/armin.rauschenberger/collasso)).

## Disclosure

Large-language models (mainly Claude Sonnet 4.6 and Claude Opus 4.6) were used for reviewing Python code and documentation as well as for drafting or reviewing configuration files (`.toml` and `.yaml`).

## Disclaimer

**Copyright** &copy; 2026 Armin Rauschenberger; Luxembourg Institute of Health (LIH), Department of Medical Informatics (DMI), Bioinformatics and Artificial Intelligence (BioAI); University of Luxembourg, Luxembourg Centre for Systems Biomedicine (LCSB), Biomedical Data Science (BDS). **All Rights Reserved.** (NB: The Python package will have an open-source license at a later stage.)
