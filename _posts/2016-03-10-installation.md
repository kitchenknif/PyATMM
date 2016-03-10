---
layout: page
title: "Installation"
category: doc
date: 2016-03-10 20:35:01
order: 1
---

### Dependencies

* Python 3+
* [NumPy](http://www.numpy.org/)
* [SciPy](https://www.scipy.org/)

Recommended

* Matplotlib (plotting)
* PyTMM (for some examples)

### Installation

After installing a Python 3 distribution with Numpy and Scipy (A list of distributions available [here](https://www.scipy.org/install.html).),
installation of the PyATMM should generally be as easy as downloading a [release]({{ site.codeurl }}/releases) or cloning the repository and either

```
    python3 setup.py install
```

for a source download

or

```
    pip3 install PyATMM-1.0.0a0-py3-none-any.whl
```

if installing a release wheel.
