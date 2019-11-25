# Examples of recent results in lossless convexification

This repository implements the following papers.

```
@ARTICLE{2019arXiv191109013M,
       author = {{Malyuta}, Danylo and {Acikmese}, Behcet},
        title = "{Lossless Convexification of Optimal Control Problems with Semi-continuous Inputs}",
      journal = {arXiv e-prints},
         year = "2019",
        month = "Nov",
        pages = {arXiv:1911.09013}
}
```

```
@ARTICLE{2019arXiv190202726M,
       author = {{Malyuta}, Danylo and {Szmuk}, Michael and {Acikmese}, Behcet},
        title = "{Lossless convexification of non-convex optimal control problems with disjoint semi-continuous inputs}",
      journal = {arXiv e-prints},
         year = "2019",
        month = "Feb",
        pages = {arXiv:1902.02726}
}
```

## Installation

To run the code, you must have Python 2.7.15 and [Gurobi
8.1](http://www.gurobi.com/downloads/download-center) installed. To install
Python and other dependenies (except Gurobi) on Ubuntu, we recommend that you
install [Anaconda for Python 2.7](https://www.anaconda.com/distribution/) and
then execute (from inside this repository's directory):

```
$ conda create -n lcvx python=2.7 anaconda # Answer yes to everything
$ source activate lcvx
$ pip install -r requirements.txt
$ source activate lcvx
```

## Examples

1. Satellite docking to a rotating space station. The actuators are 12 reaction
   control system (RCS) jets of which up to 4 can be fired simultaneously

<p align="center">
	<img width="800" src="/automatica_2019/figures/automatica_2019_example.png?raw=true">
</p>

2. Rocket landing with a two-mode thruster: a high-thrust low-gimbal mode and a low-thrust high-gimbal mode.

<p align="center">
	<img width="800" src="/ifac_wc_2020/figures/ifac_wc_2020_example.png?raw=true">
</p>
