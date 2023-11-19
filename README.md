
![Maual python](https://img.shields.io/badge/python-3.10-blue)  ![Downloads](https://static.pepy.tech/badge/tsad)  ![Downloads](https://static.pepy.tech/badge/tsad/month)  ![pypi version](https://img.shields.io/pypi/v/tsad)  [![License](https://img.shields.io/badge/license-%20%20GNU%20GPLv3%20-green?style=plastic)](https://www.gnu.org/licenses/gpl-3.0.html)  <!-- ![python](https://img.shields.io/pypi/pyversions/tsad.svg) -->


# Time Series Analysis for Simulation of Technological Processes - TSAD

## Table of Contents

  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)<!-- - [Features](#features) -->
  - [Documentation](#documentation)
  - [Getting Started](#getting-started)
    - [System Requirements](#system-requirements)
    - [Installation](#installation)
    - [Example API Usage](#example-api-usage)
  - [Comparison with Related Libraries](#comparison-with-related-libraries)
  - [References](#references)
  - [Citing TSAD](#citing-tsad)

## Introduction

TSAD is a powerful Python module designed to simplify the work of researchers utilizing machine learning techniques, specifically tailored for addressing key challenges in industrial domains.

The primary objective of TSAD is to empower researchers with effective tools for solving complex problems related to:

- **Fault Detection in Industrial Equipment:**
  - Identify and address faults in industrial machinery with precision, ensuring optimal equipment performance and reliability.

- **Improvement of Technological Processes:**
  - Boost overall performance to enhance operational efficiency.
  - Implement cost-effective measures to reduce operational expenses.
  - Strengthen quality control and management practices for superior outcomes.




### Solving Fault Detection problem

In TSAD, the problem of fault detection is reduced to the problem of detecting time series anomalies using a well-known technique:

- Forecast a multivariate Time Series (TS) one point ahead (Also works for univariate TS)
- Compute residuals between forecast and true values
- Apply analysis of residuals and thus find anomalies

![image-2](./docs/waico_pics/readme/Useful.jpg)

### Applications

1. **Predictive and Prescriptive Analysis:**
   - Predict the development of situations based on data analysis.
   - Automate decision-making for equipment diagnostics and repairs.

2. **Predictive Equipment Maintenance:**
   - Utilize mathematical modeling methods, including machine learning.
   - Reduce equipment breakdown frequency and associated damages.
   - Decrease costs for diagnostics and maintenance of machinery and industrial equipment.
   - Create artificial intelligence systems for predictive maintenance.

3. **Ultra-Short-Term Forecasting:**
   - Analyze real-time data streams.
   - Forecast abnormal situations.
   - Implement artificial intelligence systems for real-time monitoring.

4. **Detection of Anomalies in Production Processes:**
   - Identify anomalies in manufacturing processes.
   - Investigate the root causes of anomalies.
   - Develop artificial intelligence systems based on mathematical modeling algorithms, machine learning, and historical data.

5. **Wide-Scope Time Series Forecasting:**
   - Forecast time series broadly, beyond technical system diagnostics.

6. **Anomaly Detection in a Broad Context:**
   - Detect anomalies in various contexts, not limited to technical system diagnostics.

7. **Sensor Failure Detection:**
   - Detect malfunctioning sensors.

8. **Creation of Virtual Sensors:**
   - Develop virtual sensors.

9. **Quality Production Forecasting:**
   - Forecast the quality of production.

These applied use cases showcase the versatility of TSAD in addressing a wide range of scenarios within industrial data analysis. Whether it's predicting equipment failures, monitoring real-time data streams, or forecasting production quality, TSAD provides powerful solutions for enhancing decision-making and optimizing operational processes.


### Key Features

Explore the powerful features offered by TSAD for comprehensive testing, evaluation, and data analysis refinement:

#### Testing Capabilities:

- Evaluate TSAD with your own dataset.
- Test custom anomaly detection and forecasting algorithms.

#### Exploratory Data Analysis:

- Analyze distributions, missing data, and patterns.

#### Effective Data Issue Resolution:

- Identify and resolve data problems with measurable metrics.

#### Application of Machine Learning:

- Utilize machine learning for anomaly detection and forecasting tasks.

#### Performance Evaluation:

- Assess the quality of anomaly detection algorithms.



## Documentation

The [TSAD Documentation](https://tsad.readthedocs.io/) covers a wide range of topics, including:
- **Getting Started:**
  - A step-by-step guide on installing TSAD and setting up your environment.
  - Quick start examples to get you up and running with minimal effort.

- **Library Overview:**
  - In-depth explanations of the core components and concepts within the TSAD library.
  - Understanding the structure and design philosophy for effective utilization.

- **Usage Guides and Tutorials:**
  - Detailed guides on utilizing specific functionalities of the library.
  - Practical examples and use cases to demonstrate real-world applications.

- **API Reference:**
  - Comprehensive documentation of the TSAD API.
  - Detailed descriptions of classes, methods, and parameters for advanced users.

## Getting Started

This section will guide you through the process of getting started with the TSAD library.

### System Requirements

Ensure that your system meets the following hardware specifications for optimal performance of TSAD:

- **Minimum Number of Processors (Intel):** 2
- **Minimum Processor Frequency (GHz):** 2.0
- **Minimum RAM Size (GB):** 4.0
- **Minimum Video Memory Size (for an external video adapter) (MB):** 512
- **RAID Configuration Type and Minimum Array Size (GB):** RAID 5; 500

These requirements are intended to provide a guideline for the minimum hardware specifications that should be available on the system where TSAD is deployed. Adjustments may be needed based on the size and complexity of the time series data being processed.

Keep in mind that these specifications are recommended for optimal performance, and deviations may impact the efficiency of the library.


### Installation

Install the latest stable version of `tsad` using `pip` for `Python 3.10`:

```bash
pip install -U tsad
```

Alternatively, you can install the latest development version directly from the GitHub repository:

```bash
pip install git+https://github.com/waico/tsad.git
```

### Example API Usage

To quickly see TSAD in action, consider using a simple example in your Python script. Below is a basic example:
<!-- #### Getting Started -->

<!-- **Installation** through [PyPi](https://pypi.org/project/tsad): 

`pip install -U tsad` -->

```python
# Import necessary modules
import sys
sys.path.insert(1, '../')
from tsad.base.pipeline import Pipeline
from tsad.base.datasets import load_skab
from tsad.pipelines import ResidualAnomalyDetectionTaskSet

# Loading data
dataset = load_skab()
targets = dataset.target_names 
data = dataset.frame.drop(columns=targets).droplevel(level=0)

# Create a pipeline and fit/predict
pipeline = Pipeline(ResidualAnomalyDetectionTaskSet)
pred = pipeline.fit_predict(data,n_epochs=5)
```

After that, you can see:

![image-1](./docs/waico_pics/readme/1.png)

![image-1](./docs/waico_pics/readme/2.png)


Visit the [Tutorials section](https://github.com/waico/tsad/tree/main/Tutorials) in the TSAD GitHub repository for hands-on examples and practical insights.


## Roadmap and Future Developments

We welcome active participation from the community and value your feedback on the desired functionalities of TSAD. Our ongoing development plans include:

1. **Advanced Time Series Preprocessing:**
   - Exploring more sophisticated preprocessing techniques, particularly addressing challenges related to the reduction of time series to a single sampling rate, especially in cases of unevenly spaced time series.

2. **Incorporation of State-of-the-Art (SOTA) Algorithms:**
   - Continuous efforts to integrate additional state-of-the-art algorithms into TSAD, ensuring that the library remains at the forefront of time series analysis.

3. **Flexible Model Implementation:**
   - Introducing a feature that allows users to seamlessly implement and integrate their models into our pipeline by providing a straightforward link to their GitHub repositories. This feature aims to facilitate collaboration among researchers who often seek to validate and compare their models with others.

4. **Benchmark Integration:**
   - Working towards the integration of TSAD with various forecasting and anomaly detection benchmarks. This integration will enable users to assess the performance of TSAD against established standards, fostering transparency and reliability in time series analysis.

Your input and collaboration are vital in shaping the future development of TSAD. We encourage you to share your thoughts, suggestions, and contributions to enhance the functionality and versatility of the library. Together, we can continue to advance time series analysis within the TSAD community.


## Comparison with Related Libraries

Explore how TSAD compares with other libraries in various aspects:

|  | [Merlion](https://github.com/salesforce/Merlion) | [Alibi Detect](https://github.com/SeldonIO/alibi-detect) | [Kats](https://github.com/facebookresearch/Kats) | [pyod](https://github.com/yzhao062/pyod) | [GluonTS](https://github.com/awslabs/gluon-ts) | RRCF | STUMPY | Greykite | [Prophet](https://github.com/facebook/prophet) | [pmdarima](https://pypi.org/project/pmdarima/) | [deepad](https://github.com/fastforwardlabs/deepad) | TSAD
:--- | :---: | :---:|  :---:  | :---: | :---: | :---: | :---: | :---: | :----: | :---: | :---: | :---:
| Forecasting (Прогнозирование) | ✅ | | ✅ |  | ✅ | | | ✅ | ✅ | ✅ | ✅ | ✅ 
| Anomaly Detection (Поиск аномалий) | ✅ | ✅ | ✅ | ✅ | | ✅ | ✅ | ✅ | ✅ | | ✅ | ✅ 
| Metrics (Алгоритмы оценки) | ✅ | | | ✅ | ✅ | | | | | | ✅ | ✅
| Ensembles (Ансамбли) | ✅ | | | ✅ | | ✅  | | | | | | ✅ 
| Benchmarking (Бенчмарки и датасеты) | ✅ | | | ✅ | ✅ | | | | | | | ✅ 
| Visualization (Визуализация результатов) | ✅ | | ✅ | ✅ | | | | ✅ | ✅ | | | ✅ | ✅ 
| Data preprocessing (Предварительная обработка данных) | | | ✅ | | | | | | | | | ✅ 
| Automated EDA (Автоматизированный разведочный анализ данных) | | | | | | | | | | | | ✅ 


<!-- #### Dependencies

* TODO

#### Repo structure

```
  └── repo 
    ├───docs       # documentation
    ├───tutorials   # examples
    ├───tsad       # files of library
``` -->

## References

Explore related libraries and resources:

1.  https://github.com/salesforce/Merlion 
2.  https://github.com/fastforwardlabs/deepad
3.  https://github.com/HendrikStrobelt/LSTMVis 
4.  https://github.com/TezRomacH/python-package-template 
5.  https://github.com/khundman/telemanom 
6.  https://github.com/signals-dev/Orion 
7.  https://github.com/NetManAIOps/OmniAnomaly 
8.  https://github.com/unit8co/darts
9.  https://github.com/tinkoff-ai/etna-ts
9.  https://github.com/yzhao062/pyod
10.  https://www.radiativetransfer.org/misc/typhon/doc/modules.html#datasets
10.  https://github.com/AutoViML/Auto_TS
10.  https://nuancesprog.ru/p/15161/
10.  https://www.sktime.org/en/stable/
10.  https://github.com/zalandoresearch/pytorch-ts
10.  https://github.com/qdata/spacetimeformer
10.  https://joaquinamatrodrigo.github.io/skforecast/0.6.0/index.html

<!-- Про архитектуру: 

https://pypi.org/project/catalyst/  -->

## Citing TSAD
If you're using TSAD in your research or applications, please cite using this BibTeX:
```
@misc{TSAD2013,
  author = {Viacheslav Kozitsin and Oleg Berezin and Iurii Katser and Ivan Maksimov},
  title = {Time Series Analysis for Simulation of Technological Processes - TSAD},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/waico/tsad}},
}
```