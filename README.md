# Power Consumption Analysis & Regression Modeling

Predicts city power consumption from weather and environmental data. Covers data
inspection, exploratory plots, correlation analysis, feature scaling, and four
regression models (Linear, Ridge, Lasso, Elastic Net) with residual diagnostics.


## Dataset

Expects `powerconsumption.csv` with: `Datetime`, `Temperature`, `Humidity`,
`WindSpeed`, `GeneralDiffuseFlows`, `DiffuseFlows`, and `PowerConsumption_Zone1/2/3`.
The target is the mean consumption across the three zones.

## Setup & usage

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn statsmodels
python power_consumption_analysis.py
```

