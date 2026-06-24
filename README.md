# Power Consumption Analysis & Regression Modeling

Predicts city power consumption from weather and environmental data. Covers data
inspection, exploratory plots, correlation analysis, feature scaling, and four
regression models (Linear, Ridge, Lasso, Elastic Net) with residual diagnostics.
Ported from R to Python.

## Dataset

Expects `powerconsumption.csv` with: `Datetime`, `Temperature`, `Humidity`,
`WindSpeed`, `GeneralDiffuseFlows`, `DiffuseFlows`, and `PowerConsumption_Zone1/2/3`.
The target is the mean consumption across the three zones.

## Setup & usage

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn statsmodels
python power_consumption_analysis.py
```

## Note

The feature matrix includes the zone columns and the target (matching the original
analysis), which leaks the target into the predictors. Drop those columns from `X`
for a leakage-free model — see the comment at the train/test split.
