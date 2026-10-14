| model        | source                   |   precision |   recall |     f1 |
|:-------------|:-------------------------|------------:|---------:|-------:|
| DecisionTree | Original (user-reported) |      0.83   |   0.81   | 0.82   |
| DecisionTree | Pandas/CSV rebuild       |      0.7526 |   0.7449 | 0.7487 |
| DecisionTree | Mongo raw columns        |      0.7526 |   0.7449 | 0.7487 |
| DecisionTree | Mongo + engineered       |      0.6881 |   0.7653 | 0.7246 |
| RandomForest | Original (user-reported) |      0.95   |   0.76   | 0.85   |
| RandomForest | Pandas/CSV rebuild       |      0.9412 |   0.8163 | 0.8743 |
| RandomForest | Mongo raw columns        |      0.9412 |   0.8163 | 0.8743 |
| RandomForest | Mongo + engineered       |      0.9398 |   0.7959 | 0.8619 |