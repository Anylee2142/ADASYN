## ADASYN
= http://sci2s.ugr.es/keel/pdf/algorithm/congreso/2008-He-ieee.pdf
= Extended(?) version of SMOTE

``` python3
import adasyn.generate as source
data = YOUR_FEATURES
target_name = YOUR_TARGET_NAME
source.analysis(data, target_name)
oversampled_data = source.adasyn(data, target_name)
```