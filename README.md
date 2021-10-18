# Analysis Libraries

> python -m pip install bix-analysis-libraries 
An assortment of analysis libraries.

## Components
There are two components of the libraries divided by experiment type. Each component has a data prep module to prepare the data from different software into a canonical form. The prepared data can then be analyzed with the analysis modules.

**Use**

To `import` the modules use the form
```python
from bric_analysis_libraries[.<component>] import <module>
# or
import bric_analysis_libraries[.<component>].<module>
```
where `<component>` is the name of the component (if needed) and `<module>` is the name of the module. Any modules in the Standard Component do not require a component name, while modules in all other components do.

**Examples**
```python
from bric_analysis_libraries import standard_functions as std
# or
import bric_analysis_libraries.standard_functions as std
```

```python
from bric_analysis_libraries.jv import aging_analysis as aging
# or
import bric_analysis_libraries.jv.aging_analysis as aging
```

---

### Standard Component
> No component requried
Contains standard functions.

#### Standard Functions
Provides standard functions.

---

### JV Component
> Component name: `jv`  
Contains data prep and analysis packages for JV experiments.

#### Aging Analysis
> Module name: `aging_analysis`  
Analysis of degradation mecahnisms

#### Aging Data Prep
> Module name: `aging_data_prep`
Data prep from the stability lab.

#### EC Lab Analysis
> Module name: `ec_lab_analysis`
Analysis of EC experiments

#### EC Lab Data Prep
> Module name: `ec_lab_data_prep`
Data prep of experiments form EC Lab.

#### Igor JV Data Prep
> Module name: `igor_jv_data_prep`
Data prep of JV experiments coming from the old IV setup.

#### JV Analysis
> Module name: `jv_analysis`
Analysis of JV experiments.

