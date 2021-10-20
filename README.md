# Analysis Libraries

> python -m pip install bix-analysis-libraries   

An assortment of analysis libraries.

## Components
There are two components of the libraries divided by experiment type. Each component has a data prep module to prepare the data from different software into a canonical form. The prepared data can then be analyzed with the analysis modules (to be added).

**Use**

To `import` the modules use the form
```python
from bix_analysis_libraries[.<component>] import <module>
# or
import bix_analysis_libraries[.<component>].<module>
```
where `<component>` is the name of the component (if needed) and `<module>` is the name of the module. Any modules in the Standard Component do not require a component name, while modules in all other components do.

**Examples**
```python
from bix_analysis_libraries import bix_standard_functions as bsf
# or
import bix_analysis_libraries.bix_standard_functions as bsf
```

```python
from bix_analysis_libraries.dark_bias import bix_dark_bias_data_prep as dbdp
# or
import bix_analysis_libraries.dark_bias.bix_dark_bias_data_prep as dbdp
```

---

### Standard Component
> No component requried   

Contains standard functions.

#### Bix Standard Functions
Provides standard functions.

---

### Dark Bias component
> Component name: `dark_bias`  

Contains data prep for Dark Bias experiments.

#### Bix Dark Bias Data Prep
> Module name: `bix_dark_bias_data_prep`  

Data prep for MPP tracking and JV data obtained using the [Easy Biologic](https://github.com/bicarlsen/easy-biologic "github.com/bicarlsen/easy-biologic") library.

---

### Temperature Degradation component
> Component name: `temperature_degradation`  

Contains data prep for Temperature Degradation experiments.

#### Temperature Degradation Data Prep
> Module name: `temperature_degradation_data_prep`  

Data prep for MPP tracking and JV data.