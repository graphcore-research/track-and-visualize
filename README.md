# numerics vis

  

Repo for working on numerics-vis

## Installation

From the root directory of the repository, run:
```

python3 -m venv .venv

echo "export PYTHONPATH=\${PYTHONPATH}:\$(dirname \${VIRTUAL_ENV})/src:\$(dirname \${VIRTUAL_ENV})/experiments/Transformer:\$(dirname \${VIRTUAL_ENV})/experiments/Transformer/training" >> .venv/bin/activate

source .venv/bin/activate

pip install -r requirements.txt
```



## Required Log Schema

In order to visualise the data in a useful way and simplify the API design some assurances about the structure of the data are needed. Therefore the library enforces a Schema. The logging data must be in the form of a `pandas.MultiIndex`, with three top level indexes, **metadata**, **scalar_stats** and **exponent_counts**.

### Metadata
The `"metadata"` top-level contains the non-data dependent information about the tensor.

| Column Name | Tuple (for Pandas) | Description | Type |
| -------- | ------- | ------- | ------- |
| name | `("metadata","name")` | The name of neural network component (eg. `"layers.0.attention.wq"`) | `str` |
| type* | `("metadata","type")` | What module from the DL framework (e.g. torch) is tensor from  | `str` |
| tensor_type | `("metadata","tensor_type")` | What type of data is in the tensor (i.e. Pre-Activation, Gradient, Weight, Optimiser State ) | `Literal["Activation","Gradient", "Weight","Optimiser_State"]` |
| step | `("metadata","step")` | The training step at which the tensor was logged | `int` |
| dtype | `("metadata","dtype")` | The dtype of the logged tensor (`str` must be parseable by `ml_dtypes.finfo`) | `str` | 

**\*** denotes optional. We currently don't have any functionally which depends on the module type.

### Scalar Stats
The `"scalar_stats"` top-level contains all the scalar statistics logged to summarise the data contained with-in the tensors.
| Column Name | Tuple (for Pandas) | Description | Type |
| -------- | ------- | ------- | ------- |
|`+`| `("scalar_stats,*)`| There is no restriction on what scalar statistics about the tensor that can be visualised. The schema only enforces there this at-least 1 column which this criteria. The `*` can be substituted for any string you wish to use (e.g. `"std"`). | `Union[float,int]`|

### Exponent Counts
These columns store the histogram counts for each exponent in the number format. For example, Column `n` ${n}$ would contain the quanity of values that fall in the range ${2^n}$ to ${2^{n+1}}$.

| Column Name | Tuple (for Pandas) | Description | Type |
| -------- | ------- | ------- | ------- |
|`\d+`|`("exponent_counts",*)`| The * is replaced by the exponent value (${n}$ above), the schema enforces there is atleast one column which meets this criteria | `int`|
|`-inf`|`("exponent_counts",*)`| If using simulated low precision any values which underflow the range of the LP format (being cast to) should be represented as -inf  | `int`|
|`+inf`|`("exponent_counts",*)`| If using simulated low precision any values which overflow the range of the LP format (being cast to) should be represented as +inf | `int`|

Currently `+inf` & `-inf` are required. However this will likely change to optional as hardware will clamp overflows to the MRV or 0 (in the case of underflows). The work -around if you're not producing infinites in your logs is to simply add the infinity columns with each entry as zero.


## Feedback
Any feedback for the tool would be greatly appreciated.

If providing feedback on the tool could you please provide it in the format below.

Feelfree to add it as a GH issue (https://github.com/graphcore-research/numerics-vis), email me (colmb@graphcore.ai) or message me on slack (@Colm Brandon)

#### What you think of the current State of the tool:
Please provide a rating for each of the 4 criteria and if you have any additional comments put them below.
*Options: P (Poor) | NI (Need Improvement) | S (Satisfactory) | VG (Very Good) | E (Excellent)*

| Criteria | Rating |
| -------- | ------- |
| Features: | |
| Functionality:| |
| Usability: | |
| Visualisation Quality:| |



**Additional Comments:**

### Issues / Bugs
Detail any errors or bugs you encountered here.


### Features
Detail any features you wish to see added to the library (i.e. additional plots, additional inputs for interactive plots, additional interactions with the visualisation, etc..)


### Documentation
Did the documentation make sense? Any areas that need clarifying?




