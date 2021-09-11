# LEGO Regression Project

## Introduction 

In this notebook you'll perform a simple linear regression analysis and report the findings of your model, including both predictive model performance metrics and interpretation of fitted model parameters.

## Objectives

You will be able to:

* Write an SQL query to pull data from an SQL database
* Perform a simple linear regression analysis 
* Evaluate your model and interpret its predictive performance metrics
* Apply an inferential lens to interpret relationships betyouen variables identified by the model

# Task: Develop a LEGO Pricing Algorithm

![pile of legos](index_files/images/legos.jpg)

Photo by <a href="https://unsplash.com/@xavi_cabrera?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Xavi Cabrera</a> on <a href="/s/photos/lego?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>

## Business Understanding

You just got hired by LEGO! Your first project is going to be to develop a pricing algorithm in order to analyze the value of individual lego pieces.

The primary purpose of this algorithm is *inferential*, meaning that **your model should be able to tell us something about the relationship betyouen the attributes of a LEGO set and its price**. You will apply your knowledge of statistics to include appropriate caveats about these relationships.

## Data Understanding

You have been given access to an sql database containing over 700 LEGO sets released in the past, including attributes of those sets as youll as their prices.

You do not need to worry about inflation or differences in currency; just predict the same kinds of prices as are present in the past data, which have already been converted to USD.

## Loading the Data

The database for this project is stored at the path `data/legos.db` in this project repository. The entity relational diagram (E.R.D) for the legos database is displayed below. 

To load in the data, you will need to complete the following tasks:
- Open up a connection to the legos database
- Write an sql query to join all three tables together
- Run the sql query and load the joined tables into a pandas dataframe
    - The easiest method for completing this task is to use the `pd.read_sql` function ([Documentation](https://pandas.pydata.org/docs/reference/api/pandas.read_sql.html))
    - Store the pandas dataframe in a variable called `df`

![Entity Relational Diagram](index_files/images/erd.png)


```python
# Run this cell without changes
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
```


```python
# __SOLUTION__
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
```


```python
# Replace None with your code

connection = None

sql_query = None

df = None
```


```python
# __SOLUTION__

connection = sqlite3.connect('data/legos.db')

sql_query = """
            SELECT * FROM difficulty d
            INNER JOIN product_info p
            USING("Prod Id")
            INNER JOIN metrics m
            USING("Prod Id")
            """

df = pd.read_sql(sql_query, connection)
```

## Analysis Requirements

### 1. Data Exploration and Cleaning

During the data exploration phase, the datatypes of columns should be checked, the distribution of the target variable should be inspected, null values should be either removed or replaced, and duplicates (in most cases) should be dropped. 

### 2. Create a Baseline Model

In order to evaluate how youll a simple linear regression model is understanding the dependent variable, you will begin by first creating a model that predicts the mean of the dependent variable for every observation. Predicting the mean of `list_price` can be considered a highly naive model. If our simple linear regression model performs worse than this naive approach, you can safely say that it is not a very good model. 

### 3. Interpret a Correlation Heatmap

To develop a simple linear regression model, you will identify the independent variable that is most correlated with our dependent variable. To do this this you will plot a correlation heatmap to identify the variable most correlated with `list_price`.

### 4. Build a Simple Linear Regression Model

Now, create a linear regression model using the `statsmodels` library where the most correlated feature is used as the independent variable and the dependent variable is properly set. 

### 5. Interpret the Simple Linear Regression Model

Once the model has been fit, the coefficient for our independent variable, its p-value, and the coefficient confidence interval should be interpeted. You should ask ourselves whether or not the relationship your model is finding seems plausible. 

### 6. Evaluate the Simple Linear Regression Model

Before you can make a final assessment of our model, you need to compare its metrics with the baseline model created in step one, and you need to check the assumptions of linear regression.


# 1. Data Exploration and Cleaning

Inspect the dataframe by outputting the first five rows.


```python
# Replace None with your code

None
```


```python
# __SOLUTION__

df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Prod Id</th>
      <th>Min Age</th>
      <th>Max Age</th>
      <th>Difficulty Level</th>
      <th>Set Name</th>
      <th>Prod Desc</th>
      <th>Theme Name</th>
      <th>Piece Count</th>
      <th>Num Reviews</th>
      <th>Star Rating</th>
      <th>List Price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>60123</td>
      <td>7.0</td>
      <td>12.0</td>
      <td>1</td>
      <td>Volcano Supply Helicopter</td>
      <td>Fly in the exploration gear and fly out the cr...</td>
      <td>City</td>
      <td>330</td>
      <td>3.0</td>
      <td>4.3</td>
      <td>$49.99</td>
    </tr>
    <tr>
      <th>1</th>
      <td>71246</td>
      <td>7.0</td>
      <td>14.0</td>
      <td>1</td>
      <td>Adventure Time™ Team Pack</td>
      <td>Explore the Land of Ooo with Jake and Lumpy Sp...</td>
      <td>DIMENSIONS™</td>
      <td>96</td>
      <td>3.0</td>
      <td>4.7</td>
      <td>$30.362</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10616</td>
      <td>1.5</td>
      <td>5.0</td>
      <td>1</td>
      <td>My First Playhouse</td>
      <td>Learn about daily routines with this easy-to-b...</td>
      <td>DUPLO®</td>
      <td>25</td>
      <td>3.0</td>
      <td>5.0</td>
      <td>$19.99</td>
    </tr>
    <tr>
      <th>3</th>
      <td>31079</td>
      <td>8.0</td>
      <td>12.0</td>
      <td>1</td>
      <td>Sunshine Surfer Van</td>
      <td>Enjoy seaside fun with the 3-in-1 Sunshine Sur...</td>
      <td>Creator 3-in-1</td>
      <td>379</td>
      <td>5.0</td>
      <td>4.4</td>
      <td>$34.99</td>
    </tr>
    <tr>
      <th>4</th>
      <td>42057</td>
      <td>8.0</td>
      <td>14.0</td>
      <td>1</td>
      <td>Ultralight Helicopter</td>
      <td>Take to the skies with the Ultralight Helicopter!</td>
      <td>Technic</td>
      <td>199</td>
      <td>9.0</td>
      <td>4.7</td>
      <td>$19.99</td>
    </tr>
  </tbody>
</table>
</div>






Produce high-level descriptive information about your training data


```python
# Replace None with your code

None
```


```python
# __SOLUTION__

df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Prod Id</th>
      <th>Min Age</th>
      <th>Max Age</th>
      <th>Difficulty Level</th>
      <th>Piece Count</th>
      <th>Num Reviews</th>
      <th>Star Rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>8.840000e+02</td>
      <td>884.000000</td>
      <td>884.000000</td>
      <td>884.000000</td>
      <td>884.000000</td>
      <td>781.000000</td>
      <td>781.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5.821184e+04</td>
      <td>6.781674</td>
      <td>27.785068</td>
      <td>1.483032</td>
      <td>460.990950</td>
      <td>17.610755</td>
      <td>4.430602</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.388029e+05</td>
      <td>2.984551</td>
      <td>34.019566</td>
      <td>0.796898</td>
      <td>928.905788</td>
      <td>38.143280</td>
      <td>0.592083</td>
    </tr>
    <tr>
      <th>min</th>
      <td>6.300000e+02</td>
      <td>1.500000</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.800000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.113875e+04</td>
      <td>5.000000</td>
      <td>12.000000</td>
      <td>1.000000</td>
      <td>88.750000</td>
      <td>2.000000</td>
      <td>4.100000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>4.550550e+04</td>
      <td>7.000000</td>
      <td>14.000000</td>
      <td>2.000000</td>
      <td>185.500000</td>
      <td>6.000000</td>
      <td>4.600000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>7.122825e+04</td>
      <td>8.000000</td>
      <td>14.000000</td>
      <td>2.000000</td>
      <td>457.750000</td>
      <td>13.000000</td>
      <td>4.900000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2.000431e+06</td>
      <td>16.000000</td>
      <td>99.000000</td>
      <td>4.000000</td>
      <td>7541.000000</td>
      <td>367.000000</td>
      <td>5.000000</td>
    </tr>
  </tbody>
</table>
</div>



Display the number of null values for each column


```python
# Replace None with your code

None
```


```python
#__SOLUTION__
df.isna().sum()
```




    Prod Id               0
    Min Age               0
    Max Age               0
    Difficulty Level      0
    Set Name              0
    Prod Desc            20
    Theme Name            0
    Piece Count           0
    Num Reviews         103
    Star Rating         103
    List Price           40
    dtype: int64



You have four columns that contain null values. Of those four, `List Price`, which is your dependent variable, is probably the most concerning. 

In the cell below, drop all rows where `List Price` is null.


```python
# Replace None with your code

None
```


```python
#__SOLUTION__
df = df.dropna(subset=['List Price'])
```

To make things easier moving forward, in the cell below, reformat the column names so spaces have been replaced with underscores and the text has been lowered


```python
# Replace None with your code

None
```


```python
#__SOLUTION__
df.columns = df.columns.str.loyour().str.replace(' ', '_')
```

Run the cell below to check your reformatting. If it runs without throwing an error it means you have reformatted the columns correctly.


```python
assert 'min_age' in df.columns
```

Check the datatypes of the columns in the dataframe. 
> Remember, the target column and any columns you use as independent variables *must* have a numeric datatype. After inspecting the datatypes of the columns, convert columns to numeric where necessary. 


```python
# Replace None with your code

None
```


```python
#__SOLUTION__
print(df.dtypes)

df.list_price = df.list_price.str.replace('$', '').astype(float)
```

    prod_id               int64
    min_age             float64
    max_age             float64
    difficulty_level      int64
    set_name             object
    prod_desc            object
    theme_name           object
    piece_count           int64
    num_reviews         float64
    star_rating         float64
    list_price           object
    dtype: object


    <ipython-input-210-8528c0020878>:4: FutureWarning: The default value of regex will change from True to False in a future version. In addition, single character regular expressions will*not* be treated as literal strings when regex=True.
      df.list_price = df.list_price.str.replace('$', '').astype(float)


In the cell below, output the number of duplicate rows in the dataframe. If duplicates are found, drop them.


```python
# Replace None with your code

None
```


```python
#__SOLUTION__
print('Duplicated rows:', df.duplicated().sum())

df = df.drop_duplicates()
```

    Duplicated rows: 100


Visualize the distribution of the dependent variable


```python
# Replace None with your code

None
```


```python
# __SOLUTION__

fig, ax = plt.subplots(figsize=(10, 5))

ax.hist(df.list_price, bins=100)

ax.set_xlabel("Listing Price (USD)")
ax.set_ylabel("Count of LEGO Sets")
ax.set_title("Distribution of LEGO Set Prices");
```


![png](index_files/output_34_0.png)


# Create a Baseline Model

Below, create a baseline model. 

To do this you must:

1. Calculate the mean of the `list_price` column in the `train` dataframe.
2. Now that you've calculate the mean of `list_price`, create a list with the same length as the `list_price` column where every value in the list is the calculated mean.
    - Store this list in the variable `baseline_preds`.


```python
# Calculate the mean of the list_price column in the train dataframe.
mean = None

# Create a list with the same length as the list_price column 
# where every value in the list is the calculated mean.
baseline_preds = None
```


```python
#__SOLUTION__
from sklearn.metrics import r2_score, mean_squared_error

# Calculate the mean of the list_price column in the train dataframe.
mean = train.list_price.mean()

# Create a list with the same length as the list_price column where every value in the list is the mean
baseline_preds = [mean for x in train.index]

# Pass the list_price column and the baseline_preds list into the function r2_score
baseline_r2 = r2_score(train.list_price, baseline_preds)

# Pass the list_price column and the baseline_preds list 
# into the function mean_squared_error and setting squared to False
baseline_rmse = mean_squared_error(train.list_price, baseline_preds, squared=False)

print('Baseline R^2: ', baseline_r2)
print('Baseline RMSE:', baseline_rmse)
```

    Baseline R^2:  0.0
    Baseline RMSE: 70.31606908300459


Now that you have baseline predictions, you can use the predictions to calculate metrics about the model's performance. 


```python
from sklearn.metrics import r2_score, mean_squared_error

# Pass the list_price column and the baseline_preds list into the function r2_score
baseline_r2 = r2_score(train.list_price, baseline_preds)

# Pass the list_price column and the baseline_preds list 
# into the function mean_squared_error and setting squared to False
baseline_rmse = mean_squared_error(train.list_price, baseline_preds, squared=False)

print('Baseline R^2: ', baseline_r2)
print('Baseline RMSE:', baseline_rmse)
```

    Baseline R^2:  0.0
    Baseline RMSE: 70.31606908300459


**Interpret the resulting metrics for the baseline model.**

- How is the model explaining the variance of the dependent variable?
- On average, how many dollars off are the models predictions?

==SOLUTION==

The baseline model is doing a poor job at explaining the variance of the dependent variable. An $R^2$ of `0.0` can be interpreted as "The model is explaining 0% of the variance in the dependent variable."

The baseline model's predictions, on average, are mispredicted by about \$70.

# 2. Interpret a Correlation Heatmap to Build a Baseline Model

## Correlation Heatmap

Produce a heatmap showing the correlations between all of the numeric values in the data. The x and y axis labels should indicate the pair of values that are being compared, and then the color and the number should represent the correlation. 

The most important column or row shows the correlations betyouen the target (listing price) and other attributes.


```python
# Run this cell without changes

import seaborn as sns
import numpy as np
```


```python
# __SOLUTION__

import seaborn as sns
import numpy as np
```


```python
# Replace None with your code

None
```


```python
# __SOLUTION__

# Create a df with the target as the first column,
# then compute the correlation matrix
heatmap_data = train
corr = heatmap_data.corr()

# Set up figure and axes
fig, ax = plt.subplots(figsize=(5, 8))

# Plot a heatmap of the correlation matrix, with both
# numbers and colors indicating the correlations
sns.heatmap(
    # Specifies the data to be plotted
    data=corr,
    # The mask means you only show half the values,
    # instead of showing duplicates. It's optional.
    mask=np.triu(np.ones_like(corr, dtype=bool)),
    # Specifies that you should use the existing axes
    ax=ax,
    # Specifies that you want labels, not just colors
    annot=True,
    # Customizes colorbar appearance
    cbar_kws={"label": "Correlation", "orientation": "horizontal", "pad": .2, "extend": "both"}
)

# Customize the plot appearance
ax.set_title("Heatmap of Correlation Between Attributes (Including Target)");
```


![png](index_files/output_46_0.png)


Based on the heatmap, which feature is most strongly correlated with the target (`list_price`)? In other words, which feature has the strongest positive or negative correlation — the correlation with the greatest magnitude?


```python
# Replace None with the name of the feature (a string)

most_correlated_feature = None
```


```python
# __SOLUTION__

most_correlated_feature = "piece_count"
```

Create a scatter plot of that feature vs. listing price:


```python
# Replace None with your code

None
```


```python
# __SOLUTION__

fig, ax = plt.subplots()

ax.scatter(df[most_correlated_feature], df.list_price, alpha=0.5)
ax.set_xlabel(most_correlated_feature)
ax.set_ylabel("listing price")
ax.set_title("Most Correlated Feature vs. Listing Price");
```


![png](index_files/output_52_0.png)


Assuming you correctly identified `piece_count` (the number of pieces in the LEGO set) as the most correlated feature, you should have a scatter plot that shows a fairly clear linear relationship betyouen that feature and the target. It looks like you are ready to proceed with creating a simple linear regression model.

# 3. Build a Simple Linear Regression Model

Now, you'll build a linear regression model using just that feature. 

In the cell below, fit a statsmodels linear regression model to the data and output a summary for the model. 


```python
import statsmodels.formula.api as smf

# Replace None with your code

model = None
```


```python
# __SOLUTION__

import statsmodels.formula.api as smf

model = smf.ols('list_price ~ piece_count', df).fit()
model.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>       <td>list_price</td>    <th>  R-squared:         </th> <td>   0.743</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.742</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   2141.</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Thu, 09 Sep 2021</td> <th>  Prob (F-statistic):</th> <td>6.91e-221</td>
</tr>
<tr>
  <th>Time:</th>                 <td>16:11:57</td>     <th>  Log-Likelihood:    </th> <td> -3749.0</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>   744</td>      <th>  AIC:               </th> <td>   7502.</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   742</td>      <th>  BIC:               </th> <td>   7511.</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
       <td></td>          <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>   <td>   11.4565</td> <td>    1.593</td> <td>    7.192</td> <td> 0.000</td> <td>    8.329</td> <td>   14.584</td>
</tr>
<tr>
  <th>piece_count</th> <td>    0.0875</td> <td>    0.002</td> <td>   46.273</td> <td> 0.000</td> <td>    0.084</td> <td>    0.091</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>1141.526</td> <th>  Durbin-Watson:     </th>  <td>   1.961</td> 
</tr>
<tr>
  <th>Prob(Omnibus):</th>  <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>464258.231</td>
</tr>
<tr>
  <th>Skew:</th>           <td> 8.756</td>  <th>  Prob(JB):          </th>  <td>    0.00</td> 
</tr>
<tr>
  <th>Kurtosis:</th>       <td>124.117</td> <th>  Cond. No.          </th>  <td>    978.</td> 
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



# 4. Interpret the Simple Linear Regression Model

Now that the model has been fit, you should interpret the model parameters. 

Specifically:
- What do the coefficients for the intercept and independent variable suggest about the dependent variable?
- Are the coefficients found to be statistically significant?
- What are the confidence intervals for the coefficients?
- Do the relationships found by the model seem plausible? 

==SOLUTION==

This model suggests that on average, the price of a lego set increases betyouen 0.081 and 0.089 US dollars when the piece count of the lego set is increased by 1. This model finds that when there are 0 pieces, the average price of a lego set is betyouen \\$8.955 and \\$15.640. Given that a lego set must have pieces, this intercept is not very useful. The coefficients for the intercept and the piece count variable are both statistically significant (p<0.05).

# 5. Evaluate the Simple Linear Regression Model

Now that the model parameters have been interpreted, the model must be assessed based on predictive metrics and whether or not the model is meeting the assumptions of linear regression. 

### Compare the $R^2$ and the Root Mean Squared Error of the simple linear regression model with the baseline model. 


```python
# Replace None with your code
model_r2 = None

model_rmse = None

print('Baseline R^2: ', baseline_r2)
print('Baseline RMSE:', baseline_rmse)
print('----------------------------')
print('Regression R^2: ', model_r2)
print('Regression RMSE:', model_rmse)
```

    Baseline R^2:  0.0
    Baseline RMSE: 70.31606908300459
    ----------------------------
    Regression R^2:  None
    Regression RMSE: None



```python
#__SOLUTION__
model_r2 = r2_score(df.list_price, model.predict())

model_rmse = mean_squared_error(df.list_price, model.predict(), squared=False)

print('Baseline R^2: ', baseline_r2)
print('Baseline RMSE:', baseline_rmse)
print('------------------------------------')
print('Regression R^2: ', model_r2)
print('Regression RMSE:', model_rmse)
```

    Baseline R^2:  0.0
    Baseline RMSE: 70.31606908300459
    ------------------------------------
    Regression R^2:  0.7426440497844149
    Regression RMSE: 37.33856230937886


### Interpret the model metrics

==SOLUTION==

**$R^2$**:

- The simple linear regression model is explains approximately 76% of the dependent variable's variance. 
- The simple linear regression model produced a 76 point improvement over the baseline model. 

**RMSE**:

- The simple linear regression model's predictions are off, on average, by about \\$35
- The simple linear regression model's predictions are about \\$35 more accurate than the baseline model.

### Check the assumptions of simple linear regression

#### Investigating Linearity

First, let's check whether the linearity assumption holds.


```python
# Run this cell without changes

preds = model.predict()
fig, ax = plt.subplots()

perfect_line = np.arange(df.list_price.min(), df.list_price.max())
ax.plot(perfect_line, linestyle="--", color="orange", label="Perfect Fit")
ax.scatter(df.list_price, preds, alpha=0.5)
ax.set_xlabel("Actual Price")
ax.set_ylabel("Predicted Price")
ax.legend();
```


![png](index_files/output_69_0.png)



```python
# __SOLUTION__
preds = model.predict()
fig, ax = plt.subplots()

perfect_line = np.arange(df.list_price.min(), df.list_price.max())
ax.plot(perfect_line, linestyle="--", color="orange", label="Perfect Fit")
ax.scatter(df.list_price, preds, alpha=0.5)
ax.set_xlabel("Actual Price")
ax.set_ylabel("Predicted Price")
ax.legend();
```


![png](index_files/output_70_0.png)


Are you violating the linearity assumption?

==SOLUTION==

We have a few outliers that are deviating from the assumption, but in general it looks like you have a linear relationship (not violating this
assumption)

#### Investigating Normality

Now let's check whether the normality assumption holds for our model.


```python
# Run this code without changes
import scipy.stats as stats
import statsmodels.api as sm

residuals = (df.list_price - preds)
sm.graphics.qqplot(residuals, dist=stats.norm, line='45', fit=True);
```


![png](index_files/output_74_0.png)



```python
# __SOLUTION__
import scipy.stats as stats
residuals = (df.list_price - preds)
sm.graphics.qqplot(residuals, dist=stats.norm, line='45', fit=True);
```


![png](index_files/output_75_0.png)


Are you violating the normality assumption?

==SOLUTION==

Our outliers are again causing problems. This
is bad enough that you can probably say that you
are violating the normality assumption

#### Investigating Homoscedasticity

Now let's check whether the model's errors are indeed homoscedastic or if they violate this principle and display heteroscedasticity.


```python
# Run this cell without changes
fig, ax = plt.subplots()

ax.scatter(preds, residuals, alpha=0.5)
ax.plot(preds, [0 for i in range(len(df))])
ax.set_xlabel("Predicted Value")
ax.set_ylabel("Actual - Predicted Value");
```


![png](index_files/output_79_0.png)



```python
# __SOLUTION__ 
fig, ax = plt.subplots()

ax.scatter(preds, residuals, alpha=0.5)
ax.plot(preds, [0 for i in range(len(df))])
ax.set_xlabel("Predicted Value")
ax.set_ylabel("Actual - Predicted Value");
```


![png](index_files/output_80_0.png)


Are you violating the homoscedasticity assumption?

==SOLUTION==

This is not the worst "funnel" shape, although
the residuals do seem to differ some based on
the predicted price. We are probably violating
a strict definition of homoscedasticity.

### Linear Regression Assumptions Conclusion

Given your ansyours above, how should you interpret our model's coefficients? Do you have a model that can be used for inferential as youll as predictive purposes? What might your next steps be?

==SOLUTION==

Our confidence in the piece count coefficient should not be too high, since
you are violating or close to violating more than one of the
assumptions of linear regression. This really only should be used
for predictive purposes.

A good next step here would be to start trying to figure out why
our outliers behave the way they do. Additionally, adding other variables
to the model (Moving beyond simple linear regression) may help meet the assumptions.

# Level Up: Project Enhancements

After completing the project, you could consider the following enhancements if you have time:

* Identify and remove outliers, then redo the analysis
* Compile the data cleaning code into a function
* Identify lego sets with high or low value for LEGO buyers, using the differences betyouen actual and predicted prices
* [Log the target variable](https://www.codegrepper.com/code-examples/python/log+transform+pandas+dataframe) and see if that improves the model assumptions.
* Conduct statistical tests using the numeric features in the dataset to make inferences about the population of LEGO sets

## Summary

Well done! As you can see, regression can be a challenging task that requires you to make decisions along the way, try alternative approaches, and make ongoing refinements.
