# Intro data science - portfolio assignment 3: Penguins dataset


```python
import pandas as pd
import seaborn as sns
```


```python
# store the data set in a variable
penguins = sns.load_dataset('penguins')
```


```python
# let's take a look at the dataset
penguins
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
      <th>species</th>
      <th>island</th>
      <th>bill_length_mm</th>
      <th>bill_depth_mm</th>
      <th>flipper_length_mm</th>
      <th>body_mass_g</th>
      <th>sex</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Adelie</td>
      <td>Torgersen</td>
      <td>39.1</td>
      <td>18.7</td>
      <td>181.0</td>
      <td>3750.0</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Adelie</td>
      <td>Torgersen</td>
      <td>39.5</td>
      <td>17.4</td>
      <td>186.0</td>
      <td>3800.0</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Adelie</td>
      <td>Torgersen</td>
      <td>40.3</td>
      <td>18.0</td>
      <td>195.0</td>
      <td>3250.0</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Adelie</td>
      <td>Torgersen</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Adelie</td>
      <td>Torgersen</td>
      <td>36.7</td>
      <td>19.3</td>
      <td>193.0</td>
      <td>3450.0</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>339</th>
      <td>Gentoo</td>
      <td>Biscoe</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>340</th>
      <td>Gentoo</td>
      <td>Biscoe</td>
      <td>46.8</td>
      <td>14.3</td>
      <td>215.0</td>
      <td>4850.0</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>341</th>
      <td>Gentoo</td>
      <td>Biscoe</td>
      <td>50.4</td>
      <td>15.7</td>
      <td>222.0</td>
      <td>5750.0</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>342</th>
      <td>Gentoo</td>
      <td>Biscoe</td>
      <td>45.2</td>
      <td>14.8</td>
      <td>212.0</td>
      <td>5200.0</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>343</th>
      <td>Gentoo</td>
      <td>Biscoe</td>
      <td>49.9</td>
      <td>16.1</td>
      <td>213.0</td>
      <td>5400.0</td>
      <td>Male</td>
    </tr>
  </tbody>
</table>
<p>344 rows × 7 columns</p>
</div>



## Univariate analysis

Univariate analysis is the most simple form of data analysis. Only 1 variable will be analyzed - 1 column in this case. Causes or relationships are not taken into account in a univariate analysis. The purpose is to retrieve data, summarize it and find patterns in the data.

### Assignment: Perform a univariate analysis on all the categorical data of the penguins dataset.


```python
# there are three columns containing categorical data in this dataset:
# - species
# - island
# - sex

# let's see which unique values each of these columns has
penguins['species'].unique()
```




    array(['Adelie', 'Chinstrap', 'Gentoo'], dtype=object)




```python
penguins['island'].unique()
```




    array(['Torgersen', 'Biscoe', 'Dream'], dtype=object)




```python
penguins['sex'].unique()

# note: there are two ways of selecting a column. The above is one way, the other is as follows:
# penguins.sex.unique()
# the first option is convenient when, for example, a column name contains spaces 
# you could then do the following:
# penguins['contains spaces']
```




    array(['Male', 'Female', nan], dtype=object)



As you can see, there are penguins that have the value "NaN" as sex. This stands for "Not a Number", and is Pandas' default missing value marker. In this case, the sex of these penguins is unknown.

Missing values in a dataframe can be filled using various methods. For example, dataframe.ffill(), which stands for "forward fill", will propagate the last valid observation forward.

Let's demonstrate this with an example.


```python
# the first 10 penguins without the use of ffill()
penguins.head(10)
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
      <th>species</th>
      <th>island</th>
      <th>bill_length_mm</th>
      <th>bill_depth_mm</th>
      <th>flipper_length_mm</th>
      <th>body_mass_g</th>
      <th>sex</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Adelie</td>
      <td>Torgersen</td>
      <td>39.1</td>
      <td>18.7</td>
      <td>181.0</td>
      <td>3750.0</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Adelie</td>
      <td>Torgersen</td>
      <td>39.5</td>
      <td>17.4</td>
      <td>186.0</td>
      <td>3800.0</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Adelie</td>
      <td>Torgersen</td>
      <td>40.3</td>
      <td>18.0</td>
      <td>195.0</td>
      <td>3250.0</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Adelie</td>
      <td>Torgersen</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Adelie</td>
      <td>Torgersen</td>
      <td>36.7</td>
      <td>19.3</td>
      <td>193.0</td>
      <td>3450.0</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Adelie</td>
      <td>Torgersen</td>
      <td>39.3</td>
      <td>20.6</td>
      <td>190.0</td>
      <td>3650.0</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Adelie</td>
      <td>Torgersen</td>
      <td>38.9</td>
      <td>17.8</td>
      <td>181.0</td>
      <td>3625.0</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Adelie</td>
      <td>Torgersen</td>
      <td>39.2</td>
      <td>19.6</td>
      <td>195.0</td>
      <td>4675.0</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Adelie</td>
      <td>Torgersen</td>
      <td>34.1</td>
      <td>18.1</td>
      <td>193.0</td>
      <td>3475.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Adelie</td>
      <td>Torgersen</td>
      <td>42.0</td>
      <td>20.2</td>
      <td>190.0</td>
      <td>4250.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# wait... maybe we should display only the sex column
penguins['sex'].head(10)
```




    0      Male
    1    Female
    2    Female
    3       NaN
    4    Female
    5      Male
    6    Female
    7      Male
    8       NaN
    9       NaN
    Name: sex, dtype: object




```python
# let's store these penguins in a new variable and perform ffill() on the sex column.
penguinsFfill = penguins['sex'].head(10).ffill()

penguinsFfill
```




    0      Male
    1    Female
    2    Female
    3    Female
    4    Female
    5      Male
    6    Female
    7      Male
    8      Male
    9      Male
    Name: sex, dtype: object



The missing values have been filled.

...

### Back to the analysis!


```python
# we can use the value_counts() method to see how many records there are of each unique value in a column
penguins['species'].value_counts()
```




    Adelie       152
    Gentoo       124
    Chinstrap     68
    Name: species, dtype: int64



Looks like chinstrap penguins are in the minority. Sad!


```python
penguins['island'].value_counts()
```




    Biscoe       168
    Dream        124
    Torgersen     52
    Name: island, dtype: int64



Biscoe seems to be a very popular island for penguins to live on.


```python
penguins['sex'].value_counts()
```




    Male      168
    Female    165
    Name: sex, dtype: int64



Pretty equal. But wait - weren't there missing values as well?

Correct. However, according to the pandas docs, value_counts() takes a parameter "dropna" which is a boolean value that defaults to true. This excludes NaN values from the value counts.


```python
penguins['sex'].value_counts(dropna=False)
```




    Male      168
    Female    165
    NaN        11
    Name: sex, dtype: int64



Now it does include NaN values.

### Plotting

We can use plotting to visualize data in various ways.


```python
# standard convention for referencing the matplotlib API
import matplotlib.pyplot as plt
plt.close('all')
```


```python
# create a bar plot of the amount of penguins per species
penguins['species'].value_counts().plot(kind='bar')

# add a title
plt.title('Amount of penguins per species')

# add an x-axis label
plt.xlabel('Species')

# add an y-axis label
plt.ylabel('Amount')

# display the plot
plt.show()
```


    
![png](output_28_0.png)
    



```python
# create a bar plot of the amount of penguins living on each island
# note: we can use "barh" to create a horizontal bar plot. this would be
# especially useful if there were a lot of different islands
penguins['island'].value_counts().plot(kind='barh')

# add a title
plt.title('Amount of penguin inhabitants per island')

# add an x-axis label
plt.xlabel('Inhabitants')

# add an y-axis label
plt.ylabel('Island')

# display the plot
plt.show()
```


    
![png](output_29_0.png)
    



```python
# create a pie chart plot to visualize the male to female ratio of the penguins
explode = (0, 0, 0.1)
penguins['sex'].value_counts(dropna=False).plot(kind='pie', labels=None, autopct='%1.1f%%', explode=explode)

# equal aspect ratio ensures the pie is drawn as a circle
plt.axis('equal')

# add a title
plt.title('Penguins male and female percentages')

# add a legend
labels = ['Male', 'Female', 'Unknown']
plt.legend(labels=labels)

# display the plot
plt.show()
```


    
![png](output_30_0.png)
    

