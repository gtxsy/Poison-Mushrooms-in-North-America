![IronHack Logo](https://s3-eu-west-1.amazonaws.com/ih-materials/uploads/upload_d5c5793015fec3be28a63c4fa3dd4d55.png)

# An Exploration of Poison Mushrooms in North America 

## Overview

In this study, we will process and analyze data about poisonous mushrooms in North America, in order to better understand the physical chracteristics that can help determine if a given mushroom is edible or not. We will use information gathered from our analysis in order to create a supervised machine learning model that will be able to predict if a mushroom is safe to consume or not, based on these same physical attributes.

Commit Log from Original Project Repository:
https://github.com/gtxsy/datamia0619/commits/master

## The Data

The data set we explore here describes physical attributes and properties of over 8,000 North American mushrooms. Most significantly, each data point in the set distinguishes the mushroom as either poisonous or edible. This data was taken from OpenML.org (https://www.openml.org/d/24), extracted from "The Audubon Society Field Guide to North American Mushrooms" (1981).


## Loading the Data

In order to work with this data set, we first had to ingest it. The file containing the data was in CSV format, so after downloading it, we used the `read_csv` method to read the data into a Pandas data frame.

```python
import pandas as pd

original_mushroom_data = pd.read_csv('mushrooms.csv')
```

Next, we checked that the data set loaded correctly by checking the shape of the data frame, looking at the first few rows of the data, and evaluating the column names to ensure that they look as we would expect.

```python
original_mushroom_data.shape

(8124, 23)
```

```python
original_mushroom_data.head(5)
```

![Mushroom Data](./images/original_dataset.png)

Below, we can see the original code-values contained in each column, representing the physical attributes of each mushroom:

![Mushroom Column Values](./images/original_dataset_columns.png)


From this initial assessment, we have successfully loaded the data set and examined the shape and attributes contained within the dataset. We can now move on to evaluating the quality of the data, cleaning it, and wrangling it so that it is ready to be analyzed and modeled.

## Data Wrangling and Cleaning

The first obstacle in this dataset is that every value across all columns are stored as a codified string-value, with decoded values symbolized in the previous table. Our first task should be to create a duplicate set of data where all string-values are hot-encoded into binary (0 or 1) values, assuming that no ordinal data is available.

Before we convert this data from string to integer values, let's first remove the extraneous single quotes ' that appear in every cell to avoid problems further down the road:

```python
original_mushroom_data = original_mushroom_data.applymap(lambda x: x.replace("'", ""))
```

Next, we can check if the dataset itself is complete, by looking for missing (null or NaN) values within the dataframe:

```python
original_mushroom_data.isna().sum()
```

![Mushroom Missing Values](./images/original_dataset_missing.png)

The output from our code, shown above, tells us that the original dataset is complete with no missing values. However, you can observe that one of our column names "bruises%3F" got mangled somehow, either when importing the CSV file or originally from the source. We can fix this easily:

```python
original_mushroom_data.rename(columns={"bruises%3F": "bruises"}, inplace=True)
```

Finally, we can pass our complete and cleaned data to the Pandas "get_dummies" function in order to convert all of our string-values into boolean (numerical) data. We will call this new numerical copy of our dataframe "decoded_mushroom_data".

```python
decoded_mushroom_data = pd.get_dummies(original_mushroom_data)
```

![Decoded Mushroom Data](./images/decoded_dataset.png)


As a precaution, let's check to make sure that each column in our new "decoded_mushroom_data" is of the correct data type:

```python
decoded_mushroom_data.dtypes
```

![Decoded Mushroom Columns](./images/decoded_dataset_columns.png)


## Data Storage

Now that the data has been ingested and cleaned, we can store it in a MySQL database. To do this we first created a *mushroom* database using the `CREATE DATABASE` command in the MySQL terminal.

```sql
CREATE DATABASE mushroom;
```

We then used `pymsql` and `sqlalchemy` to connect and write the data to the database. The Python library `getpass` allows us to securely store the password used to connect to the MySQL database. The code below shows how we get that password and then use it to connect:

```python
import pymysql
import getpass
from sqlalchemy import create_engine

p = getpass.getpass(prompt='Password: ', stream=None) 
engine = create_engine('mysql+pymysql://root:'+p+'@localhost/mushroom')
```

After successful connection, we can now write both our categorical and numerical dataframes to the new MySQL database as tables called "mushroom_cat" and "mushroom_num":

```python
original_mushroom_data.to_sql('mushroom_cat', engine, if_exists='replace', index=False)
decoded_mushroom_data.to_sql('mushroom_num', engine, if_exists='replace', index=False)
```

To read the stored data back at a later date as a dataframe, we use the Pandas `read_sql_query` method:

```python
pd.read_sql_query('SELECT * FROM mushroom.mushroom_cat', engine)
pd.read_sql_query('SELECT * FROM mushroom.mushroom_num', engine)
```

We can also export the cleaned data set as a CSV file, to be imported into Tableau for exploration and reporting later on:

```python
original_mushroom_data.to_csv('./export/categorical_mushroom_data.csv', index=False)
decoded_mushroom_data.to_csv('./export/numerical_mushroom_data.csv', index=False)
```

## Data Exploration and Analysis

After cleaning and storing the data, we can begin exploring and analyzing the data. Each row in the data set represents the physical properties of different observed mushrooms and each column represents attributes belonging to those properties. We looked through these attributes to determine which ones would potentially yield the most informative insights, based on the total number of unique values in each column and how easy it should be to observe each property. Below is a summary of the steps of our explorations, followed by a series of data visualizations conveying insights discovered:

* Poisonous vs Edible Mushrooms
* Mushroom Color Caps
* Distribution of Poison Mushrooms by Color Caps
* Total Percent of Poison Mushrooms by Color Caps
* Mushroom Odors
* Total Percent of Poison Mushrooms by Odors

The first thing we wanted to look at was the distribution of poisonous vs edible mushrooms in our dataset, or how many instances of each are recorded:

![Poison Distribution](./images/distribution_poison.png)

Here we can see the data is almost evenly split, with nearly half of all records contaning poisonous mushrooms, and the other half edible.

*******

Now, let's explore the distribution of different colors of the mushrooms' caps: 

![Cap Color Distribution](./images/distribution_colors.png)

The spread of cap colors across all the mushrooms in our dataset is not evenly spread here, with the majority of mushroom caps having earthy colors like Brown, Grey, Red, and Yellow. We can build on top of this, by exploring the relationship of the cap colors with the distribution of poisonous mushrooms:

![Poison Distribution by Cap Color](./images/distribution_colors_by_poison.png)

The chart above now provides extra insight into how likely a mushroom is to be poison, based on the color of it's cap. Because the distribution of cap colors is not very even, the graph is harder to read for some cap colors. We can fix this by looking at a pecentage-based breakdown which is easily visualized using pie charts:

![Cap Color Pie Charts 1](./images/pie_colors_1.png)
![Cap Color Pie Charts 2](./images/pie_colors_2.png)

Above, we can see that the distribution of poisonous (colored-Red) and edible (colored-Green) mushrooms is fairly even (50/50 split) across all cap colors, with the exception of Green and Purple-capped mushrooms which are both exclusively edible.

*******

Although we discovered that the color of a mushroom's cap is generally not enough to determine if it is poisonous or not, there may be other properties that serve as a better indicator. If our sense of sight is not much help (at least in the previous case), what about our sense of smell?

![Odor Distribition](./images/distribution_odors.png)

Here we can see the distribution of odors in our mushroom's dataset. Observe that almost half of all mushrooms (3528 out of 8124) have no odor whatsoever, while the remainder is unevenly distributed. Similar to our exploration of cap colors, we can look at percentage-based distribution of poison and edible mushrooms across all of the different odors:

![Odor Pie Charts 1](./images/pie_odors_1.png)
![Odor Pie Charts 2](./images/pie_odors_2.png)

Above, we can see that mushrooms with odors of "almond" and "anise" are exclusively edible mushrooms, whereas all other columns -- except for those with no smell -- are exclusively poisonous mushrooms.


## Feature Selection

Now, with a better understanding of the data and some insight as to how our column-attributes can help determine if a given mushroom is poisonous or not, we can proceed to prep our data set for modeling. Our goal is to train a regression model to predict if any mushroom is poisonous or not based on its physical attributes. 

```python
decoded_mushroom_data.corr()
```

![Correlations Matrix](./images/correlations.png)


The generated matrix shown above displays the correlation coefficients across all combinations of columns in our numerical dataset. We can explore this further, by looking exclusively at correlations within the "class_p" column to examine the relationships of all columns in relation to our mushroom's poisonous property with this code:

```python
pd.DataFrame(decoded_mushroom_data.corr()['class_p'].sort_values())
```

The output from the code above gives a list, in sorted order, of all columns and their correlation to the 
"class_p" property, which represents the mushrooms in the dataset that are poisonous or not. The correlation coefficients here can help us select the best features to train the data on. We should pick those features that have the strongest correlations (both positive and negative), in order to minimize noise and optimize our regression model.

We can then select all column names with correlations higher than 0.4 and lower than -0.4:

```python
correlations = pd.DataFrame(decoded_mushroom_data.corr()['class_p'].sort_values())
X_columns = []

for index, row in correlations.iterrows():
    if (0.4 < row.class_p < 1):
        X_columns.append(index)
    elif (-1 < row.class_p < -0.4):
        X_columns.append(index)
        
X_columns 
```

The output from the code above, then, will yield our final list of feature X-components to train our model on:

```python
['odor_n',
 'ring-type_p',
 'gill-size_b',
 'bruises_t',
 'stalk-surface-above-ring_s',
 'stalk-surface-below-ring_s',
 'spore-print-color_n',
 'population_v',
 'ring-type_l',
 'spore-print-color_h',
 'bruises_f',
 'gill-color_b',
 'gill-size_n',
 'stalk-surface-below-ring_k',
 'stalk-surface-above-ring_k',
 'odor_f']
```

## Model Training and Evaluation

Our final goal is to create a model to accurately predict if a given mushroom in our dataset is poisonous or not. We can do this using supervised machine learning, with either a Logistic Regression model, or KNN. Here we will try both.

We'll start by splitting the data into two sets: one for training a regression model, and one for testing predictions on that model:

```python
from sklearn.model_selection import train_test_split

# Split X and Y sets into 80% training data and 20% testing data:
X_data = decoded_mushroom_data[X_columns]
Y_data = decoded_mushroom_data['class_p']
X_train, X_test, Y_train, Y_test  = train_test_split(X_data, Y_data, test_size = 0.2)

```

With our data prepared, we can now create and fit our model onto the training data. We will begin by looking at a Logistic Regression model:

```python
from sklearn import linear_model

# Initialize the Logistic Regression model
log_model = linear_model.LogisticRegression(solver='lbfgs')

# Train the model
log_model.fit(X_train, Y_train) 

```

With the model created and fitted onto our training data, we can begin to test predictions on our test data:

```python
from sklearn import metrics

# Create predictions based on test data, and then evaluate the Linear Regression model
y_pred = log_model.predict(X_test)
print("Accuracy Score:", metrics.accuracy_score(Y_test, y_pred))
metrics.confusion_matrix(Y_test, y_pred)
```

Example output from code run above:

```python
Accuracy Score: 0.9790769230769231
array([[808,  19],
       [ 15, 783]])
```

Shown above, the Linear Regression model predicted correctly that 808 of the given mushrooms would be poisionous, and 783 would be edible. Out of over 1,600 data points that we tested on, it only had 34 incorrect guesses, resulting in a final accuracy score of 97.9%

*******

Finally, we will try a KNN model in order to improve the accuracy of our predictions:

```python
from sklearn.neighbors import KNeighborsClassifier

# Create an alternate machine learning model with KNN
neighbor_model = KNeighborsClassifier(n_neighbors=3)
neighbor_model.fit(X_train, Y_train)

# Create predictions based on test data, and then evaluate the KNN model
y_pred_knn = neighbor_model.predict(X_test)
print("Accuracy Score:", metrics.accuracy_score(Y_test, y_pred_knn))
metrics.confusion_matrix(Y_test, y_pred_knn)
```

Example output:

```python
Accuracy Score: 0.9815384615384616
array([[808,  19],
       [ 11, 787]])
```

In this example, our KNN model was able to improve predictions of edible mushrooms (787 correct compared to 783 in the previous model). Although there was no improvement in the accuracy of predicting poisonous mushrooms, we still observe a 0.25% increase in the total accuracy of our predictions with this model for a final accuracy score of 98.2%

## Conclusion

In this study, we have explored a dataset consisting of over 8,000 North American mushrooms and 23 distinct physical attributes. Our data analysis workflow started with data ingestion, wrangling and cleaning, external storage, and analysis, before finally creating a superivsed machine learning model consisting of feature selection/engineering, model selection, and model evaluation. In the end, we were able to predict if a given mushroom is safe to eat with a final accuracy of 98.2% using a KNN machine learning model.
