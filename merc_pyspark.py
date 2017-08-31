#--------------------------- Importing essential libraries -----------------------------------------------

from pyspark import SparkContext
from pyspark import SQLContext
from pyspark.sql.functions import col, isnan, lit, sum
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.feature import PCA
from pyspark.ml.regression import RandomForestRegressor as RF
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator

sc = SparkContext()
sqlContext = SQLContext(sc)

#--------------------------- Importing data -----------------------------------------------

df = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferSchema = True).load('/Users/amrutshintre/Downloads/train.csv')

#--------------------------- Basic Inspection ----------------------------------------------------

df.show() # Taking a glimpse at the data
df.count() # Counting number of rows
len(df.columns) # Counting number of columns
df.printSchema() # Looking at the type of data
df.select("y").show() # Taking a look at our dependent variable

# Checking for missing values

def count_null(c, nan_as_null = False):
    pred = col(c).isNull() & (isnan(c) if nan_as_null else lit(True))
    return sum(pred.cast("integer")).alias(c)

df.agg(*[count_null(c) for c in df.columns]).show()

# There are zero missing values.

#--------------------------- Feature Engineering -------------------------------------------------------

# Selecting categorical columns for label encoding. Also including "ID" column 
# so that we can join the label encoded columns back to the main dataframe.


df1 = df.select(["ID", "X0", "X1", "X2", "X3", "X4", "X5", "X6", "X8"])

cols = df1.columns
    
indexers = [StringIndexer(inputCol=x, outputCol=x+'_tmp') for x in cols if x != "ID" ]

# Label Encoding
for i in indexers:
    df1 = i.fit(df1).transform(df1)

# droping the initial categorical columns from the dataframes.
for i in cols:
    if i != "ID":
        df1 = df1.drop(i)
        df = df.drop(i)

# Joining the dataframe label encoded features to our main dataframe
df_final = df.join(df1, on = "ID")


features = df_final.drop("ID") # Data with features and label.
feature = features.drop("y") # Data with only features
feat = feature.columns  # Extracting column names of the features.

assembler_features = VectorAssembler(inputCols=feat, outputCol='features_h')
pca = PCA(k = 100, inputCol = 'features_h', outputCol = "pca_features") # Applying PCA to reduce the dimentionality

pipeline = Pipeline(stages = [assembler_features, pca]) 

all_data = pipeline.fit(features).transform(features)

# Splitting the data into training and testing dataset.
training_data, test_data = all_data.randomSplit([0.8, 0.2], seed = 0)

#--------------------------- Model Start -----------------------------------------------

# Random Forrest Regression
rf = RF(labelCol = "y", featuresCol = "pca_features", numTrees = 200)

model = rf.fit(training_data)
validation = model.transform(test_data)

evaluator = RegressionEvaluator(predictionCol = "prediction", labelCol = "y", metricName = "r2")

evaluator.evaluate(validation)


