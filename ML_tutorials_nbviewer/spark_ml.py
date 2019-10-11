import pyspark
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
from pyspark.sql.functions import lit

"""
INFO: pyspark works with java 8 so export it before running the noteboo
export JAVA_HOME=$(/usr/libexec/java_home -v 1.8)
you can added it on ~/.bashrc
"""


print("Using pyspark versio: {}".format(pyspark.__version__))

file_name = "ml-100k/u.item"     
def load_dataset():
    
    """
    This method opens u.item and returns the movie title as key and its ID as value (key-value pairs)
    """
    movie_names = {}
    infile = open(file_name, "r", encoding="utf8", errors="ignore")
    for line in infile:
        fields = line.split("|")
        movie_names[int(fields[0])] = fields[1]
    
    return movie_names

def parse_input(line):
    
    
    """
    This method returns (UserID, MovieID, Ratings) from u.data, we need u.item to match movie ID and movie Title
    """
    
    fields = line.value.split()
    return Row(userID = int(fields[0]), movieID = int(fields[1]), rating = float(fields[2]))
    
    

#print("movies ", load_dataset())
spark = SparkSession.builder.appName("first_spark").getOrCreate() #Get or crate for recoveries
lines = spark.read.text("ml-100k/u.data").rdd

#Lets parse it!
ratingsRDD = lines.map(parse_input)
#Convert it to dataframe and cache it not to be building it several times!
ratings = spark.createDataFrame(ratingsRDD).cache()
print(ratings.collect())
