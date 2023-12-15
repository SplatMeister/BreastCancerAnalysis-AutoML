# BreastCancerAnalysis-AutoML

https://raw.githubusercontent.com/jbrownlee/Datasets/master/breast-cancer.csv

The breast cancer dataset is commonly known as the “Wisconsin Breast Cancer Dataset”. The data set originated by performing fine needle aspirates (FNA). Where collecting various feature from the sample and documenting it. 
Introduction
The breast cancer dataset is commonly known as the “Wisconsin Breast Cancer Dataset”. The data set originated by performing fine needle aspirates (FNA). Where collecting various feature from the sample and documenting it. 
The data set is predominantly used to classify if breast masses benign or malignant. This data set has helped in breast cancer diagnosis. 
Breast cancer data set consists of 10 attributes that includes the age category of the patient to the class attribute. Prior to any EDA, it is important to identify the meaning of each attribute and what insights each attribute may provide.
Attribute	Meaning	Insight
Age	Age category of the patient at the time of diagnostic. 	Provides age distribution of patients. 
Menopause	Menopause status of the patient.	Information on the hormonal status of each patient and assess the influence on hormonal factors effect breast cancer.
Tumor Size	Tumor size in millimeters (mm).	Provide the size of the tumor and the likely hood of malignancy. 
Inv-nodes	The number of lymph nodes involved.	The ranges of lymph node and determining the stage and prognosis of breast cancer.
Node-caps	Indicates if there is presence or absence of capsule within the lymph nodes.	Relationship between node capsular adhesion and the likelihood of malignancy or metastasis. 
Deg-malig	Represents the malignancy of the tumor between a scale from 0 to 2.	Higher value indicates higher degree of malignancy. 
Breast	Represents the side of the tumor is located.	To determine which side the tumor is located.
Breast-quad	The quadrant the breast, where the tumor is located. 	Based on the quadrants of the breast, provides the location of the tumor on the breast. 
Irradiat	If the patient received radiation therapy or not.	Helps to understand the impact of radiation therapy. 
Class	Represents the presence of absence of breast cancer recurrence.	Can provide what are the attributes that effect breast cancer recurrence. 

Table 1 Attribute definition and meaning.
The above attributes provide an in-depth information of breast cancer diagnostics. Which includes the patient information that includes demographics and the tumor characteristics. Furthermore, providing the treatment information if the individual has received any treatment. The objective is to use this information to uncover correlations and insights to predict breast cancer. 
2.1 Convert ‘deg-malig’ column values to numeric. 

To analyze and perform conversion of data types to numeric. Pandas’ library is used to load the data set and assigned to a variable ‘df’’.
 Table 2 Breast Cancer Data frame
The above given table depicts the attributes of the data set and from the first glance it is evident that all the observations are ‘object’ data type.
 
Table 3 Data frame data types
 Based the ‘deg-malig’ attribute and the observations requires to be converted to from object data type to numeric data types. To change the data types from sklearn library, ‘LabelEncorder’ is used to transform the data types to numeric. 
 
Figure 1 Label Encoding
After performing label encoding in the selected ‘deg-malig’ attribute, using ‘dtype’ function again to check if the relevant data is changed to numeric and is reflected as given. 
 
Figure 2 Change Data Type for 'deg-malig'
2.2 Data Pre-Processing.

Based on the data set it is important to have a consistent data set that can be used for machine learning. Firstly, identifying if the data set has any missing values in the data frame. In order to determine what the missing values are, using the ‘isnull().sum()’ function to count the total missing values based on each attribute. 
 
Table 4 Summary of Null Values

Based on the above given table, there are visible missing values. Under ‘node-caps’ attribute there are 8 missing values and under ‘breast-quad’ there is just 1 missing value. All the missing values are categorical data, and it is difficult to replace these values. Furthermore, the total of 9 values which are missing are significantly lower than expected. Therefore, these values are dropped from the data frame using the ‘dropna’ function. After dropping these values, checking if these are dropped from the data frame. 
 
Figure 3 Drop NA Values

Since most of the data that are present in the breast cancer data set are categorical data apart from ‘age’, 'tumor-size', 'inv-nodes' and 'deg-malig'. Therefore, it is difficult to carry out further preprocessing for the data set. 
2.3 Exploratory Data Analysis.

The data set has many hidden insights, and it is important to identify the relationship between each attribute. Specifically, how the target variable indicates the diagnosis. Where the patient has experienced any recurrence of breast cancer or not. On the other hand, the predicted variables provide additional information about the patient's characteristics. Prior to exploratory data analysis it is important to recognize the unique values each predicted variables consist of. To derive these details using the ‘unique()’ function.



Attributes	Unique Values
Age	40-49, 50-59, 60-69', 30-39, 70-79, 20-29
Menopause	Premeno, ge40, lt40
Tumor Size	15-19, 35-39, 30-34, 25-29, 40-44, 10-14, 
0-4, 20-24, 45-49, 50-54, 5-9
Inv-nodes	0-2, 3-5, 15-17, 6-8, 9-11, 24-26, 12-14
Node-caps	Yes, no
Deg-malig	2, 0, 1
Breast	Right, left
Breast-quad	left_up, central, left_low, right_up,  right_low
Irradiat	Yes, no
Class	recurrence-events, no-recurrence-events

Table 5 Unique Value Table

After understanding the unique values of the data set and its attributes. The exploratory data analysis is performed. 
It is important to analyze and explore the overview of the data set and how each attribute data points are distributed among each attribute. Therefore, a histogram is generated and visualized to understand overall impact of the attributes and the unique values to further dive into the data set.
 
Figure 4 Value Distribution of the data with Unique Values
Based on the above given data set most of the patients are from the age group of 50 to 59 and followed by 40 to 49. Also, women who have not experience menopause are the highest. There tumor size is the highest with a size of 30 -34 mm. Furthermore, the invasive nodes represent the tumor growth at 0-2 as the highest across all. In terms of the level of malignancy is visible at a level 1. Where grade 1 represents that the tumor is well differentiated and less aggressive. In terms of the location of the breast tumor it is mostly visible on the left low area and followed by left up and in general most occurrences take place on the patients left breast. Finally, in terms of radiation therapy a large proportionate have not undergone radiation therapy. 



 
Figure 5 Histogram Distribution of Age, Tumor Size, Inv-nodes, and Deg-malig
The above figure analyzes the numerical attributes against the class target. Higher number of values are visible for age group for 30-39 years and inv nodes being the highest for 0-2 for both classes. Finally in terms of malignancy level there are a large portion for level 2. 
It is important in identifying the target variable and understand how the distribution of recurring events and non-recurring events. To understand the relationship between the two a pie chart visualized. 
 
Figure 6 Distribution of Class Pie Chart
Based on the above figure, there is a large percentage of non-recurring individuals at 70% in the data set and 29% of recurring individuals. Therefore, exploratory data analysis is performed to understand how the predicted variables may affect across the two classes. Therefore, the original data set is divided into two data frames which includes all the class observations which are ‘no-recurrence-events’ and the other as ‘recurrence-events'.
 
Figure 7 Creating New Data Frames
2.4 Visualization of Findings.

After creating two data frames, each of the predicted variables against the are ‘no-recurrence-events’ and the other as ‘recurrence-events'.
1.	Age
 
Figure 8 Age Histogram Distribution
Age group is an important factor in the data set. In the non-recurring events, the highest is among the age 50 to 59 and the recurring events the highest is between age group 40 to 49. Based on the findings, early detection and treatment on no recurring events age groups is 20 to 29 is not visible in recuring events. Therefore, its better to detect and treat at the age group of 20 to 29.
2.	Menopause
 
Figure 9 Menopause Histogram Distribution
The menopause category represents ‘premeno’ which are individuals who have not yet experience menopause and ‘ge40’ represents individuals who are aged over 40 and have gone through menopause. Finally, ‘lt40’ represents people who are younger than 40 and had early menopause. 
Based on the histogram most of the ‘no recurring events’ and ‘recurring events’ fall under the individuals who are yet to experience menopause. As an insight under ‘no recurring events’ the ‘ge40’ category has a significant higher count. Therefore, postmenopausal women have a lower recurrence in comparison to premenopausal women. Finally, women who had early menopause has no recurrence events. Therefore, after treatment women generally do not have any recurrence of cancer. 
3.	Tumor Size
 
Figure 10 Tumor Size Histogram Distribution
Most of the cases in both groups fall under the tumor size of 20-24 and 25-29 and that is common among both classes. The count of recurrence events is higher than the no recurring event across most of the tumor size categories. 


4.	Inv-nodes
 
Figure 11  Inv-nodes Histogram Distribution
The inv-nodes both classes fall into the 0-2 category. Under the recurrence events suggest that there are a few cases with a higher number of involved lymph nodes. The higher the number of lymph nodes the higher the number of likelihoods of a recurrence. 
5.	Node-Caps
 
Figure 12 Node-Caps Histogram Distribution
Based on the node caps information there is a correlation with the recurrence of breast cancer. Where most of the cases are ‘no’ for both ‘no recurring' and ‘recurrence events’. However, node caps appearing as ‘yes’ is higher than the ‘no recurring events’ category. Therefore, a assumption is made that node caps might be associated with an increased likelihood of recurrence of breast cancer. 
6.	Deg-Malig
 
Figure 13 Deg-malig Histogram Distribution
The malignancy level plays an important role in determining the occurrence, where a higher value of 2 more likely to experience recurrence. Patients under the category ‘No Recurring Event’ malignancy level is at 1 with 101 observations. 
7.	Breast
 
Figure 14 Breast Histogram Distribution
Based on the observations against the ‘no recurring events’ and ‘recurring events', most of the tumors occur on the left side and on both classes. However, there many observations are non recuring events. 
8.	Breast-quad
 
Figure 15 Breast-quad Histogram Distribution
In terms of the distribution of the location of the tumor location it is evident that among both classes the ‘no recurring events’ and ‘recurrence events’ that ‘left low’ are where the tumor is found across both classes and the distribution is similar across all locations. 
9.	Irradiat
 
Figure 16Irradiat Histogram Distribution
This represents if the patient has received radiation therapy. Based on the observations, it is expected that non-recurring events the patients are not required to carry out radiation therapy. However, there is a small portion of women who require radiation therapy for recurrence events. 
After analyzing both classes ‘no recurrence events’ and ‘recurrence events’ and comparing each attribute there are several insights that has come into light. Specially the age attribute, where the highest number of non recuring events occurs in the age group of 50 to 59 and recurrence events highest in the age group of 40 to 49. Therefore, early detection and treatment of age group 20 to 29 will be beneficial in reducing the recurrence rate. Furthermore, menopause has a significant impact on both classes and most of the patients belong to the category who have not experienced menopause. Furthermore, postmenopausal women have a lower recurrence rate compared to premenopausal women and women who had early menopause depicts no recurrence events that after undergoing treatment women do not experience a recurrence of cancer. 
Based on the age and the menopause a heat map is visualized to determine what attribute factors affect. 
 
Figure 17 Heatmap Age vs Menopause
The above heatmap shows a high number of observations among age group of 40 – 49 with menopause individuals who are younger than 40 and had early menopause. Furthermore, the age group 50-59 and 60-69 are individuals who have not experienced breast cancer. 
To further deep dive how the age and menopause influence the recurrence event and the non-recurrence event. 
 
Figure 18 Heatmap against Two Classes
The two classes ‘recurrence events’ and ‘no recurrence events’ both has a high value depicted among the age group of 40 to 49. However, recurrence events are higher for women who had menopause above the age 40 and higher number on individuals who had menopause below the age of 40. 




2.5 Methods in addressing categorical data and libraries.

2.5.1 Importance of transforming categorical data

In the given breast cancer data set, there are several categorical data, and it is important to address and change the data accordingly. Since, the current data set can be used to carry out exploratory data analysis. However, to predict or apply to machine learning algorithms the values require to be numerical data. Therefore, these categorical data will be assigned to numeric values. 
Understanding categorical information through encoding, the data will be able to provide more meaningful predictions for machine learning. If the data is not encoded correctly the predictions may not be accurate. Therefore, it is important to encode the data for better accuracy of predictions.
By just applying numeric values will not help the output of the algorithm. For instance, if the data is transformed to numeric values at random, this will imply a false prediction. By encoding the data will ensure that it is transformed to numerical values in a proper manner. 
Finally assists to handle high cardinality with categorical variables. Large number of unique categories have high cardinality and may be difficult to analyze. Therefore, encoding helps to address these high cardinalities in the data. Therefore, these variables will be able to provide a better information for the machine learning model and making it simple.
Based on the reasons provided, it is evident that the breast cancer data set has a large portion of the data as categorical and requires to be addressed accordingly. 



2.5.2 Methods to address categorical data.

There are several methods in terms of transforming these categorical data.
1.	Label encoding
This is one of the most used methods used. Where, the categorical values are assigned with a numerical value to each category and under each feature. This is quite useful for ordinal data as the order of the categories are important. The below table represents how types of vehicles are encoded, using label encoding.
Vehicle	Vehicle Encoded
Car	0
Bus	1
Van	2
Bike	3
Table 6 Label Encoding Table
2.	One-hot Encoding
This method is used when the order of categories is not important as it is most suitable for nominal data. This is method creates binary columns for each category. 
Vehicle Car	Vehicle Bus	Vehicle Van	Vehicle Bike
0.0	0.0	1.0	0.0
0.0	1.0	0.0	0.0
0.0	0.0	0.0	1.0
1.0	0.0	0.0	0.0
Table 7 One Hot Encoding Table


3.	Ordinal Encoding
This method assigns numerical values based on the order of the data. This method is useful for ordinal data where the order of the data is important. 
Size	Size Encoded
Small	0.0
Medium	1.0
Large	2.0
Medium	1.0
Table 8 Ordinal Encoding Table
4.	Binary Encoding
This method is a simple method where each category represents as a binary code.
Color	Color 0	Color 1	Color 2
Red	0	0	1
Green	0	1	1
Blue	0	1	0
Table 9 Binary Encoding Table
5.	Count Encoding
This method uses the count of occurrences of that category in the dataset. This method is used when there is high cardinality in the categorical data. Based on the below example, the car observation appears twice. Therefore, it is labeled as 2 and the remaining values only appear once, and it is recognized as one under count encoding.


Vehicle	Vehicle
Car	2
Bus	1
Car	2
Bike	1
Van	1
Table 10 Count Encoding Table
To carry out the given encoding methods there are several libraries that can be used. 
1.	Sciket-learn 
This library provides all the above-mentioned encoding and handling specifically categorical data. 
2.	Category_encoder 
This library includes a larger encoding method and handles multiple data types. This a useful way for users to experiment and use best fit encoding methods based on the given data set.
3.	Pandas
Apart from being a prominent data manipulation library, pandas helps uses to analyse and transform the data based on the category of the data and encode within the data frame. 
4.	TenserFlow
There are some of the most sophisticated tools in relation to preprocessing categorical data using this library. The users have the flexibility to transform their categorical data as they wish. 
2.6 Addressing categorical data in the breast cancer data set. 

As per the exploratory data analysis the breast cancer data set had a large portion of the data as categorical data. However, each attribute should be evaluated on what method should be used. Therefore, each attribute and its respective unique values should be analyzed and use appropriate encoding methods.
1.	Age
Since the age of the categories are important and contains categorical values. Which are not ordinal. Therefore, one hot encoding is used with Pandas library.
 
Figure 19 Age One hot Encoding
2.	Menopause
Since the unique values are three and quite like the age column, one hot encoding can be applied to the menopause column. 
 

Figure 20 Menopasue One hot Encoding
3.	Tumor Size
Based on the tumor size variable the categories have an inherit order. Therefore, ordinal encoding is applied based on the order of the unique values. 
 
Figure 21 Tumor Size Ordinal Encoding
4.	Inv-nodes
This column represents the number of lymph nodes, and which was from lowest to the highest. Therefore, ordinal encoding is applied. 
 
Figure 22 Inv Nodes Ordinal Encoding
5.	Node-caps
This column contains ‘yes’ or ‘no’ values. Therefore, binary encoding is used.
 
Figure 23 Node Caps Binary Encoding
6.	Breast
This column has ‘left’ and ‘right’ which can be easily encoded using binary encoding.
 
Figure 24 Breast Binary Encoding
7.	Breast-quad
This column represents different locations of the tumor. There is no order or ranking and one hot encoding can be used to encode the values.
 
Figure 25 Breast quad One hot Encoding
8.	Irradiant
This has either ‘yes’ or ‘no’ values and the values can be assigned to binary values.
 
Figure 26 Irradiant Binary Encoding
Since the ‘deg-malig’ column was encoded at the initial stage of the preprocessing and the class column or the target column does not need to be encoded.
2.7 Grid search for Random Forest Machine Learning Algorithm.

Since the initial stage of the exploratory data analysis and analysis and based on the insights that were generated. There are large portion of categorical data in the given data set. Therefore, random forest is used and applied to this data set.

Firstly, the parameter grid needs to be defined. This specifies the different hyperparameters and its combinations that can be tested during the model training. This includes the number of estimators, maximum depth, and minimum samples leaf. Thereafter, the data is split into features ‘x’ and the target variable as ‘y’. Thereafter, the data is split into training and testing sets using sklearn. The test is assigned to 20% of the total data and 42 is used for reproducibility. Then, grid search cross validation is performed. Where the ‘random forest’ classifier is taken with different parameter combinations and evaluated using cross validation. Based on this, the best parameters are printed along with the best score based on the grid search that was performed. Then predictions are made on the test data. Finally, the classification report is generated.












The best parameters for the grid search are as follows:
•	Max depth: None
•	Minimum Sample leaf: 4
•	Number of estimators: 300
Based on the above given parameters, maximum depth is unlimited depth. In relation to the minimum number of samples required to be at a leaf node is 4 and number of decision trees are at 300. Based on the given parameters the best score reached during grid search is 73% accuracy level. The classification report indicates that the model achieved a precision of 0.80 for the class ‘no recurring events’ and 0.67 for the class ‘recurrence events. In terms of recall the given model achieved 095 for ‘no recurrence events’ and 0.29 for the other class. The F1 score represents the mean of precision and recall and provides a measure of the models’ performance with a score of 0.87. Finally, the overall accuracy of the model is 79%, which indicates that the model correctly classified 79% of the instances.
 
Figure 28 Decision Tree Parameters and Score
2.8 Auto ML Libraries.

Auto ML or automated machine learning simply means automating steps in the machine learning framework. The following areas are handled in auto ML.

1.	Data preprocessing.
2.	Feature engineering.
3.	Model Selection.
4.	Hyperparameter tunning.
5.	Model Selection.
This helps to simply and make machine learning model implementations much more efficient and effective. This is has helped in many domains, where users can apply some of the known auto ML libraries to their data sets and perform machine learning an easy manner. 
There are several available auto ML libraries that can be accessed. Some of these libraries have its unique features that makes it better in different areas. 
1.	H2O
This library is developed by h20.ai and helps support a wide range of machine learning algorithms along with automated feature engineering. This has an easy-to-use interface which supports on Python and R. Furthermore, this library supports missing values and categorical values automatically. 
2.	TPOT
This library stands for tree-based pipeline optimization tool which is built on top of sckit learn. This library uses machine learning pipeline to utilize genetic programming. This helps in classification, regression, and other forecasting tasks. 
3.	AutoGuon
AutoGluon is also an open-source library developed by amazon. Where easy to use capabilities for machine learning. What’s unique about this library is that it supports tabular and image data also has the capability of handing large amounts of data. The users can customize the fine tune the process. 
4.	Auto SkLearn
This was designed around the scikit learn framework. It automatically selects all preprocessing to meta learning. The platform supports classification to multilabel classification tasks and is widely used in many industries. Furthermore, the library has a positive performance and has won considerable amount of AutoML challenges.
When comparing these auto ML libraries there are few findings. Auto SK learn and H2O are mostly used auto ml libraries and has a good community backing. TPOT and AutoGluon provide advance scalability in comparison to the rest. H2O and AutoGluon offer for both tabuar and image data and can be used for many tasks. Finally, TPOT and AutoGluon provide more customization options and allowing user to control the auto ML process.
2.9 Applying Auto ML to Breast Cancer Data Set.

Based on the breast cancer data set, in order to perform auto ML the TPOT auto ML library is used. Firstly, the library is required specifically the TPOTClassifier to perform auto ML. Thereafter the code is fitted to the training data and precision, recall and F1 score is calculated using SkLearn. Thereafter, it creates a TPOT classifier object with the parameters and fits it to the training data. Then TPOT discovers the best pipeline configuration for the classification. Thereafter the best pipeline is extracted from the rest. Then the accuracy is at 0.75 and predictions are made on the remaining test set using the best pipeline. By automating the pipeline users can discover best models with any additional experiments.
 
Figure 29 TPOT Auto ML





References 
1.	Smith, J., Johnson, A. (2021). "A Comparative Analysis of Breast Cancer Detection and Diagnosis Using Data Visualization and Machine Learning Applications.": https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7349542/
2.	Brown, S., Wilson, L. (2020). "Analysis of Breast Cancer Detection Using Different Machine Learning Techniques.” : https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7351679/
3.	Thompson, R., Davis, M. (2019). "A Survey of Data Mining Techniques for Breast Cancer Diagnosis.": https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6371954/

