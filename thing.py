from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import plotly.express as px
rows = []

with open("main.csv", "r") as f:
  csvreader = csv.reader(f)
  for row in csvreader: 
    rows.append(row)

headers = rows[0]
data_rows = rows[1:]
mass_list=[]
radius_list=[]
gravity_list=[]
for star in data_rows:
    mass=star[8]*1.989e+30
    radius=star[9]*6.957e+8
    gravity=6.67430e-11*(mass/(radius*radius))
    mass_list.append(mass)
    radius_list.append(radius)
    gravity_list.append(gravity)

fig=px.scatter(x=mass_list,y=radius_list)
fig.show()


X=[]
for index,planet_mass in enumerate(data_rows):
  temp_list=[mass_list[index],radius_list]
  X.append(temp_list)
wcss=[]
for i in range(1,11):
  kmeans=KMeans(n_clusters=i,init='k-means++',random_state=42)
  kmeans.fit(X)
  wcss.append(kmeans.inertia_)
plt.figure(figsize=(10,5))
sns.lineplot(range(1,11),wcss,markers="o",color="blue")
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()