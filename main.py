import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/Mall_Customers.csv")

print(df.head())

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nSummary Statistics:")
print(df.describe())

print("\nUnique Genders:")
print(df["Gender"].value_counts())


plt.scatter(
    df["Annual Income (k$)"],
    df["Spending Score (1-100)"]
)

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score")
plt.title("Customer Distribution")

plt.show()

print("\nCorrelation Matrix:")

print(
    df[
        [
            "Age",
            "Annual Income (k$)",
            "Spending Score (1-100)"
        ]
    ].corr()
)

plt.figure(figsize=(8, 6))

plt.scatter(
    df["Annual Income (k$)"],
    df["Spending Score (1-100)"]
)

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Segmentation Data")

plt.grid(True)

# plt.show()


wcss = []

for k in range(1, 11):

    kmeans = KMeans(
        n_clusters=k,
        random_state=42
    )

    kmeans.fit(
        df[
            [
                "Annual Income (k$)",
                "Spending Score (1-100)"
            ]
        ]
    )

    wcss.append(kmeans.inertia_)

print(wcss)

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, 11),
    wcss,
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")
plt.title("Elbow Method")

plt.grid(True)

# plt.show()

# ==========================
# K-MEANS CLUSTERING
# ==========================

kmeans = KMeans(
    n_clusters=5,
    random_state=42
)

clusters = kmeans.fit_predict(
    df[
        [
            "Annual Income (k$)",
            "Spending Score (1-100)"
        ]
    ]
)

print(clusters)
df["Cluster"] = clusters

print(df.head(20))
print(df["Cluster"].value_counts())
print(kmeans.cluster_centers_)

df.to_csv(
    "data/segmented_customers.csv",
    index=False
)

print("Segmented dataset saved!")


plt.figure(figsize=(10, 7))

plt.scatter(
    df[df["Cluster"] == 0]["Annual Income (k$)"],
    df[df["Cluster"] == 0]["Spending Score (1-100)"],
    label="Standard Customers"
)

plt.scatter(
    df[df["Cluster"] == 1]["Annual Income (k$)"],
    df[df["Cluster"] == 1]["Spending Score (1-100)"],
    label="VIP Customers"
)

plt.scatter(
    df[df["Cluster"] == 2]["Annual Income (k$)"],
    df[df["Cluster"] == 2]["Spending Score (1-100)"],
    label="Impulsive Spenders"
)

plt.scatter(
    df[df["Cluster"] == 3]["Annual Income (k$)"],
    df[df["Cluster"] == 3]["Spending Score (1-100)"],
    label="Careful Customers"
)

plt.scatter(
    df[df["Cluster"] == 4]["Annual Income (k$)"],
    df[df["Cluster"] == 4]["Spending Score (1-100)"],
    label="Budget Customers"
)

plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=300,
    marker="X",
    label="Centroids"
)

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Segments using K-Means")
plt.legend()
plt.grid(True)

# plt.show()

score = silhouette_score(
    df[
        [
            "Annual Income (k$)",
            "Spending Score (1-100)"
        ]
    ],
    clusters
)

print("\nSilhouette Score:")
print(score)

features = df[
    [
        "Age",
        "Annual Income (k$)",
        "Spending Score (1-100)"
    ]
]

scaler = StandardScaler()

scaled_features = scaler.fit_transform(
    features
)

print(scaled_features[:5])

# ==========================
# KMEANS ON SCALED DATA
# ==========================

kmeans_scaled = KMeans(
    n_clusters=5,
    random_state=42
)

clusters_scaled = kmeans_scaled.fit_predict(
    scaled_features
)

print("\nScaled Cluster Counts:\n")

print(
    pd.Series(clusters_scaled)
    .value_counts()
)

scaled_score = silhouette_score(
    scaled_features,
    clusters_scaled
)

print(
    "\nScaled Silhouette Score:",
    round(scaled_score, 4)
)

features_age = df[
    [
        "Age",
        "Annual Income (k$)",
        "Spending Score (1-100)"
    ]
]

scaler = StandardScaler()

scaled_features_age = scaler.fit_transform(
    features_age
)

kmeans_age = KMeans(
    n_clusters=5,
    random_state=42
)

clusters_age = kmeans_age.fit_predict(
    scaled_features_age
)

score_age = silhouette_score(
    scaled_features_age,
    clusters_age
)

print(score_age)

print(
    df.groupby("Cluster")[
        [
            "Age",
            "Annual Income (k$)",
            "Spending Score (1-100)"
        ]
    ].mean()
)

cluster_report = df.groupby("Cluster").agg({
    "Age": "mean",
    "Annual Income (k$)": "mean",
    "Spending Score (1-100)": "mean",
    "CustomerID": "count"
})

print(cluster_report)

segment_names = {
    0: "Standard",
    1: "VIP",
    2: "Impulsive",
    3: "Careful",
    4: "Budget"
}

cluster_report["Segment"] = (
    cluster_report.index.map(segment_names)
)

print(cluster_report)

cluster_report = cluster_report.round(1)

print(cluster_report)
