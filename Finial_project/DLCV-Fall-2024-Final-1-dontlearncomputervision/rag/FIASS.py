import faiss
import numpy as np

# Set dimensions and number of data points
dimension = 128
database_size = 10000

# Generate random vector data (as the database)
database_vectors = np.random.random((database_size, dimension)).astype('float32')

# Initialize FAISS index
index = faiss.IndexFlatL2(dimension)      # L2 distance
# index = faiss.IndexFlatIP(dimension)      # inner product
# index = faiss.IndexBinaryFlat(dimension)  # Hamming distance
print("Is trained:", index.is_trained)

# Add vectors to the index
index.add(database_vectors)
print("Number of vectors in index:", index.ntotal)

# Query vector
query_vector = np.random.random((1, dimension)).astype('float32')

# Search for the 5 closest vectors
search_closest_num = 5
distances, indices = index.search(query_vector, search_closest_num)

# Print the distances to the nearest neighbors
print("Distances:", distances)
# Print the indices of the nearest neighbors
print("Indices:", indices)




