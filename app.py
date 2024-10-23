# Install required libraries (in Streamlit you would install them via requirements.txt or manually in the terminal)
# !pip install requests trafilatura sentence-transformers numpy torch tqdm scikit-learn pandas advertools streamlit

import streamlit as st
import requests
import trafilatura
from sentence_transformers import SentenceTransformer, util
import numpy as np
import pandas as pd
import advertools as adv
from sklearn.cluster import KMeans
from collections import Counter

# Title of the app
st.title("Site Focus Calculator")
st.write("A tool for calculating the site focus score of a website or a series of URLs.")


# Load the model
#model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")
model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)



# Input fields for sitemap or list of URLs (separated by newlines)
sitemap_url = st.text_input("Enter your XML sitemap URL (optional)", "")
url_list_input = st.text_area("Enter a list of URLs (separated by newlines, optional)", "")

# Add a "Run" button to trigger the URL processing
if st.button("Run Analysis"):
    # Process either sitemap or URL list
    urls = []
    if sitemap_url:
        st.write("Fetching URLs from the sitemap...")
        # Read sitemap and extract URLs using advertools
        sitemap_df = adv.sitemap_to_df(sitemap_url)
        urls = sitemap_df['loc'].tolist()
        urls = urls[:50]  # Limit to first 50 URLs for testing purposes
        st.write(f"Processing {len(urls)} URLs from sitemap.")
    elif url_list_input:
        # Parse URL list from input (newlines separated)
        urls = [url.strip() for url in url_list_input.split('\n') if url.strip()]
        st.write(f"Processing {len(urls)} URLs from the input list.")
    else:
        st.warning("Please provide either a sitemap URL or a list of URLs.")

    # Function to get embeddings
    def get_embedding(text):
        """Generate embedding for the given text using the mxbai-embed-large-v1 model."""
        prompt = "Represent this sentence for searching relevant passages: " + text
        embedding = model.encode(prompt)
        return embedding

    # Initialize lists to store embeddings and corresponding URLs
    embeddings = []
    valid_urls = []
    extracted_texts = []
    error_urls = []

    # Define headers with User-Agent
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                      'AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/58.0.3029.110 Safari/537.3'
    }

    # Only process if URLs are provided
    if urls:
        st.write("Processing URLs...")
        for url in urls:
            try:
                response = requests.get(url, headers=headers, timeout=10)
                if response.status_code == 200:
                    html_content = response.text
                    extracted_text = trafilatura.extract(html_content)
                    if extracted_text:
                        embedding = get_embedding(extracted_text)
                        embeddings.append(embedding)
                        valid_urls.append(url)
                        extracted_texts.append(extracted_text)
                    else:
                        error_urls.append((url, "No content extracted"))
                else:
                    error_urls.append((url, f"Status code {response.status_code}"))
            except Exception as e:
                error_urls.append((url, f"Error: {str(e)}"))

    # Check if we have any valid embeddings
    if embeddings:
        # Stack embeddings into a single array
        embeddings_array = np.vstack(embeddings)

        # Compute the site embedding by averaging all embeddings
        site_embedding = np.mean(embeddings_array, axis=0)

        # Compute cosine similarity between each content embedding and the site embedding
        similarities = util.cos_sim(embeddings_array, site_embedding)
        similarities = similarities.numpy().flatten()

        # Calculate pairwise cosine similarities for site focus score
        pairwise_similarities = []
        for i in range(len(embeddings_array)):
            for j in range(i+1, len(embeddings_array)):
                sim = util.cos_sim(embeddings_array[i], embeddings_array[j]).item()
                pairwise_similarities.append(sim)

        # Calculate average pairwise similarity
        if pairwise_similarities:
            site_focus_score = sum(pairwise_similarities) / len(pairwise_similarities)
        else:
            site_focus_score = 0.0

        st.write(f"Site Focus Score: {site_focus_score:.4f}")

        # Perform KMeans clustering if there are enough samples
        if len(embeddings_array) >= 2:
            try:
                n_clusters = 2  # Adjust the number of clusters as needed
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                kmeans.fit(embeddings_array)
                labels = kmeans.labels_

                # Analyze cluster sizes
                cluster_counts = Counter(labels)

                # Assign a cluster-based score to each page based on cluster size
                cluster_sizes = dict(cluster_counts)
                page_cluster_scores = []
                for label in labels:
                    score = cluster_sizes[label] / len(embeddings_array)  # Fraction of pages in the cluster
                    page_cluster_scores.append(score)

                # Create a DataFrame with the desired columns
                df = pd.DataFrame({
                    'URL': valid_urls,
                    'PageSiteSimilarity': similarities,
                    'ClusterLabel': labels,
                    'ClusterScore': page_cluster_scores
                })

                # Display the DataFrame
                st.write("URL Analysis Results")
                st.dataframe(df)

                # Option to download the results as CSV
                csv = df.to_csv(index=False)
                st.download_button(label="Download data as CSV", data=csv, file_name='url_analysis_results.csv', mime='text/csv')
            except ValueError as ve:
                st.error(f"KMeans error: {ve}. Try using a smaller number of clusters.")
        else:
            st.warning("Not enough URLs to perform clustering. Need at least 2 samples.")
    else:
        st.warning("No valid embeddings were generated.")

    # If there are any error URLs, show them
    if error_urls:
        st.write("The following URLs encountered errors and were not processed:")
        error_df = pd.DataFrame(error_urls, columns=["URL", "Error"])
        st.dataframe(error_df)
else:
    st.info("Click 'Run Analysis' to start the process.")
