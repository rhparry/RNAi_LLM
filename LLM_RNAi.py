#LLM_RNAi.py v0.1a Rhys H Parry University of Queensland Australia r.parry@uq.edu.au
#Usage python LLM_RNAi.py --ref_fa inputvirusgenome.fa --input_fastq inputfastq.fq --input_bam sorted_mapping.bam --output outputdir/
#Takes input fasta file and happed extracted fastq reads and the input BAM file with only mapped reads, tokenises them and embeds them and analyses them with BERT.
import argparse
import os
import time
import torch
import pysam
from Bio import SeqIO
from transformers import BertTokenizer, BertModel
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter

# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Analyze small RNA sequencing data with BERT and genomic context.')
    parser.add_argument('--input_bam', required=True, help='Path to the BAM file.')
    parser.add_argument('--input_fastq', required=True, help='Path to the FASTQ file.')
    parser.add_argument('--ref_fa', required=True, help='Path to the reference FASTA file.')
    parser.add_argument('--output', required=True, help='Path to the output directory.')
    return parser.parse_args()

# Ensure output directory exists
def ensure_output_dir(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

# Check GPU availability and decide whether to use GPU or CPU
def get_device(min_memory_required=4):
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        allocated_memory = torch.cuda.memory_allocated(0) / (1024**3)
        available_memory = total_memory - allocated_memory

        if available_memory >= min_memory_required:
            print(f"Using GPU with {available_memory:.2f} GB available memory.")
            return torch.device('cuda')
        else:
            print(f"Insufficient GPU memory. Available: {available_memory:.2f} GB, Required: {min_memory_required} GB. Falling back to CPU.")
            return torch.device('cpu')
    else:
        print("No GPU found. Using CPU.")
        return torch.device('cpu')

# Load FASTQ data
def load_fastq(fastq_file):
    print("Loading FASTQ data...")
    start_time = time.time()
    sequences = [str(record.seq) for record in SeqIO.parse(fastq_file, "fastq")]
    end_time = time.time()
    print(f"FASTQ data loaded in {end_time - start_time:.2f} seconds.")
    return sequences

# Tokenize sequences with consistent padding across all batches
def tokenize_sequences(sequences, batch_size=16):
    print("Tokenizing sequences...")
    start_time = time.time()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Step 1: Find the maximum sequence length in the dataset
    max_length = max([len(tokenizer.tokenize(seq)) for seq in sequences])
    print(f"Maximum sequence length: {max_length}")

    all_tokenized = []
    
    # Step 2: Tokenize and pad each batch to the same length (max_length)
    for i in range(0, len(sequences), batch_size):
        batch_sequences = sequences[i:i+batch_size]
        
        # Pad and truncate sequences to the maximum sequence length found
        tokenized_batch = tokenizer(batch_sequences, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')
        all_tokenized.append(tokenized_batch)
    
    # Debugging: check the sizes of tensors in each batch
    for i, batch in enumerate(all_tokenized):
        for key in batch:
            print(f"Batch {i}, key {key}, shape: {batch[key].shape}")
    
    # Step 3: Concatenate the batches along dimension 0 (batch dimension)
    end_time = time.time()
    print(f"Sequences tokenized in {end_time - start_time:.2f} seconds.")
    
    return {key: torch.cat([batch[key] for batch in all_tokenized], dim=0) for key in all_tokenized[0]}

# Get embeddings
def get_embeddings(tokenized_data, device):
    print("Generating embeddings...")
    start_time = time.time()
    model = BertModel.from_pretrained('bert-base-uncased').to(device)
    tokenized_data = {key: value.to(device) for key, value in tokenized_data.items()}
    with torch.no_grad():
        outputs = model(**tokenized_data)
        embeddings = outputs.last_hidden_state
    end_time = time.time()
    print(f"Embeddings generated in {end_time - start_time:.2f} seconds.")
    return embeddings

# Analyze embeddings
def analyze_embeddings(embeddings, output_dir):
    print("Analyzing embeddings...")
    start_time = time.time()
    embeddings_flattened = embeddings[:, 0, :].cpu().numpy()

    # PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(embeddings_flattened)
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(pca_result[:, 0], pca_result[:, 1])
    plt.title('PCA of RNA Sequence Embeddings')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.savefig(os.path.join(output_dir, 'pca_embeddings.pdf'))

    # Clustering
    kmeans = KMeans(n_clusters=5)
    clusters = kmeans.fit_predict(embeddings_flattened)
    plt.subplot(1, 2, 2)
    plt.scatter(pca_result[:, 0], pca_result[:, 1], c=clusters, cmap='viridis')
    plt.title('Clustering of RNA Sequence Embeddings')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.savefig(os.path.join(output_dir, 'clustering_embeddings.pdf'))
    plt.close()

    similarity_matrix = cosine_similarity(embeddings_flattened)
    end_time = time.time()
    print(f"Embedding analysis completed in {end_time - start_time:.2f} seconds.")
    return embeddings_flattened, similarity_matrix

def main():
    args = parse_args()
    
    ensure_output_dir(args.output)
    
    global device
    device = get_device()

    sequences = load_fastq(args.input_fastq)
    tokenized_data = tokenize_sequences(sequences)
    embeddings = get_embeddings(tokenized_data, device)
    embeddings_flattened, similarity_matrix = analyze_embeddings(embeddings, args.output)

    print("Analysis complete.")

if __name__ == "__main__":
    main()
