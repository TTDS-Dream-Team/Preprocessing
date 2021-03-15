import pandas as pd
from tqdm.auto import tqdm
from nnsplit import NNSplit
import numpy as np
from threading import Thread
import torch
import pickle
import h5py
from io import BytesIO
import pymongo
from pymongo import MongoClient
from bson.objectid import ObjectId
import zlib
from sentence_transformers import SentenceTransformer

splitter = NNSplit.load("en", use_cuda=True)
client = MongoClient(
    "mongodb+srv://cdminix:LTEG2pfoDiKfH29M@cluster0.pdjrf.mongodb.net/Reviews_Data?retryWrites=true&w=majority"
)
db = client.Reviews_Data
model = SentenceTransformer("paraphrase-distilroberta-base-v1")


class LSH:
    def __init__(
        self,
        hdf5_file="data.hdf5",
        input_dim=768,
        hash_dim=6,
        seed=42,
        chunksize=1_000,
        dtype="int8",
        file_write="w",
    ):
        self.planes = []
        self.input_dim = input_dim
        np.random.seed(seed)
        for i in range(hash_dim):
            v = np.random.rand(input_dim)
            v_hat = v / np.linalg.norm(v)
            self.planes.append(v_hat)

        self.planes = np.matrix(self.planes)
        self.data = h5py.File(hdf5_file, file_write)
        self.chunksize = chunksize
        self.buckets = {}
        self.id_buckets = {}
        self.dtype = dtype

    # Returns LSH of a vector
    def hash(self, vector):
        hash_vector = np.where((self.planes @ vector) < 0, 1, 0)[0]
        hash_string = "".join([str(num) for num in hash_vector])
        return hash_string

    def quantize(self, vector_list):
        vector_list = np.array(vector_list)
        if self.dtype in ["float16", "float32"]:
            return vector_list.astype(self.dtype)
        if self.dtype == "int8":
            return np.asarray(vector_list * 128, dtype=np.int8)
        raise ValueError(f"dtype needs to be float32, float16 or int8")

    # Add vector to bucket
    def add(self, vector, i):
        hashed = self.hash(vector)

        if hashed not in self.buckets:
            self.buckets[hashed] = []
            self.id_buckets[hashed] = []

        self.buckets[hashed].append(vector)
        self.id_buckets[hashed].append(i)

        if len(self.buckets[hashed]) >= self.chunksize:
            if hashed not in self.data:
                self.data.create_dataset(
                    hashed,
                    (self.chunksize, self.input_dim),
                    compression="gzip",
                    dtype=self.dtype,
                    chunks=True,
                    maxshape=(None, self.input_dim),
                )
                self.data.create_dataset(
                    hashed + "_id",
                    (self.chunksize,),
                    compression="gzip",
                    dtype="int32",
                    chunks=True,
                    maxshape=(None,),
                )
            else:
                hf = self.data[hashed]
                hf_id = self.data[hashed + "_id"]
                hf.resize((hf.shape[0] + self.chunksize), axis=0)
                hf_id.resize((hf_id.shape[0] + self.chunksize), axis=0)
            self.data[hashed][-self.chunksize :] = self.quantize(self.buckets[hashed])
            self.data[hashed + "_id"][-self.chunksize :] = self.id_buckets[hashed]
            self.buckets[hashed] = []
            self.id_buckets[hashed] = []

    def flush(self):
        for hashed, vectors in self.buckets.items():
            list_size = len(vectors)
            if hashed not in self.data:
                self.data.create_dataset(
                    hashed,
                    (list_size, self.input_dim),
                    compression="gzip",
                    dtype=self.dtype,
                    chunks=True,
                    maxshape=(None, self.input_dim),
                )
                self.data.create_dataset(
                    hashed + "_id",
                    (list_size,),
                    compression="gzip",
                    dtype="int32",
                    chunks=True,
                    maxshape=(None,),
                )
            else:
                hf = self.data[hashed]
                hf_id = self.data[hashed + "_id"]
                hf.resize((hf.shape[0] + list_size), axis=0)
                hf_id.resize((hf_id.shape[0] + list_size), axis=0)
            self.data[hashed][-list_size:] = self.quantize(self.buckets[hashed])
            self.data[hashed + "_id"][-list_size:] = self.id_buckets[hashed]
            self.buckets[hashed] = []
            self.id_buckets[hashed] = []

    # Returns bucket vector is in
    def get(self, vector):
        hashed = self.hash(vector)

        if hashed in self.data:
            return self.data[hashed]

        return []


batch_size = 1_000
i = 0
r_i = 0
try:
    with open("current_index.txt", "r") as f:
        i = int(f.read().split(",")[0])
        r_i = int(f.read().split(",")[1])
except:
    print("no index retrieved, starting new hdf5 file")
init_i = i

insert_thread = None

if i == 0:
    file_write = "w"
else:
    file_write = "a"

lsh_store = LSH(chunksize=batch_size, file_write=file_write)

max_entries = db["review_data"].count()
max_entries = 1_000_000

for review in tqdm(db["review_data"].find(), total=max_entries - i):
    with open("current_index.txt", "w") as f:
        f.write(f"{i},{r_i}")
    if i >= max_entries:
        lsh_store.flush()
        break
    if i % batch_size == 0:
        if i > init_i:
            review_l = []
            sentence_l = []
            start_index_l = []
            end_index_l = []
            for j, val in enumerate(splitter.split(texts)):
                for k, sentence in enumerate(val):
                    sentence = str(sentence)
                    strip_sentence = sentence.strip()
                    if len(strip_sentence) > 0:
                        review_l.append(ids[j])
                        sentence_l.append(strip_sentence)
                        if k >= 1:
                            start_index_l.append(end_index_l[-1] + 1)
                        else:
                            start_index_l.append(0)
                        end_index_l.append(start_index_l[-1] + len(sentence))
            embeddings = model.encode(sentence_l, convert_to_tensor=True)
            insert_list = []
            for k, (indv_review, sentence, start_index, end_index) in enumerate(
                zip(review_l, embeddings, start_index_l, end_index_l)
            ):
                lsh_store.add(sentence.numpy(), r_i)
                insert_list.append(
                    {
                        "_id": r_i,
                        "review_id": indv_review,
                        "s": start_index,
                        "e": end_index,
                    }
                )
                r_i += 1
            try:
                db["sentence_data"].insert_many(insert_list, ordered=False)
            except pymongo.errors.BulkWriteError:
                print("duplicate ids skipped")
        texts = []
        ids = []
    texts.append(zlib.decompress(review["review_text"]).decode())
    ids.append(review["_id"])
    i += 1
