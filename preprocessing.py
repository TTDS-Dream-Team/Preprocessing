# imports
from pymongo import MongoClient
from nnsplit import NNSplit
from sentence_transformers import SentenceTransformer
import numpy as np
import h5py
from tqdm.auto import tqdm
import zlib
import pymongo
import json
from bson import ObjectId


class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, ObjectId):
            return str(o)
        return json.JSONEncoder.default(self, o)


splitter = NNSplit.load("en", use_cuda=True)

db_pwd = "LTEG2pfoDiKfH29M"
client = MongoClient(
    f"mongodb+srv://cdminix:{db_pwd}@cluster0.pdjrf.mongodb.net/Reviews_Data?retryWrites=true&w=majority"
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
        self.dtype = dtype

    # Returns LSH of a vector
    def hash(self, vector):
        hash_vector = np.where((self.planes @ vector) < 0, 1, 0)[0]
        hash_string = "".join([str(num) for num in hash_vector])
        return hash_string

    def quantize(self, item_list):
        vector_list = [i["vector"] for i in item_list]
        vector_list = np.array(vector_list)
        if self.dtype in ["float16", "float32"]:
            return vector_list.astype(self.dtype)
        if self.dtype == "int8":
            return np.asarray(vector_list * 128, dtype=np.int8)
        raise ValueError(f"dtype needs to be float32, float16 or int8")

    def dict_to_hdf5(self, hashed, flush=True):
        list_size = self.chunksize
        if flush:
            list_size = len(self.buckets[hashed])
        if len(self.buckets[hashed]) >= list_size and list_size > 0:
            items = self.buckets[hashed]
            if hashed not in self.data:
                self.data.create_dataset(
                    hashed,
                    (list_size, self.input_dim),
                    compression="gzip",
                    dtype=self.dtype,
                    chunks=True,
                    maxshape=(None, self.input_dim),
                )
            else:
                hf = self.data[hashed]
                hf.resize((hf.shape[0] + list_size), axis=0)
            self.data[hashed][-list_size:] = self.quantize(self.buckets[hashed])
            self.buckets[hashed] = []
            idx = np.arange(list_size) + len(self.data[hashed]) - 1
            for i, id in enumerate(idx):
                del items[i]["vector"]
                items[i]["_id"] = f"{hashed}_{id}"
            return items
        return []

    # Add vector to bucket
    def add(self, item):
        vector = item["vector"]
        hashed = self.hash(vector)

        if hashed not in self.buckets:
            self.buckets[hashed] = []

        self.buckets[hashed].append(item)

        return self.dict_to_hdf5(hashed)

    def flush(self):
        items = []
        for hashed in self.buckets.keys():
            items += self.dict_to_hdf5(hashed, flush=True)
        return items

    # Returns bucket vector is in
    def get(self, vector):
        hashed = self.hash(vector)
        if hashed in self.data:
            return self.data[hashed]
        return []


batch_size = 512
i = 0
init_i = i

if i == 0:
    file_write = "w"
else:
    file_write = "a"

try:
    lsh_store.data.close()
except:
    pass

lsh_store = LSH(chunksize=batch_size, file_write=file_write)

max_entries = db["review_data"].count()
percentage = 0.1
max_entries *= percentage
max_entries = int(max_entries)
print(max_entries)

json_file = open("db_objects.json", "w")

try:
    for review in tqdm(db["review_data"].find(), total=max_entries - i):
        if i % batch_size == 0 or i >= max_entries:
            if i > init_i:
                items = []
                sentence_list = []
                for j, val in enumerate(splitter.split(texts)):
                    whitespace = 0
                    for k, sentence in enumerate(val):
                        sentence = str(sentence)
                        strip_sentence = sentence.strip()
                        if any(c.isalpha() for c in sentence):
                            sentence_list.append(strip_sentence)
                            item = {}
                            item["review"] = ids[j]
                            if k == 0 or len(items) == 0:
                                item["start"] = 0
                            else:
                                item["start"] = items[-1]["end"] + whitespace
                                whitespace = 0
                            item["end"] = item["start"] + len(sentence)
                            # print(str(val)[item['start']:item['end']].strip())
                            # print('-----')
                            items.append(item)
                        else:
                            whitespace += len(sentence)
                embeddings = model.encode(sentence_list, convert_to_tensor=True)
                for k, item in enumerate(items):
                    item["vector"] = embeddings[k].numpy()
                    db_items = lsh_store.add(item)
                    if len(db_items) > 0:
                        for db_item in db_items:
                            json.dump(db_item, json_file, cls=JSONEncoder)
                            json_file.write("\n")
            if i >= max_entries:
                db_items = lsh_store.flush()
                for db_item in db_items:
                    json.dump(db_item, json_file, cls=JSONEncoder)
                    json_file.write("\n")
                break
            texts = []
            ids = []
        texts.append(zlib.decompress(review["review_text"]).decode())
        ids.append(review["_id"])
        i += 1
    lsh_store.data.close()
except:
    db_items = lsh_store.flush()
    for db_item in db_items:
        json.dump(db_item, json_file, cls=JSONEncoder)
        json_file.write("\n")
    lsh_store.data.close()
    with open("last_index.txt", "w") as f:
        f.write(i)
