import click
from pymongo import MongoClient
from nnsplit import NNSplit
from sentence_transformers import SentenceTransformer
import numpy as np
import h5py
from tqdm.auto import tqdm
import zlib
import pymongo
from mongo_proxy import MongoProxy
import json
from bson import ObjectId
import time
from threading import Thread, Lock
import gc
from guppy import hpy

splitter = NNSplit.load("en", use_cuda=True)

lock = Lock()

class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, ObjectId):
            return str(o)
        return json.JSONEncoder.default(self, o)


db_pwd = "LTEG2pfoDiKfH29M"
client = MongoProxy(
    MongoClient(
        f"mongodb+srv://cdminix:{db_pwd}@cluster0.pdjrf.mongodb.net/Reviews_Data?retryWrites=true&w=majority"
    )
)
db = client.Reviews_Data
model = None

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
        add_neg=True,
        random_factor=3,
        estimated_max_size=None,
        max_size_factor=9*4*3,
        log_step=50_000,
        list_ratio=1, # the proportion of each chunk to write after (should not be >1)
    ):
        self.planes = []
        self.input_dim = input_dim
        np.random.seed(seed)
        factor = random_factor
        for i in range(hash_dim * factor):
            if add_neg:
                v = (np.random.rand(input_dim) * 2) - 1
            else:
                v = np.random.rand(input_dim)
            v_hat = v / np.linalg.norm(v)
            self.planes.append(v_hat)
        dists = np.zeros((hash_dim * factor, hash_dim * factor))
        for i in range(hash_dim * factor):
            for j in range(hash_dim * factor):
                if i == j:
                    dists[i, j] = np.inf
                else:
                    dists[i, j] = np.linalg.norm(self.planes[i] - self.planes[j])
        remove_idx = []
        while len(remove_idx) < hash_dim * factor - hash_dim:
            ind = np.unravel_index(np.argmin(dists, axis=None), dists.shape)
            if ind[0] not in remove_idx:
                remove_idx.append(ind[0])
            dists[ind] = np.inf
            dists[ind[1], ind[0]] = np.inf
        print(remove_idx, len(remove_idx), len(set(remove_idx)), hash_dim)
        new_planes = []
        for i, p in enumerate(self.planes):
            if i not in remove_idx:
                new_planes.append(self.planes[i])
        print(len(new_planes), len(self.planes))

        self.planes = np.matrix(new_planes)
        self.data = h5py.File(hdf5_file, file_write)
        self.file_path = hdf5_file
        self.chunksize = chunksize
        self.buckets = {}
        self.bucket_ind = {}
        self.dtype = dtype
        self.max_size = (estimated_max_size/(2**hash_dim))*max_size_factor
        self.step = 0
        self.log_step = (log_step // chunksize) + 1
        self.list_ratio = list_ratio

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
        list_size = int(self.chunksize*self.list_ratio)+1
        if flush:
            list_size = len(self.buckets[hashed])
        if len(self.buckets[hashed]) >= list_size and list_size > 0:
            items = self.buckets[hashed]
            if hashed not in self.data:
                self.data.create_dataset(
                    hashed,
                    (self.max_size, self.input_dim),
                    compression="lzf",
                    dtype=self.dtype,
                    chunks=(self.chunksize,self.input_dim),
                    maxshape=(None, self.input_dim),
                )
            last_i = self.bucket_ind[hashed]
            self.data[hashed][last_i:last_i+list_size] = self.quantize(self.buckets[hashed])
            del self.buckets[hashed]
            self.buckets[hashed] = []
            gc.collect()
            self.bucket_ind[hashed] = last_i+list_size
            idx = np.arange(list_size) + last_i
            for i, id in enumerate(idx):
                del items[i]["vector"]
                items[i]["_id"] = f"{hashed}_{id}"
            self.step += 1
            if self.step % self.log_step == 0 and not flush:
                fullest = max(self.bucket_ind.values())/self.max_size*100
                least_full = min(self.bucket_ind.values())/self.max_size*100
                print(f'FULLEST BUCKET: {fullest:.2f}%')
                print(f'LEAST FULL BUCKET: {least_full:.2f}%')
            return items
        return []

    # Add vector to bucket
    def add(self, item):
        vector = item["vector"]
        hashed = self.hash(vector)

        if hashed not in self.buckets:
            self.buckets[hashed] = []
            self.bucket_ind[hashed] = 0

        self.buckets[hashed].append(item)

        return self.dict_to_hdf5(hashed, flush=False)

    def flush(self, resize=True):
        items = []
        for hashed in self.buckets.keys():
            items += self.dict_to_hdf5(hashed, flush=True)
            
        for hashed, ind in self.bucket_ind.items():
            if ind > 0 and resize:
                print(f'resizing {hashed} to size {ind}')
                self.data[hashed].resize((ind, self.input_dim))
            
        return items

    # Returns bucket vector is in
    def get(self, vector):
        hashed = self.hash(vector)
        if hashed in self.data:
            return self.data[hashed]
        return []
    
    def reopen(self):
        self.data.close()
        self.data = h5py.File(self.file_path, 'a')
 
def write_json(db_items, json_file):
    with lock:
        for db_item in db_items:
            json.dump(db_item, json_file, cls=JSONEncoder)
            json_file.write("\n")
        
            
def add_h5py(items, embeddings, lsh_store, json_file):
    db_items = []
    for k, item in enumerate(items):
        item["vector"] = embeddings[k].numpy()
        db_items += lsh_store.add(item)
    if len(db_items) > 0:
        write_json(db_items, json_file)

@click.command()
@click.option("--batch-size", default=512)
@click.option("--chunk-size", default=1_000)
@click.option("--percentage", default=0.25)
@click.option("--hash-dim", default=6)
@click.option("--postfix", prompt="data file postfix")
@click.option("--only-db", default=False)
@click.option("--no-db", default=False)
@click.option("--start", default=0)
def main(batch_size, chunk_size, percentage, hash_dim, postfix, only_db, no_db, start):
    start = int(start)
    
    if not only_db:
        model = SentenceTransformer("paraphrase-distilroberta-base-v1")
        max_entries = db["reviews"].count()
        if percentage <= 1:
            max_entries = min(int(max_entries),int(max_entries*percentage)+start)
        else:
            max_entries = int(percentage + start)
        print(f"loading {max_entries} entries ({percentage*100:.2f})% of the data)")
        file_write = "w"

#        if start > 0:
#            file_write = "a"
#        else:
#            file_write = "w"
            
        lsh_store = LSH(
            hdf5_file=f"data_{postfix}.h5py",
            chunksize=chunk_size,
            file_write=file_write,
            hash_dim=hash_dim,
            estimated_max_size=max_entries
        )
        json_file = open(f"db_objects_{postfix}.json", file_write)

        _thread = None

        reviews = db["reviews"].find().sort('_id')
        review = None
        for i in tqdm(range(start, max_entries+1)): # miniters=int(max_entries/200)
            j = 0
            while j == 0 or (review is None and j < 9):
                if j > 0:
                    print(f'retrying fetching review')
                try:
                    review = next(reviews, None)
                except Exception as e:
                    print(e)
                    time.sleep(0.5)
                    reviews = db["reviews"].find().sort('_id').skip(i)
                j += 1
            if j >= 9:
                raise
            if i >= max_entries or (i > start and i % (max_entries//20) == 0):
                resize = True
                if i % (max_entries//20) == 0:
                    print('flushing (every 5%)')
                    resize = False
                if _thread is not None:
                    _thread.join() # avoid "thread buildup"
                db_items = lsh_store.flush(resize)
                lsh_store.reopen() # idk why I need this but please let this shit work
                _thread = Thread(target=write_json, args=(db_items,json_file,))
                _thread.start()
                del db_items
                if i >= max_entries:
                    break
            if i % batch_size == 0 or i >= max_entries or review is None:
                if i > start:
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
                                items.append(item)
                            else:
                                whitespace += len(sentence)
                    try:
                        embeddings = model.encode(sentence_list, convert_to_tensor=True)
                        if _thread is not None:
                            _thread.join() # avoid "thread buildup"
                        _thread = Thread(target=add_h5py, args=(items, embeddings, lsh_store, json_file, ))
                        _thread.start()
                    except Exception as e:
                        print(e)
                        print(sentence_list)
                texts = []
                ids = []
            if review is not None:
                texts.append(zlib.decompress(review["review_text"]).decode())
                ids.append(review["_id"])
                i += 1
            else:
                print('skipped review')
        _thread.join()
        lsh_store.data.close()

        json_file.close()
        
    if not no_db:
        db[f"sentence_data_{postfix}"].drop()
        chunk_size = 10_000
        with open(f"db_objects_{postfix}.json", "r") as objects:
            item = 0
            chunk = []
            line = objects.readline()
            while line is not None:
                try:
                    item = json.loads(line)
                except:
                    print(line)
                    break
                item["review"] = ObjectId(item["review"])
                chunk.append(item)
                if len(chunk) >= chunk_size:
                    db[f"sentence_data_{postfix}"].insert_many(chunk, ordered=False)
                    chunk = []
                line = objects.readline()
        db[f"sentence_data_{postfix}"].insert_many(chunk, ordered=False)


if __name__ == "__main__":
    main()
