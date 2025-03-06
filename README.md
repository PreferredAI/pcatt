# A partition cover approach to tokenization
In this work, we formulate tokenization as an optimization objective, show that it is NP-hard via a simple reduction from vertex cover, and propose a polynomial-time greedy algorithm **GreedTok**.
Our formulation naturally relaxes to the well-studied weighted maximum coverage problem which has a simple $(1 - 1/e)$-approximation greedy algorithm.

### Compute Performance Update
Compared to the base implementation, beta6 has a considerable speedup! The key difference is the implementation of greedy updates on top of greedy choice. 

<center>

|Dataset| $k$ | #uniq words | #candidates | Previous (arXiv paper's) | Current (beta6) | improvement |
| ------- | --- | -------: | -------: | -------: | -------: | :----: |
| UN    |5K|105,505|884,630| ~140 seconds |  ~6 seconds  | x23 |
| arXiv |5K|881,233|7,625,530| ~28 minutes | ~63 seconds | x26 |
| Wiki  |10K|8,769,943|93,243,449| ~12.5 hours | ~11 minutes | x68 |
| PubMed|10K|6,527,614|97,870,366| ~24.5 hours | ~11 minutes | x133|

</center>

Table results shows time to solve (obtain a $k$-sized token set) from word counts. Since most of the compute is front-heavy, solving for larger $k$ size is trivial.
For detailed logs, compare cpp_logs/{$data}/{$data}.log versus cpp_logs/{$data}/{$data}_fast.log. 


### Beta: Huggingface AutoTokenizer interface

Install the beta version (for transformers >= 4):
```
wget "https://github.com/PreferredAI/pcatt/archive/refs/tags/v0.14-beta8.zip"
unzip v0.14-beta8.zip -d pcatt
cd pcatt
pip install -r requirements.txt
pip install transformers
pip install .
```

For "training" either:
```
from pcatt.hf.greedtok import GreedTok
greedtok = GreedTok().train_new_from_iterator(word_iterator, 100, max_token_length=5, min_word_count=1)
```
or
```
from pcatt.hf.greedtok import GreedTok
greedtok = GreedTok().train_new_from_counts(word_count_dict, 100, max_token_length=5, min_word_count=1)
```
To use either:
```
from pcatt.hf.greedtok import GreedTok
greedtok = GreedTok.from_pretrained(greedtok_file_directory)
```
or
```
import pcatt.hf
greedtok = AutoTokenizer.from_pretrained("greedtok_file_directory")
```
Refer to [eval_hf.ipynb](https://github.com/PreferredAI/aoatt/blob/main/eval_hf.ipynb) for examples and tips. Note that the code in pcatt.hf is Apache 2.0 (following Huggingface Tokenizers).

### GreedTok 
1. If using python wrapper
   
    a. Using pip (use the lightweight source code w/o data/notebooks):
      ```
      wget "https://github.com/PreferredAI/pcatt/archive/refs/tags/v0.13.tar.gz"
      unzip pcatt-0.13.zip -d pcatt
      cd pcatt
      pip install -r requirements.txt
      pip install .
      ```
    b. Or compile manually e.g. (have to specify links)
      ```
      c++ -O3 -Wall -shared -std=c++20 \
      -fPIC $(python3 -m pybind11 --includes) \
      -I$CONDA_PREFIX/include/ \
      -I$CONDA_PREFIX/include/tbb \
      -I$CONDA_PREFIX/include/oneapi \
      -L$CONDA_PREFIX/lib/ \
      -l tbb \
      ./pcatt/greedy_builder.cpp \
      -o ./pcatt/greedy_builder$(python3-config --extension-suffix) 
      ```
    c. import and use! Examples in [eval_tokenizer_example.ipynb](https://github.com/PreferredAI/aoatt/blob/main/eval_tokenizer_example.ipynb)
2. If using C++ files directly

    a. Install dependencies for C++ code, we use oneTBB to parallelize the code, simplest way is to use Conda or pip:
      ```
      conda install tbb-devel
      ```

    b. Compile greedy_cache.py e.g.:
      ```
      c++ -O3 -std=c++20 \
      -I$CONDA_PREFIX/include/ \
      -I$CONDA_PREFIX/include/tbb \
      -I$CONDA_PREFIX/include/oneapi \
      -L$CONDA_PREFIX/lib/ \
      -l tbb \
      pcatt/greedy_cache.cpp \
      -o pcatt/greedy.exe 
      ```
    c. Prepare inputs (refer to [cpp_inputs](https://github.com/PreferredAI/aoatt/blob/main/cpp_inputs) for examples):
      * counts: a file with '\n' delimited integers
      * words: a file with ' ' (space) delimited words
        
    d. Run compiled program (currently looks for domain inputs in fixed path under cpp_inputs/*)
        ```
         ./greedy.exe <domain> <k>
        ```
    e. Now we obtained our ranked token sequence (refer to [cpp_outputs](https://github.com/PreferredAI/aoatt/blob/main/cpp_outputs/) for examples):
      * merges: the number of covers at each step, delimited by '\n'
      * tokens: byte sequences in hex-format, delimited by '\n'

Evaluations in [eval_notebook.ipynb](https://github.com/PreferredAI/aoatt/blob/main/eval_notebook.ipynb)

### Citation
```
@article{lim2025partition,
      title={A partition cover approach to tokenization}, 
      author={Lim, Jia Peng and Choo, Davin and Lauw, Hady W.},
      year={2025},
      journal={arXiv preprint arXiv:2501.06246},
      url={https://arxiv.org/abs/2501.06246}, 
}
```
