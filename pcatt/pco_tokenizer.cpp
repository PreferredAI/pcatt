#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/stl.h>
namespace py = pybind11;
#include <iostream>
#include <chrono>
#include <vector>
#include <set>
#include <string>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <regex>
#include <limits.h>
#include <execution>
#include "tbb.h"
using namespace std;
namespace chrono = std::chrono;

/*
c++ -O3 -Wall -shared -std=c++20 \
-fPIC $(python3 -m pybind11 --includes) \
-I$CONDA_PREFIX/include/ \
-I$CONDA_PREFIX/include/tbb \
-L$CONDA_PREFIX/lib/ \
-l tbb \
./pcatt/pco_tokenizer.cpp \
-o ./pcatt/pco_tokenizer$(python3-config --extension-suffix)
*/

struct SubstringPos
{
    long unsigned arr_start;
    long unsigned arr_end;
    unsigned int substr_start;
    unsigned int substr_end;
    bool skip = false;
    /**
     * @brief Construct a new Substring Pos object, meant for internal use
     *
     * @param a index of start of word in array
     * @param b index of end of word in array
     * @param c start of substring in word
     * @param d end of substring in word
     */
    SubstringPos(long unsigned a, long unsigned b, unsigned int c, unsigned int d)
    {
        arr_start = a;
        arr_end = b;
        substr_start = c;
        substr_end = d;
    }
};

class GreedyPCOTokenizer
{

public:
    static bool token_score_sorter(
        const pair<string, long unsigned> a,
        const pair<string, long unsigned> b)
    {
        return a.second < b.second;
    }

    long unsigned singleton_count = 0;
    vector<string> ranks;
    vector<int unsigned> T_arr;
    vector<int unsigned> D_arr;
    vector<long unsigned> scores;
    unordered_set<string> shortlist;
    unordered_set<string> candidate_tokens{};
    unordered_map<string, long unsigned> results;
    unordered_map<string, long unsigned> word_counts{};
    unordered_map<long unsigned, string> index_to_word;
    unordered_map<long unsigned, long unsigned> id_to_count;
    unordered_map<string, unordered_set<string>> word_to_substring;
    unordered_map<string, vector<SubstringPos>> substring_to_index;
    unordered_map<string, pair<long unsigned, long unsigned>> word_to_index;

    /**
     * @brief Construct a new Greedy P C O Tokenizer object
     *
     * @param word_counts word to count mapping
     * @param candidate_tokens to investigate
     */
    GreedyPCOTokenizer(
        unordered_map<string, long unsigned> _word_counts = {},
        unordered_set<string> _candidate_tokens = {})
    {
        word_counts = _word_counts;
        candidate_tokens = _candidate_tokens;
    }

    virtual ~GreedyPCOTokenizer() {}

    void build_counter_from_text(const vector<vector<string>> &texts)
    {
        tbb::concurrent_hash_map<string, unsigned long> async_counter;
        tbb::parallel_for(
            tbb::blocked_range<long unsigned>(0, texts.size()),
            [&](tbb::blocked_range<long unsigned> r)
            {
                unordered_map<string, unsigned long> temp_counter;
                for (long unsigned i = r.begin(); i < r.end(); ++i)
                {
                    for (const string &w : texts.at(i))
                    {
                        auto p = temp_counter.try_emplace(w, 0);
                        p.first->second += 1;
                    }
                }
                tbb::concurrent_hash_map<string, unsigned long>::accessor a;
                for (const auto &item : temp_counter)
                {
                    async_counter.insert(a, item.first);
                    a->second += item.second;
                    a.release();
                }
            });

        for (const auto &item : async_counter)
        {
            if (word_counts.find(item.first) == word_counts.end())
            {
                word_counts[item.first] = 0;
            }
            word_counts[item.first] += item.second;
        }
    }

    /**
     * @brief Create a bipartite graph representation and allocate spaces for tracking arrays
     */
    void initialize_graph(
        const size_t max_token_length = UINT8_MAX,
        const unsigned int min_word_count = 1)
    {
        cout << "Word counts size: " << word_counts.size() << endl;
        cout << "Token set size: " << candidate_tokens.size() << endl;
        if (candidate_tokens.size() == 0)
        {
            cout << "Empty token set size selected -> all possible substrings with..." << endl;
        }
        cout << "Max token size: " << max_token_length << endl;
        cout << "Min. word count: " << min_word_count << endl;
        /* Initialize variables */
        auto start = chrono::high_resolution_clock::now();
        long unsigned next_id = 0;
        long unsigned end_id = 0;
        vector<pair<string, long unsigned>> wc{};
        wc.reserve(word_counts.size());
        /* Initialize array positions */
        for (const auto &item : word_counts)
        {
            if (item.second < min_word_count)
            {
                continue;
            }

            singleton_count += item.first.size();
            end_id = next_id + item.first.size();
            id_to_count[next_id] = item.second;

            word_to_index[item.first] = pair(next_id, end_id);
            wc.emplace_back(pair(item.first, item.second));
            next_id = end_id;
        }

        tbb::concurrent_hash_map<string, vector<SubstringPos>> substring_to_index_collector;
        tbb::concurrent_hash_map<string, unordered_set<string>> word_to_substring_collector;
        tbb::parallel_for(
            tbb::blocked_range<long unsigned>(0, wc.size()),
            [&](tbb::blocked_range<long unsigned> r)
            {
                tbb::concurrent_hash_map<string, vector<SubstringPos>>::accessor a;
                tbb::concurrent_hash_map<string, unordered_set<string>>::accessor b;
                for (long unsigned i = r.begin(); i < r.end(); ++i)
                {
                    const string word = wc.at(i).first;
                    const auto idx = word_to_index[word];
                    const unsigned long start = idx.first;
                    const unsigned long end = idx.second;
                    const unsigned long size = word.size();
                    unordered_map<string, vector<SubstringPos>> substring_to_index_helper;
                    substring_to_index_helper.reserve(size * size);

                    for (unsigned int i = 0; i < size; ++i)
                    {
                        for (unsigned int j = i + 2; j < min(max_token_length + i, size + 1); ++j)
                        {
                            string substr = word.substr(i, j - i);
                            if (substr.size() <= 1)
                            {
                                continue;
                            }
                            if (candidate_tokens.size() > 0 && candidate_tokens.find(substr) == candidate_tokens.end())
                            {
                                continue;
                            }
                            if (substring_to_index_helper.find(substr) == substring_to_index_helper.end())
                            {
                                substring_to_index_helper[substr] = vector<SubstringPos>();
                            }
                            substring_to_index_helper[substr].push_back({start, end, i, j});
                        }
                    }
                    unordered_set<string> substr_set;
                    substr_set.reserve(substring_to_index_helper.size());
                    for (const auto &pair : substring_to_index_helper)
                    {
                        substring_to_index_collector.insert(a, pair.first);
                        a->second.insert(a->second.begin(), pair.second.begin(), pair.second.end());
                        a.release();
                        substr_set.emplace(pair.first);
                    }
                    word_to_substring_collector.insert(b, word);
                    b->second = substr_set;
                    b.release();
                }
            });

        substring_to_index.insert(
            substring_to_index_collector.cbegin(),
            substring_to_index_collector.cend());

        word_to_substring.insert(
            word_to_substring_collector.cbegin(),
            word_to_substring_collector.cend());

        for (const auto &kv : word_to_index)
        {
            index_to_word[kv.second.first] = kv.first;
        }

        if (candidate_tokens.size() == 0)
        {
            cout << "Final candidate token set size: " << substring_to_index.size() << endl;
        }

        /* initialize more variables */
        T_arr = vector<int unsigned>(singleton_count, 0);
        D_arr = vector<int unsigned>(singleton_count, 0);
        auto stop = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
        cout << "Initial setup phase: " << duration.count() << " ms" << endl;
    }

    /**
     * @brief Get the total number of elements that we wish to cover
     *
     * @return unsigned long
     */
    unsigned long get_singleton_counts()
    {
        return singleton_count;
    }

    /**
     * @brief Get the candidate token size
     *
     * @return unsigned long
     */
    unsigned long get_candidate_token_size()
    {
        if (candidate_tokens.size() == 0)
        {
            return substring_to_index.size();
        }
        else
        {
            return candidate_tokens.size();
        }
    }

    /**
     * @brief Calculate scores of a given substring defined by its positions in the array
     *
     * @param places of SubstringPos locations of a substring
     * @param T_arr_ptr Array to track assigned tokens
     * @param D_arr_ptr Array to track duplicates
     * @param id_to_count word to count mapping
     * @return long unsigned : score of a particular substring
     */
    long unsigned calculate_score(
        const vector<SubstringPos> &places,
        const vector<int unsigned> *T_arr_ptr,
        const vector<int unsigned> *D_arr_ptr,
        const unordered_map<long unsigned, long unsigned> &id_to_count)
    {
        long unsigned counts = 0;
        long unsigned prev_end = 0;
        long unsigned current_arr_start = 0;
        for (const auto &p : places)
        {

            const long unsigned ws = p.arr_start;
            const long unsigned we = p.arr_end;
            const int i = p.substr_start;
            const int j = p.substr_end;

            if (p.arr_start != current_arr_start)
            {
                prev_end = 0;
                current_arr_start = p.arr_start;
            }

            if (ws + i < prev_end)
            {
                continue;
            }
            if (i > 0 && (*T_arr_ptr)[ws + i - 1] != 0 && (*T_arr_ptr)[ws + i - 1] == (*T_arr_ptr)[ws + i] && (*D_arr_ptr)[ws + i - 1] == (*D_arr_ptr)[ws + i])
            {
                continue;
            }
            if (ws + j < we && (*T_arr_ptr)[ws + j] != 0 && (*T_arr_ptr)[ws + j - 1] == (*T_arr_ptr)[ws + j] && (*D_arr_ptr)[ws + j - 1] == (*D_arr_ptr)[ws + j])
            {
                continue;
            }
            int nones = 0;
            vector<pair<int unsigned, int unsigned>> uniqs;
            uniqs.reserve(j - i);
            for (int k = i; k < j; ++k)
            {
                if ((*T_arr_ptr)[ws + k] == 0)
                {
                    nones += 1;
                }
                else
                {
                    uniqs.emplace_back(pair((*T_arr_ptr)[ws + k], (*D_arr_ptr)[ws + k]));
                }
            }
            sort(uniqs.begin(), uniqs.end());
            counts += id_to_count.at(ws) * (nones + (unique(uniqs.begin(), uniqs.end()) - uniqs.begin()) - 1);
            prev_end = ws + j;
        }
        return counts;
    }

    /**
     * @brief Change graph to reflect new state
     *
     * @param items Substring Positions to cover with rank_idx
     * @param T_arr_ptr Array to track assigned tokens
     * @param D_arr_ptr Array to track duplicates
     * @param rank_idx Assigning elements to tokens with rank
     * @return unordered_set<long unsigned> word start positions affected by change
     */
    unordered_set<long unsigned> alter_graph(
        const vector<SubstringPos> &items,
        vector<int unsigned> *T_arr_ptr,
        vector<int unsigned> *D_arr_ptr,
        const int &rank_idx)
    {

        unordered_set<long unsigned> visited;
        long unsigned prev_w_start = -1;
        int d_counter = 0;
        for (const auto &p : items)
        {
            const long unsigned ws = p.arr_start;
            const long unsigned we = p.arr_end;
            const int i = p.substr_start;
            const int j = p.substr_end;

            if (i > 0 && (*T_arr_ptr)[ws + i - 1] != 0 && (*T_arr_ptr)[ws + i - 1] == (*T_arr_ptr)[ws + i] && (*D_arr_ptr)[ws + i - 1] == (*D_arr_ptr)[ws + i])
            {
                continue;
            }
            if (ws + j < we && (*T_arr_ptr)[ws + j] != 0 && (*T_arr_ptr)[ws + j - 1] == (*T_arr_ptr)[ws + j] && (*D_arr_ptr)[ws + j - 1] == (*D_arr_ptr)[ws + j])
            {
                continue;
            }

            visited.insert(ws);
            if (ws != prev_w_start)
            {
                d_counter = 0;
            }
            if (ws == prev_w_start)
            {
                d_counter += 1;
            }
            for (long unsigned k = ws + i; k < ws + j; ++k)
            {
                (*T_arr_ptr)[k] = rank_idx;
                (*D_arr_ptr)[k] = d_counter;
            }
            prev_w_start = ws;
        }
        return visited;
    }

    vector<py::bytes> get_ranks()
    {
        vector<py::bytes> pybytes_ranks(0);
        pybytes_ranks.reserve(ranks.size());
        for (const auto &r : ranks)
        {
            pybytes_ranks.emplace_back(r);
        }
        return pybytes_ranks;
    }

    /**
     * @brief Advancing the current state with specific tokens
     *
     * @param tokens the order of tokens to be used
     * @return pair<vector<string>, vector<long unsigned>> current ranking of tokens and scores
     */
    pair<vector<py::bytes>, vector<long unsigned>> custom_steps(const vector<string> &tokens)
    {
        for (const string &token : tokens)
        {
            unsigned int rank = ranks.size();
            ranks.emplace_back(token);
            unsigned long score = calculate_score(substring_to_index[token], &T_arr, &D_arr, id_to_count);
            scores.emplace_back(score);
            alter_graph(substring_to_index[token], &T_arr, &D_arr, rank);
            cout << rank << ". |" << token << " [" << hex;
            for (auto c : token)
            {
                cout << (unsigned int)(unsigned char)c << " ";
            }
            cout << dec << "] | " << score << endl;
        }

        for (const auto &r : ranks)
        {
            shortlist.erase(r);
            results.erase(r);
        }
        return pair(get_ranks(), scores);
    }

    /**
     * @brief Advance the current state till we have k number of tokens
     *
     * @param k target number of tokens
     * @return pair<vector<string>, vector<long unsigned>> current ranking of tokens and scores
     */
    pair<vector<py::bytes>, vector<long unsigned>> solve_to_step(const unsigned int k)
    {
        auto total_start = chrono::high_resolution_clock::now();

        for (const auto &s : substring_to_index)
        {
            shortlist.insert(s.first);
        }
        for (const auto &s : shortlist)
        {
            results[s] = 0;
        }

        /* Main GreedTok routine */
        // cout << "Starting main routine..." << endl;
        for (unsigned int rank = ranks.size() + 1; rank <= k; ++rank)
        {
            auto start = chrono::high_resolution_clock::now();

            vector<string> items(shortlist.cbegin(), shortlist.cend());
            tbb::parallel_for(
                tbb::blocked_range<long unsigned>(0, items.size()),
                [&](tbb::blocked_range<long unsigned> r)
                {
                    for (long unsigned i = r.begin(); i < r.end(); ++i)
                    {
                        results[items[i]] = calculate_score(
                            substring_to_index[items[i]],
                            &T_arr,
                            &D_arr,
                            id_to_count);
                    }
                });

            pair<string, long unsigned> best = *max_element(
                execution::par_unseq,
                results.cbegin(),
                results.cend(),
                token_score_sorter);
            ranks.emplace_back(best.first);
            scores.emplace_back(best.second);

            auto stop = chrono::high_resolution_clock::now();
            auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
            unordered_set<long unsigned> visited = alter_graph(substring_to_index[best.first], &T_arr, &D_arr, rank);

            shortlist.clear();
            for (const auto v : visited)
            {
                shortlist.insert(
                    word_to_substring[index_to_word[v]].cbegin(),
                    word_to_substring[index_to_word[v]].cend());
            }
            for (const auto &r : ranks)
            {
                shortlist.erase(r);
            }
            results.erase(best.first);

            stop = chrono::high_resolution_clock::now();
            auto duration2 = chrono::duration_cast<chrono::milliseconds>(stop - start);
            cout << rank << ". |" << best.first << " [" << hex;
            for (const auto c : best.first)
            {
                cout << (unsigned int)(unsigned char)c << " ";
            }
            cout << dec << "] | " << best.second << " | " << duration.count() << " ms | " << duration2.count() << " ms | shortlist: " << shortlist.size() << endl;
        }
        auto total_duration = chrono::duration_cast<chrono::seconds>(chrono::high_resolution_clock::now() - total_start);
        cout << "Total time taken: " << total_duration.count() << " seconds" << endl;
        return pair(get_ranks(), scores);
    }
};

class PyGreedyPCOTokenizer : public GreedyPCOTokenizer
{
public:
    using GreedyPCOTokenizer::build_counter_from_text;
    using GreedyPCOTokenizer::calculate_score;
    using GreedyPCOTokenizer::custom_steps;
    using GreedyPCOTokenizer::get_candidate_token_size;
    using GreedyPCOTokenizer::get_ranks;
    using GreedyPCOTokenizer::get_singleton_counts;
    using GreedyPCOTokenizer::initialize_graph;
    using GreedyPCOTokenizer::solve_to_step;
};

GreedyPCOTokenizer *build(
    unordered_map<string, long unsigned> word_counts = {},
    unordered_set<string> candidate_tokens = {})
{
    return new GreedyPCOTokenizer(word_counts, candidate_tokens);
}

PYBIND11_MODULE(pco_tokenizer, var)
{
    var.doc() = "greedy module";
    py::class_<GreedyPCOTokenizer, PyGreedyPCOTokenizer>(var, "GreedyPCOTokenizer")
        .def(py::init<>(
            [](
                unordered_map<string, long unsigned> word_counts = {},
                unordered_set<string> candidate_tokens = {})
            {
                return new GreedyPCOTokenizer(
                    word_counts,
                    candidate_tokens);
            }))
        .def("get_ranks", &GreedyPCOTokenizer::get_ranks)
        .def("solve_to_step", &GreedyPCOTokenizer::solve_to_step)
        .def("calculate_score", &GreedyPCOTokenizer::calculate_score)
        .def("initialize_graph", &GreedyPCOTokenizer::initialize_graph)
        .def("alter_graph", &GreedyPCOTokenizer::alter_graph)
        .def("custom_steps", &GreedyPCOTokenizer::custom_steps)
        .def("build_counter_from_text", &GreedyPCOTokenizer::build_counter_from_text)
        .def("get_singleton_counts", &GreedyPCOTokenizer::get_singleton_counts)
        .def("get_candidate_token_size", &GreedyPCOTokenizer::get_candidate_token_size);
    var.def("build",
            &build,
            py::arg("word_counts") = unordered_map<string, long unsigned>(),
            py::arg("candidate_tokens") = unordered_set<string>(),
            "Factory function for greedy PCO tokenizer, use this to create your token sets.");
}