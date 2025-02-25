#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/stl.h>
namespace py = pybind11;
#include <chrono>
#include <execution>
#include <iostream>
#include <limits.h>
#include <numeric>
#include <queue>
#include <regex>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "tbb.h"
using namespace std;
namespace chrono = std::chrono;

/*
c++ -O3 -Wall -shared -std=c++23 \
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

class Compare
{
public:
    bool operator()(
        const pair<string, unsigned long> a,
        const pair<string, unsigned long> b)
    {
        if (a.second == b.second)
        {
            return a.first.size() > b.first.size();
        }
        return a.second < b.second;
    }
};

// static bool min_token_score_compare(
//     const pair<string, long unsigned> a,
//     const pair<string, long unsigned> b)
// {
//     if (a.second == b.second)
//     {
//         return a.first < b.first;
//     }
//     return a.second > b.second;
// }

class ResultsCache
{
    unordered_set<string> blacklist;

public:
    bool initialized = false;
    priority_queue<pair<string, unsigned long>,
                   vector<pair<string, unsigned long>>,
                   Compare>
        large_heap;
    unsigned int shortlist_size;
    ResultsCache(unsigned int shortlist_size)
        : shortlist_size(shortlist_size)
    {
        cout << "shortlist size: " << shortlist_size << endl;
    }

    virtual ~ResultsCache() {}

    void init(vector<pair<string, unsigned long>> &results)
    {
        large_heap = priority_queue<
            pair<string, unsigned long>,
            vector<pair<string, unsigned long>>,
            Compare>(results.begin(), results.end());
        initialized = true;
    }

    // void make_mini_heap()
    // {
    //     // nothing to init or enough
    //     if (large_heap.size() == 0 || mini_heap.size() > shortlist_size)
    //     {
    //         return;
    //     }
    //     while (mini_heap.size() < shortlist_size && large_heap.size() != 0)
    //     {
    //         pop_heap(large_heap.begin(), large_heap.end(), max_token_score_compare);
    //         pair<string, unsigned long> p = large_heap.back();
    //         large_heap.pop_back();
    //         if (blacklist.find(p.first) != blacklist.end())
    //         {
    //             continue;
    //         }
    //         if (p.second < current_threshold)
    //         {
    //             current_threshold = p.second;
    //         }
    //         cout << "adding: " << p.first << " " << p.second << endl;
    //         mini_heap.emplace_back(p);
    //     }
    //     make_heap(mini_heap.begin(), mini_heap.end(), max_token_score_compare);
    //     cout << "init mini heap add " << shortlist_size - mini_heap.size() << endl;
    // }

    vector<string> get_checkables(const unordered_set<string> &shortlist_set)
    {
        vector<string> to_check_again{};

        while (to_check_again.size() < shortlist_size && large_heap.size() > 0)
        {
            pair<string, unsigned long> p = large_heap.top();
            // cout << "checking :" << p.first << " " << p.second << endl;
            if (blacklist.find(p.first) != blacklist.end())
            {
                large_heap.pop();
                continue;
            }
            if (shortlist_set.find(p.first) != shortlist_set.end())
            {
                to_check_again.push_back(p.first);
                large_heap.pop();
            }
            else
            {
                break;
            }
        }
        // cout << "get checkables: " << to_check_again.size() << endl;
        return to_check_again;
    }

    pair<string, unsigned long> pop_best()
    {
        const pair<string, unsigned long> p = large_heap.top();
        large_heap.pop();
        // cout << "RETURNING " << p.first << p.second << endl;
        return p;
    }

    void update(const vector<pair<string, unsigned long>> &updates)
    {
        for (const pair<string, unsigned long> &u : updates)
        {
            large_heap.push(u);
        }
    }

    void erase(const string &s)
    {
        blacklist.emplace(s);
    }
};

class GreedyPCOTokenizer
{

public:
    long unsigned singleton_count = 0;
    ResultsCache results;
    vector<string> ranks;
    vector<int unsigned> T_arr;
    vector<int unsigned> D_arr;
    vector<long unsigned> scores;
    unordered_set<string> altered;
    unordered_set<string> candidate_tokens{};
    unordered_map<string, long unsigned> word_counts;
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
        unordered_map<string, long unsigned> word_counts = {},
        unordered_set<string> candidate_tokens = {},
        unsigned long shortlist_size = 100)
        : results(ResultsCache(shortlist_size)),
          candidate_tokens(candidate_tokens),
          word_counts(word_counts)
    {
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
            auto p = word_counts.try_emplace(item.first, 0);
            p.first->second += item.second;
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
        for (const auto &item : word_counts)
        {

            singleton_count += item.first.size();

            end_id = next_id + item.first.size();
            id_to_count[next_id] = item.second;

            word_to_index[item.first] = pair(next_id, end_id);
            word_to_substring[item.first] = unordered_set<string>();
            for (unsigned int i = 0; i < item.first.size(); ++i)
            {
                for (unsigned int j = i + 2; j < min(max_token_length + i, item.first.size() + 1); ++j)
                {
                    if (item.second < min_word_count)
                    {
                        continue;
                    }
                    const string substr = item.first.substr(i, j - i);
                    if (substr.size() <= 1)
                    {
                        continue;
                    }
                    auto p = substring_to_index.try_emplace(substr, vector<SubstringPos>());
                    p.first->second.push_back({next_id, end_id, i, j});
                    word_to_substring[item.first].insert(move(substr));
                }
            }
            next_id = end_id;
        }

        for (const auto &kv : word_to_index)
        {
            index_to_word[kv.second.first] = move(kv.first);
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

    static void print_step(
        const unsigned int rank,
        const string &token,
        const unsigned long score,
        const string &suffix = "")
    {
        cout << rank << ". |" << token << " [" << hex;
        for (auto c : token)
        {
            cout << (unsigned int)(unsigned char)c << " ";
        }
        cout << dec << "] | " << score << " " << suffix << endl;
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
            unsigned int rank = ranks.size() + 1;
            ranks.emplace_back(token);
            unsigned long score = calculate_score(substring_to_index[token], &T_arr, &D_arr, id_to_count);
            scores.emplace_back(score);
            unordered_set<long unsigned> visited = alter_graph(substring_to_index[token], &T_arr, &D_arr, rank);
            print_step(rank, token, score);
            for (const auto v : visited)
            {
                altered.insert(
                    word_to_substring[index_to_word[v]].cbegin(),
                    word_to_substring[index_to_word[v]].cend());
            }
        }
        for (const auto &r : ranks)
        {
            altered.erase(r);
            results.erase(r);
        }
        return pair(get_ranks(), scores);
    }

    void init_sorted_heap()
    {
        unordered_set<string> ranks_cache(ranks.cbegin(), ranks.cend());
        for (const auto &p : substring_to_index)
        {
            if (ranks_cache.find(p.first) == ranks_cache.end())
            {
                altered.emplace(p.first);
            }
        }
        vector<string> items(altered.cbegin(), altered.cend());
        vector<pair<string, unsigned long>> all = solve(items);
        results.init(all);
        altered.clear();
    }

    vector<pair<string, unsigned long>> solve(const vector<string> &items)
    {
        if (items.size() == 0)
        {
            return {};
        }
        vector<pair<string, unsigned long>> token_score_pairs(items.size());
        tbb::parallel_for(
            tbb::blocked_range<long unsigned>(0, items.size()),
            [&](tbb::blocked_range<long unsigned> r)
            {
                for (long unsigned i = r.begin(); i < r.end(); ++i)
                {
                    token_score_pairs[i] = pair(
                        items[i],
                        calculate_score(
                            substring_to_index[items[i]],
                            &T_arr,
                            &D_arr,
                            id_to_count));
                }
            });
        return token_score_pairs;
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
        auto start = chrono::high_resolution_clock::now();
        size_t num_checked = 0;

        // if not initialized, count everything
        if (!results.initialized)
        {
            init_sorted_heap();
            num_checked += substring_to_index.size();
            altered.clear();
        }

        for (unsigned int rank = ranks.size() + 1; rank <= k; ++rank)
        {
            vector<string> to_check = results.get_checkables(altered);
            while (to_check.size() > 0)
            {
                vector<pair<string, unsigned long>> token_score_pairs = solve(to_check);
                num_checked += to_check.size();
                results.update(token_score_pairs);
                for (const string &t : to_check)
                {
                    altered.erase(t);
                }
                to_check = results.get_checkables(altered);
            }
            pair<string, long unsigned> best = results.pop_best();
            ranks.emplace_back(best.first);
            scores.emplace_back(best.second);

            auto stop = chrono::high_resolution_clock::now();
            auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
            unordered_set<long unsigned> visited = alter_graph(substring_to_index[best.first], &T_arr, &D_arr, rank);
            for (const auto v : visited)
            {
                altered.insert(
                    word_to_substring[index_to_word[v]].cbegin(),
                    word_to_substring[index_to_word[v]].cend());
            }
            for (const auto &r : ranks)
            {
                altered.erase(r);
            }

            stop = chrono::high_resolution_clock::now();
            auto duration2 = chrono::duration_cast<chrono::milliseconds>(stop - start);
            print_step(rank, best.first, best.second,
                       " | " + to_string(duration.count()) + " ms | " + to_string(duration2.count()) + " ms | num. checks: " + to_string(num_checked) + " | altered size: " + to_string(altered.size()));
            start = chrono::high_resolution_clock::now();
            num_checked = 0;
        }
        auto total_duration = chrono::duration_cast<chrono::seconds>(chrono::high_resolution_clock::now() - total_start);
        cout << "Time taken for steps: " << total_duration.count() << " seconds" << endl;
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
    using GreedyPCOTokenizer::init_sorted_heap;
    using GreedyPCOTokenizer::initialize_graph;
    using GreedyPCOTokenizer::print_step;
    using GreedyPCOTokenizer::solve;
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