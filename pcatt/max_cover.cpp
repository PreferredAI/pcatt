#include <iostream>
#include <iterator>
#include <fstream>
#include <string>
#include <unordered_set>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include "oneapi/tbb.h"

using namespace std;
namespace chrono = std::chrono;

/*
c++ -O3 -std=c++20 \
-I$CONDA_PREFIX/include/ \
-I$CONDA_PREFIX/include/tbb \
-I$CONDA_PREFIX/include/oneapi \
-L$CONDA_PREFIX/lib/ \
-l tbb \
pcatt/max_cover.cpp \
-o pcatt/max_cover.exe
*/
vector<string> splitString(string &str, const char &splitter)
{
    vector<string> result;
    string current = "";
    for (int i = 0; i < str.size(); i++)
    {
        if (str[i] == splitter)
        {
            if (current != "")
            {
                result.push_back(current);
                current = "";
            }
            continue;
        }
        current += str[i];
    }
    if (current.size() != 0)
        result.push_back(current);
    return result;
}

unordered_map<string, long unsigned> get_counts(const string &domain)
{
    vector<string> keys;
    vector<long unsigned> values;
    unordered_map<string, long unsigned> outputs;
    fstream data_file;

    data_file.open("cpp_inputs/counts/" + domain + ".txt", ios::in);
    if (data_file.is_open())
    {
        string data;
        while (getline(data_file, data))
        {
            values.push_back(stoi(data));
        }
    }
    data_file.close();
    data_file.open("cpp_inputs/words/" + domain + ".txt", ios::in);
    string full_string = "";
    if (data_file.is_open())
    {
        string data;
        while (getline(data_file, data))
        {
            full_string = full_string + data + '\n';
        }
        full_string.pop_back();
        for (string &s : splitString(full_string, ' '))
        {
            keys.push_back(s);
        }
    }

    cout << "keys size " << keys.size() << endl;
    cout << "values size " << values.size() << endl;

    for (int i = keys.size() - 1; i >= 0; --i)
    {
        outputs[keys[i]] = values[i];
    }

    return outputs;
}

struct SubstringPos
{
    long unsigned arr_start;
    long unsigned arr_end;
    int substr_start;
    int substr_end;
    SubstringPos() {};
    SubstringPos(long unsigned a, long unsigned b, int c, int d)
    {
        arr_start = a;
        arr_end = b;
        substr_start = c;
        substr_end = d;
    }
};

bool sp_sorter(SubstringPos const &lhs, SubstringPos const &rhs)
{
    return lhs.substr_start < rhs.substr_start;
}

long unsigned get_score_helper(const vector<SubstringPos> &places, const vector<bool> *T_arr_ptr, const unordered_map<long unsigned, long unsigned> &id_to_count)
{
    long unsigned counts = 0;

    unordered_map<long unsigned, vector<SubstringPos>> pplaces;
    for (auto p : places)
    {
        if (pplaces.find(p.arr_start) == pplaces.end())
        {
            pplaces[p.arr_start] = vector<SubstringPos>();
        }
        pplaces[p.arr_start].push_back(p);
    }

    for (auto pp : pplaces)
    {
        long unsigned prev_end = 0;
        sort( // execution::par,
            pp.second.begin(), pp.second.end(), &sp_sorter);
        for (auto &p : pp.second)
        {
            const long unsigned ws = p.arr_start;
            const long unsigned we = p.arr_end;
            const int i = p.substr_start;
            const int j = p.substr_end;
            const int start = ws + i < prev_end ? prev_end : i;
            int cover = 0;
            for (int k = start; k < j; ++k)
            {
                if ((*T_arr_ptr)[ws + k] == 0)
                {
                    cover += 1;
                }
            }
            counts += id_to_count.at(ws) * cover;
            prev_end = ws + j;
        }
    }
    return counts;
}

unordered_set<long unsigned> alter_graph(const vector<SubstringPos> &items, vector<bool> *T_arr_ptr)
{

    unordered_set<long unsigned> visited;
    long unsigned prev_w_start = -1;
    int d_counter = 0;
    for (auto &p : items)
    {
        const long unsigned ws = p.arr_start;
        const long unsigned we = p.arr_end;
        const int i = p.substr_start;
        const int j = p.substr_end;

        visited.insert(ws);
        for (long unsigned k = ws + i; k < ws + j; ++k)
        {
            (*T_arr_ptr)[k] = true;
        }
        prev_w_start = ws;
    }
    return visited;
}

int main(int argc, char *argv[])
{
    string domain = argv[1];
    int k = stoi(argv[2]);
    auto start = chrono::high_resolution_clock::now();
    unordered_map<string, long unsigned> word_counts = get_counts(domain);

    cout << "counts size " << word_counts.size() << endl;

    long unsigned char_count = 0;
    long unsigned next_id = 0;
    long unsigned end_id = 0;
    unordered_map<long unsigned, long unsigned> id_to_count;
    unordered_map<string, pair<long unsigned, long unsigned>>
        word_to_index;
    unordered_map<long unsigned, string> index_to_word;
    unordered_map<string, vector<SubstringPos>> substring_to_index;
    unordered_map<string, unordered_set<string>> word_to_substring;

    for (auto &item : word_counts)
    {
        if (item.first.size() <= 1)
        {
            continue;
        }
        char_count += item.first.size() - 1;
        end_id = next_id + item.first.size() - 1;
        id_to_count[next_id] = item.second;

        word_to_index[item.first] = pair(next_id, end_id);
        word_to_substring[item.first] = unordered_set<string>();
        for (int i = 0; i < item.first.size(); ++i)
        {

            for (int j = i + 1; j < item.first.size() + 1; ++j)
            {
                string substr = item.first.substr(i, j - i);
                if (substr.size() <= 1)
                {
                    continue;
                }
                if (substring_to_index.find(substr) == substring_to_index.end())
                {
                    substring_to_index[substr] = vector<SubstringPos>();
                }
                substring_to_index[substr].push_back({next_id, end_id, i, j - 1});
                word_to_substring[item.first].insert(substr);
            }
        }
        next_id = end_id;
    }

    for (auto &kv : word_to_index)
    {
        index_to_word[kv.second.first] = kv.first;
    }

    vector<bool> T_arr(char_count, 0);
    vector<bool> D_arr(char_count, 0);
    unordered_set<string> shortlist;
    vector<string> saved_merges;
    vector<string> ranks;
    vector<long unsigned> scores;
    for (auto &s : substring_to_index)
    {
        shortlist.insert(s.first);
    }
    unordered_map<string, long unsigned> results;
    for (auto &s : shortlist)
    {
        results[s] = 0;
    }
    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
    cout << "Initial Setup: "
         << duration.count() << " ms" << endl;

    cout << "starting" << endl;
    for (int rank = 1; rank <= k; ++rank)
    {
        start = chrono::high_resolution_clock::now();

        vector<string> items(shortlist.begin(), shortlist.end());
        oneapi::tbb::parallel_for(tbb::blocked_range<long unsigned>(0, items.size()), [&](tbb::blocked_range<long unsigned> r)
                                  { for (long unsigned i=r.begin(); i<r.end(); ++i){
                    results[items[i]] = get_score_helper(substring_to_index[items[i]], &T_arr, id_to_count); } });

        pair<string, long unsigned> best = *max_element(results.begin(), results.end(),
                                                        [](const pair<string, long unsigned> a, const pair<string, long unsigned> b)
                                                        {
                                                            if (a.second == b.second)
                                                            {
                                                                return a.first.size() < b.first.size();
                                                            }
                                                            else
                                                            {
                                                                return a.second < b.second;
                                                            }
                                                        });
        ranks.push_back(best.first);
        scores.push_back(best.second);

        stop = chrono::high_resolution_clock::now();
        duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
        unordered_set<long unsigned> visited = alter_graph(substring_to_index[best.first], &T_arr);

        shortlist.clear();
        for (auto &v : visited)
        {
            shortlist.insert(word_to_substring[index_to_word[v]].begin(),
                             word_to_substring[index_to_word[v]].end());
        }
        for (auto &r : ranks)
        {
            shortlist.erase(r);
        }
        results.erase(best.first);

        stop = chrono::high_resolution_clock::now();
        auto duration2 = chrono::duration_cast<chrono::milliseconds>(stop - start);
        cout << rank << ". |" << best.first << " [" << hex;
        for (auto c : best.first)
        {
            cout << (unsigned int)(unsigned char)c << " ";
        }
        cout << dec << "] | " << best.second << " | " << duration.count() << " ms | " << duration2.count() << " ms | shortlist: " << shortlist.size() << endl;
    }

    string out_dir = "cpp_outputs/" + domain;
    if (!filesystem::is_directory(out_dir) || !filesystem::exists(out_dir))
    {
        filesystem::create_directory(out_dir);
    }
    ofstream f;
    f.open(out_dir + "/max_cover_seq.txt");
    f << hex << setfill('0');
    for (auto r : ranks)
    {
        for (auto c : r)
        {
            f << setw(2) << (unsigned int)(unsigned char)c << " ";
        }
        f << endl;
    }
    f << dec;
    f.close();
    f.open(out_dir + "/max_cover_count.txt");
    for (auto s : scores)
    {
        f << s << endl;
    }
    f.close();
}
