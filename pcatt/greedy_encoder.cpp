#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
namespace py = pybind11;
#include <iostream>
#include <chrono>
#include <set>
#include <algorithm>
#include <string>
#include <numeric>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <stdexcept>
#include <regex>
#include <limits.h>
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
./pcatt/greedy_encoder.cpp \
-o ./pcatt/greedy_encoder$(python3-config --extension-suffix)
*/

struct TrieNode
{
public:
    unordered_map<char, TrieNode *> to;
    unsigned int rank = 0;
    unsigned int level;
    char c = -1;
    TrieNode()
    {
        to = unordered_map<char, TrieNode *>{};
        level = -1;
    }

    /**
     * @brief Construct a new Trie Node object, meant for internal use
     *
     * @param current_char
     * @param current_level
     */
    TrieNode(const char current_char, const unsigned int current_level)
    {
        to = unordered_map<char, TrieNode *>{};
        c = current_char;
        level = current_level;
    }
    virtual ~TrieNode() {};

    /**
     * @brief Construct a forward-only linked list using Trie Node objects
     *
     * @param s string of characters/singletons
     * @param start start from index of string
     * @param rank_idx to be assigned to the Trie Node that ends s
     */
    void add_new_sequence(const string &s, const unsigned int &start, const unsigned int &rank_idx)
    {

        if (auto search = to.find(s[start]); search == to.end())
        {
            to[s[start]] = new TrieNode(s[start], level + 1);
        }
        if (start + 1 == s.size())
        { // end of sequence
            to[s[start]]->rank = rank_idx;
        }
        else
        {
            to[s[start]]->add_new_sequence(s, start + 1, rank_idx);
        }
        return;
    }

    /**
     * @brief Determines if there is a Trie Node in c direction
     *
     * @param c direction to traverse
     * @return true if there is no Trie Node
     * @return false otherwise
     */
    bool is_new_path(const char &c)
    {
        return to.find(c) == to.end();
    }

    /**
     * @brief Link a new Trie Node to current Trie Node
     *
     * @param t new Trie Node
     */
    void add_new_path(TrieNode &t)
    {
        to[t.c] = &t;
    }

    /**
     * @brief Get the next Trie Node in c direction
     *
     * @param c
     * @return TrieNode*
     */
    TrieNode *get_next_path(const char &c)
    {
        return to[c];
    }
};

struct CoverPos
{
    unsigned int start;
    unsigned int offset;
    unsigned int rank;
    CoverPos() {};
    /**
     * @brief Construct a new Cover Pos object, meant for internal use
     *
     * @param start_idx of the cover in text
     * @param num_char that separates the start and end indices
     * @param substr_rank rank assigned to this cover
     */
    CoverPos(
        const unsigned int start_idx,
        const unsigned int num_char,
        const unsigned int substr_rank)
    {
        start = start_idx;
        offset = num_char;
        rank = substr_rank;
    }
};

class TrieCache
{
public:
    TrieNode root{};
    vector<string> rules;
    TrieCache() {};
    /**
     * @brief Construct a new Trie Cache object, meant for internal use
     * We use a Trie Cache to interate over text as std::regex is slow.
     * This helps us to determine if there is a possible cover in O(N)
     *
     * @param rules order of tokens in decreasing cover priority
     */
    TrieCache(vector<string> &rules) : rules(rules)
    {
        for (unsigned int i = 0; i < rules.size(); ++i)
        {
            string rule = rules.at(i);
            if (rule.size() <= 1)
            {
                continue;
            }
            root.add_new_sequence(rules.at(i), 0, i + 1);
        }
        cout << "Trie constructed" << endl;
    }
    virtual ~TrieCache() {}

    /**
     * @brief Determine if there is a subsequence that correspond to a token
     *
     * @param s string to investigate
     * @param start_idx starting from which index
     * @param end_idx ending at which index
     * @param final_location to store results of possible Cover Pos
     * @return unsigned int of which index the traversal stopped at
     */
    unsigned int traverse(
        const string &s,
        const unsigned int &start_idx,
        const unsigned int &end_idx,
        vector<CoverPos> *final_location)
    {
        TrieNode *prev = &root;
        if (prev->is_new_path(s[start_idx]))
        {
            return start_idx + 1;
        }
        prev = prev->to[s[start_idx]];
        unsigned int curr_idx = start_idx;
        for (; curr_idx < end_idx;)
        {
            curr_idx++;
            if (prev->is_new_path(s[curr_idx]))
            {
                break;
            }
            prev = prev->to[s[curr_idx]];
            if (prev->rank != 0)
            {
                final_location->emplace_back(CoverPos(start_idx, prev->level + 1, prev->rank));
            }
        }
        return curr_idx;
    }
};

enum class TruncationStrategy : char
{
    ONLY_FIRST,
    ONLY_SECOND,
    LONGEST_FIRST,
    DO_NOT_TRUNCATE
};

enum class TruncationSide : char
{
    RIGHT,
    LEFT
};

enum class PaddingStrategy : char
{
    LONGEST,
    MAX_LENGTH,
    DO_NOT_PAD,
};

enum class PaddingSide : char
{
    RIGHT,
    LEFT
};

class GreedyTokenizer
{
    unordered_map<string, int unsigned> rules_cache;
    unordered_map<char, int unsigned> singleton_cache;
    TrieCache trie_cache;
    regex re;
    sregex_token_iterator end;
    vector<string> rules;
    unsigned int max_token_size = 0;
    unsigned int _max_length = UINT16_MAX;
    unsigned int _pad_to_multiple_of = 1;
    unordered_map<string, string> _special_tokens;
    unordered_map<string, unsigned int> _special_tokens_map;
    unordered_set<unsigned int> _special_token_ids;
    TruncationStrategy _truncation_strategy = TruncationStrategy::DO_NOT_TRUNCATE;
    TruncationSide _truncation_side = TruncationSide::RIGHT;
    PaddingSide _padding_side = PaddingSide::RIGHT;
    PaddingStrategy _padding_strategy = PaddingStrategy::DO_NOT_PAD;

    /**
     * @brief custom function to sort CoverPos, meant for internal use
     *
     * @param lhs
     * @param rhs
     * @return true if lhs have a greater cover priority than rhs
     * @return false otherwise
     */
    static bool cover_pos_sorter(CoverPos const &lhs, CoverPos const &rhs)
    {
        if (lhs.rank == rhs.rank)
        {
            return lhs.start < rhs.start;
        }
        return lhs.rank < rhs.rank;
    }

public:
    /**
     * @brief Construct a new Greedy Tokenizer object
     *
     * @param rules_input order of tokens in decreasing cover priority
     */
    GreedyTokenizer(
        vector<string> rules_input,
        unordered_map<string, string> special_tokens)
    {
        trie_cache = TrieCache(rules_input);
        for (int unsigned i = 0; i < rules_input.size(); ++i)
        {
            rules_cache[rules_input.at(i)] = 1 + rules.size();
            rules.emplace_back(rules_input.at(i));
            if (rules_input.at(i).size() > max_token_size)
            {
                max_token_size = rules_input.at(i).size();
            }
        }
        for (int unsigned i = 0; i < 256; ++i)
        {
            string b({(char)i});
            if (auto search = rules_cache.find(b); search == rules_cache.end())
            {
                rules.emplace_back(b);
                singleton_cache[(char)i] = rules.size();
                rules_cache[b] = rules.size();
            }
            else
            {
                singleton_cache[(char)i] = search->second;
            }
        }
        _special_tokens = special_tokens;
        for (auto t : special_tokens)
        {
            _special_token_ids.insert(rules_cache.at(t.second) - 1);
            unsigned int sp_idx = rules_cache.at(t.second) - 1;
            cout << t.first << " " << t.second << " " << sp_idx << endl;
            _special_tokens_map.emplace(t.first, sp_idx);
        }
    }
    virtual ~GreedyTokenizer() {}

    unordered_set<unsigned int> get_special_token_ids()
    {
        return _special_token_ids;
    }

    void set_post_embedding_strategy(
        TruncationSide truncation_side = TruncationSide::RIGHT,
        TruncationStrategy truncation_strategy = TruncationStrategy::DO_NOT_TRUNCATE,
        PaddingSide padding_side = PaddingSide::RIGHT,
        PaddingStrategy padding_strategy = PaddingStrategy::DO_NOT_PAD,
        unsigned int max_length = UINT16_MAX,
        unsigned int pad_to_multiple_of = 1)
    {
        _truncation_strategy = truncation_strategy;
        _truncation_side = truncation_side;
        _padding_side = padding_side;
        _padding_strategy = padding_strategy;
        if (max_length > 0)
        {
            _max_length = max_length;
        }
        _pad_to_multiple_of = 1;
    }

    /**
     * @brief Get the rules at token position
     *
     * @param index of token
     * @return vector<uint8_t> representation of the token
     */
    py::bytes get_rule(int index)
    {
        return py::bytes(rules.at(index));
    }

    unsigned int get_rules_size()
    {
        return rules.size();
    }

    /**
     * @brief Tokenize text that were already pre-split
     *
     * @param text list of words
     * @return vector<int unsigned> list of tokens' ids
     */
    vector<int unsigned> tokenize_presplit(const vector<string> &text)
    {
        vector<int unsigned> results;

        for (unsigned int i = 0; i < text.size(); ++i)
        {
            tokenize_text(text.at(i), &results);
        };
        return results;
    }

    /**
     * @brief Tokenize text that were already pre-split
     *
     * @param texts list of lists of words
     * @return vector<vector<int unsigned>> list of lists of tokens' ids
     */
    vector<vector<int unsigned>> batch_tokenize_presplit(const vector<vector<string>> &texts)
    {
        vector<vector<int unsigned>> results(texts.size(), vector<int unsigned>{});
        tbb::parallel_for(
            tbb::blocked_range<int unsigned>(0, texts.size()),
            [&](tbb::blocked_range<int unsigned> r)
            {
                for (int unsigned i = r.begin(); i < r.end(); ++i)
                {
                    results[i] = tokenize_presplit(texts.at(i));
                }
            });
        return results;
    }

    /**
     * @brief tokenizes a portion of texts, O(W^2lgW) complexity
     *
     * @param text
     * @param start_idx starting at which index
     * @param offset difference in start and end indices
     * @param to_check a list of Cover Pos objects to investigate
     * @param result place to add tokens' ids
     */
    void tokenize_portion(const string &text, const unsigned int &start_idx, const unsigned int &offset, vector<CoverPos> &to_check, vector<int unsigned> *result)
    {
        if (offset == 1)
        {
            result->emplace_back(singleton_cache.at(text[start_idx]) - 1);
            return;
        }
        if (auto search = rules_cache.find(text.substr(start_idx, offset)); search != rules_cache.end())
        {
            result->emplace_back(search->second - 1);
            return;
        }
        vector<int unsigned> T_arr(offset, 0);

        sort(to_check.begin(), to_check.end(), &cover_pos_sorter);
        for (CoverPos cover : to_check)
        {
            int unsigned i = cover.start - start_idx;
            int unsigned i_o = i - 1;
            if (i > 0 && T_arr[i_o] != 0 && T_arr[i_o] == T_arr[i])
            {
                continue;
            }
            int unsigned j = i + cover.offset;
            int unsigned j_o = j - 1;
            if (j < offset && T_arr[j] != 0 && T_arr[j_o] == T_arr[j])
            {
                continue;
            }
            for (int unsigned k = i; k < j; ++k)
            {
                T_arr[k] = cover.rank;
            }
        }

        for (int unsigned i = 0; i < T_arr.size(); ++i)
        {
            if (T_arr[i] == 0)
            {
                result->emplace_back(singleton_cache.at(text[start_idx + i]) - 1);
            }
            else
            {
                result->emplace_back(T_arr[i] - 1);
                i += rules.at(T_arr[i] - 1).size() - 1;
            }
        }
    }

    /**
     * @brief Tokenize whole strings without splitting
     *
     * @param text
     * @param token_results place to store tokenization results
     */
    void tokenize_text(
        const string &text,
        vector<unsigned int> *token_results)
    {
        for (unsigned int start_idx = 0; start_idx < text.size(); start_idx++)
        {
            vector<CoverPos> intermediate_results;
            unsigned int end_idx = text.size() < start_idx + max_token_size ? text.size() : start_idx + max_token_size;
            unsigned int best_stop_idx = trie_cache.traverse(text, start_idx, end_idx, &intermediate_results);
            for (unsigned int mid_idx = start_idx + 1; mid_idx < best_stop_idx; mid_idx++)
            {
                unsigned int next_stop_idx = trie_cache.traverse(text, mid_idx, mid_idx + max_token_size, &intermediate_results);
                best_stop_idx = best_stop_idx < next_stop_idx ? next_stop_idx : best_stop_idx;
            }
            tokenize_portion(text, start_idx, best_stop_idx - start_idx, intermediate_results, token_results);
            start_idx = best_stop_idx - 1;
        }
    }

    /**
     * @brief batch tokenize whole strings, O(W^2lgW) complexity
     *
     * @param texts list of strings
     * @return vector<vector<int unsigned>>
     */
    vector<vector<int unsigned>> batch_tokenize_whole(
        const vector<string> &texts)
    {
        vector<vector<int unsigned>> results(texts.size());
        tbb::parallel_for(
            tbb::blocked_range<int unsigned>(0, texts.size()),
            [&](tbb::blocked_range<int unsigned> r)
            {
                for (int unsigned i = r.begin(); i < r.end(); ++i)
                {
                    tokenize_text(texts.at(i), &results.at(i));
                }
            });
        return results;
    }

    /**
     * @brief Set the regex pattern object
     *
     * @param pattern
     */
    void set_regex_pattern(string pattern)
    {
        re = regex(pattern, regex::optimize);
    }

    /**
     * @brief Split texts using std::regex, then tokenize texts
     *
     * @param texts list of strings
     * @return vector<vector<int unsigned>>
     */
    vector<vector<int unsigned>> batch_split_and_tokenize(vector<string> &texts)
    {
        vector<vector<int unsigned>> results(texts.size(), vector<int unsigned>{});
        tbb::parallel_for(
            tbb::blocked_range<int unsigned>(0, texts.size()),
            [&](tbb::blocked_range<int unsigned> r)
            {
                for (int unsigned i = r.begin(); i < r.end(); ++i)
                {
                    vector<string> text(sregex_token_iterator(texts[i].begin(), texts[i].end(), re), end);
                    results[i] = tokenize_presplit(text);
                }
            });
        return results;
    }

    void pad(
        vector<int unsigned> *encoded_input,
        unsigned int num_pads_to_add,
        unsigned int pad_token_id)
    {
        if (_padding_side == PaddingSide::LEFT)
        {
            encoded_input->insert(
                encoded_input->begin(),
                num_pads_to_add,
                pad_token_id);
        }
        else
        {
            encoded_input->insert(
                encoded_input->end(),
                num_pads_to_add,
                pad_token_id);
        }
    }

    pair<vector<int unsigned>, vector<int unsigned>> truncate_sequence_get_overflow(
        const vector<int unsigned> &ids,
        int unsigned num_tokens_to_remove = 0,
        int unsigned stride = 0)
    {
        // new_ids, overflowing
        if (ids.size() > num_tokens_to_remove)
        {
            int unsigned window_length = min((int unsigned)ids.size(), stride + num_tokens_to_remove);
            if (_truncation_side == TruncationSide::RIGHT)
            {
                return pair(
                    vector<int unsigned>(ids.begin(), ids.begin() + (ids.size() - num_tokens_to_remove)),
                    vector<int unsigned>(ids.begin() + (ids.size() - window_length), ids.end()));
            }
            else if (_truncation_side == TruncationSide::LEFT)
            {
                return pair(
                    vector<int unsigned>(ids.begin() + num_tokens_to_remove, ids.end()),
                    vector<int unsigned>(ids.begin(), ids.begin() + window_length));
            }
        }
        return pair(ids, vector<int unsigned>{});
    }

    vector<int unsigned> truncate_sequence(
        const vector<int unsigned> &ids,
        int unsigned num_tokens_to_remove = 0,
        int unsigned stride = 0)
    {
        // new_ids, overflowing
        if (ids.size() > num_tokens_to_remove)
        {
            if (_truncation_side == TruncationSide::RIGHT)
            {
                return vector<int unsigned>(ids.begin(), ids.begin() + (ids.size() - num_tokens_to_remove));
            }
            else if (_truncation_side == TruncationSide::LEFT)
            {
                return vector<int unsigned>(ids.begin() + num_tokens_to_remove, ids.end());
            }
        }
        return ids;
    }

    pair<vector<int unsigned>, vector<int unsigned>> truncate_pair_longest(
        vector<int unsigned> &ids,
        vector<int unsigned> &pair_ids,
        int num_tokens_to_remove = 0)
    {
        // new_ids, new_pair_ids
        int larger_size = max(ids.size(), pair_ids.size());
        int smaller_size = min(ids.size(), pair_ids.size());
        int extra_size = larger_size - smaller_size;
        int size_diff = max(num_tokens_to_remove - extra_size, 0) / 2;
        int final_size = max(larger_size - num_tokens_to_remove - size_diff, 0);
        int short_size = max(smaller_size - size_diff, 0);

        if (ids.size() >= pair_ids.size())
        {
            if (_truncation_side == TruncationSide::RIGHT)
            {
                return pair(
                    vector<int unsigned>(ids.begin(), ids.begin() + final_size),
                    vector<int unsigned>(pair_ids.begin(), pair_ids.begin() + short_size));
            }
            else if (_truncation_side == TruncationSide::LEFT)
            {
                return pair(
                    vector<int unsigned>(ids.begin() + (ids.size() - final_size), ids.end()),
                    vector<int unsigned>(pair_ids.begin() + (pair_ids.size() - short_size), pair_ids.end()));
            }
        }
        else
        {
            if (_truncation_side == TruncationSide::RIGHT)
            {
                return pair(
                    vector<int unsigned>(ids.begin(), ids.begin() + short_size),
                    vector<int unsigned>(pair_ids.begin(), pair_ids.begin() + final_size));
            }
            else if (_truncation_side == TruncationSide::LEFT)
            {
                return pair(
                    vector<int unsigned>(ids.begin() + (ids.size() - short_size), ids.end()), vector<int unsigned>(pair_ids.begin() + (pair_ids.size() - final_size), pair_ids.end()));
            }
        }
        return pair(ids, pair_ids);
    };

    pair<vector<unsigned int>, vector<unsigned int>> basic_callback_pair(
        vector<unsigned int> &encoding, vector<unsigned int> &encoding_pair)
    {
        vector<unsigned int> v(encoding.begin(), encoding.end());
        v.insert(v.end(), encoding_pair.begin(), encoding_pair.end());
        v.emplace_back(rules_cache[" "]);
        vector<unsigned int> token_type(encoding.size(), 0);
        token_type.insert(token_type.end(), encoding_pair.size(), 1);
        return pair(v, token_type);
    }

    vector<unsigned int> build_special_token_mask(
        vector<unsigned int> &encodings)
    {
        vector<unsigned int> mask(encodings.size());
        for (unsigned int i = 0; i < encodings.size(); i++)
        {
            if (_special_token_ids.find(encodings.at(i)) != _special_token_ids.end())
            {
                mask.at(i) = 1;
            }
        }
        return mask;
    }

    void build_masks(
        unordered_map<string, vector<vector<unsigned int>>> *results,
        bool return_attention_mask,
        bool return_special_tokens_mask,
        bool return_token_type_ids,
        unsigned int stride = 0)
    {
        if (return_special_tokens_mask)
        {
            results->emplace("special_tokens_mask", vector<vector<unsigned int>>(results->at("input_ids").size()));
            tbb::parallel_for(
                tbb::blocked_range<int unsigned>(
                    0,
                    results->at("input_ids").size()),
                [&](tbb::blocked_range<int unsigned> r)
                {
                    for (int unsigned i = r.begin(); i < r.end(); ++i)
                    {
                        results->at("special_tokens_mask").at(i) = build_special_token_mask(
                            results->at("input_ids").at(i));
                    }
                });
        }

        // if do not pad, no masking returned
        // padding strategy longest/max_length or do not pad
        if (_padding_strategy != PaddingStrategy::DO_NOT_PAD)
        {
            if (return_attention_mask)
            {
                results->emplace(
                    "attention_mask",
                    vector<vector<unsigned int>>(results->at("input_ids").size()));
            }

            // either MAX_LENGTH or LONGEST
            unsigned int actual_length = _max_length;
            if (_padding_strategy != PaddingStrategy::MAX_LENGTH)
            {
                // then strategy is longest
                actual_length = 0;
                for (auto &r : results->at("input_ids"))
                {
                    actual_length = r.size() > actual_length ? r.size() : actual_length;
                }
            }

            if (actual_length % _pad_to_multiple_of != 0)
            {
                actual_length = ((actual_length / _pad_to_multiple_of) + 1) * _pad_to_multiple_of;
            }

            tbb::parallel_for(
                tbb::blocked_range<int unsigned>(0, results->at("input_ids").size()),
                [&](tbb::blocked_range<int unsigned> r)
                {
                    for (int unsigned i = r.begin(); i < r.end(); ++i)
                    {
                        unsigned int num_pads = actual_length < results->at("input_ids").at(i).size() ? 0 : actual_length - results->at("input_ids").at(i).size();
                        if (return_attention_mask)
                        {

                            results->at("attention_mask").at(i) = vector<unsigned int>(results->at("input_ids").at(i).size(), 1);

                            if (_padding_side == PaddingSide::RIGHT)
                            {
                                results->at("attention_mask").at(i).insert(results->at("attention_mask").at(i).end(), num_pads, 0);
                            }
                            else
                            {
                                results->at("attention_mask").at(i).insert(results->at("attention_mask").at(i).begin(), num_pads, 0);
                            }
                        }

                        if (num_pads == 0)
                        {
                            continue;
                        }

                        if (return_special_tokens_mask)
                        {
                            if (_padding_side == PaddingSide::RIGHT)
                            {
                                results->at("special_tokens_mask").at(i).insert(results->at("special_tokens_mask").at(i).end(), num_pads, 1);
                            }
                            else
                            {
                                results->at("special_tokens_mask").at(i).insert(results->at("special_tokens_mask").at(i).begin(), num_pads, 1);
                            }
                        }

                        if (return_token_type_ids && results->find("token_type_ids") != results->end())
                        {
                            if (_padding_side == PaddingSide::RIGHT)
                            {
                                results->at("token_type_ids").at(i).insert(results->at("token_type_ids").at(i).end(), num_pads, _special_tokens_map["pad_token"]);
                            }
                            else
                            {
                                results->at("token_type_ids").at(i).insert(results->at("token_type_ids").at(i).begin(), num_pads, _special_tokens_map["pad_token"]);
                            }
                        }
                        pad(
                            &results->at("input_ids").at(i),
                            num_pads,
                            _special_tokens_map["pad_token"]);
                    }
                });
        }
    }

    void batch_transform_pair(
        unordered_map<string, vector<vector<unsigned int>>> *results,
        vector<vector<unsigned int>> &encodings,
        vector<vector<unsigned int>> &encoding_pairs,
        bool return_token_type_ids,
        function<pair<vector<unsigned int>, vector<unsigned int>>(vector<unsigned int> &encoding, vector<unsigned int> &encoding_pair)> f = NULL)
    {
        results->emplace(
            "input_ids",
            vector<vector<unsigned int>>(encodings.size()));
        if (return_token_type_ids)
        {
            results->emplace(
                "token_type_ids",
                vector<vector<unsigned int>>(encodings.size()));
        }
        if (f == NULL)
        {
            for (unsigned int i = 0; i < encodings.size(); ++i)
            {

                pair<vector<unsigned int>, vector<unsigned int>> temp = basic_callback_pair(
                    encodings.at(i),
                    encoding_pairs.at(i));
                results->at("input_ids").at(i) = temp.first;
                if (return_token_type_ids)
                {
                    results->at("token_type_ids").at(i) = temp.second;
                }
            }
        }
        else
        {
            for (unsigned int i = 0; i < encodings.size(); ++i)
            {
                pair<vector<unsigned int>, vector<unsigned int>> temp = f(encodings.at(i), encoding_pairs.at(i));
                results->at("input_ids").at(i) = temp.first;
                if (return_token_type_ids)
                {
                    results->at("token_type_ids").at(i) = temp.second;
                }
            }
        }
    }

    void truncate_pairs(
        unordered_map<string, vector<vector<unsigned int>>> *results,
        vector<vector<unsigned int>> &encodings,
        vector<vector<unsigned int>> &encoding_pairs,
        bool return_token_type_ids,
        std::function<pair<vector<unsigned int>, vector<unsigned int>>(vector<unsigned int> &encoding, vector<unsigned int> &encoding_pair)> f = NULL)
    {
        // for pairs, four possible strategies
        if (_truncation_strategy != TruncationStrategy::DO_NOT_TRUNCATE)
        {
            vector<vector<unsigned int>> truncated_a(encodings.size());
            vector<vector<unsigned int>> truncated_b(encodings.size());
            if (_truncation_strategy == TruncationStrategy::LONGEST_FIRST)
            {

                tbb::parallel_for(
                    tbb::blocked_range<int unsigned>(0, encodings.size()),
                    [&](tbb::blocked_range<int unsigned> r)
                    {
                        for (int unsigned i = r.begin(); i < r.end(); ++i)
                        {
                            unsigned int combined_size = encodings.at(i).size() + encoding_pairs.at(i).size();
                            unsigned int to_remove = combined_size > _max_length ? combined_size - _max_length : 0;

                            auto temp = truncate_pair_longest(
                                encodings.at(i),
                                encoding_pairs.at(i),
                                to_remove);
                            truncated_a.at(i) = temp.first;
                            truncated_b.at(i) = temp.second;
                        }
                    });
            }
            else if (_truncation_strategy == TruncationStrategy::ONLY_FIRST)
            {
                tbb::parallel_for(
                    tbb::blocked_range<int unsigned>(0, encodings.size()),
                    [&](tbb::blocked_range<int unsigned> r)
                    {
                        for (int unsigned i = r.begin(); i < r.end(); ++i)
                        {
                            unsigned int combined_size = encodings.at(i).size() + encoding_pairs.at(i).size();
                            unsigned int to_remove = combined_size > _max_length ? combined_size - _max_length : 0;

                            truncated_a.at(i) = truncate_sequence(
                                encodings.at(i),
                                to_remove,
                                0);
                        }
                    });
                truncated_b = encoding_pairs;
            }
            else if (_truncation_strategy == TruncationStrategy::ONLY_SECOND)
            {
                tbb::parallel_for(
                    tbb::blocked_range<int unsigned>(0, encodings.size()),
                    [&](tbb::blocked_range<int unsigned> r)
                    {
                        for (int unsigned i = r.begin(); i < r.end(); ++i)
                        {
                            unsigned int combined_size = encodings.at(i).size() + encoding_pairs.at(i).size();
                            unsigned int to_remove = combined_size > _max_length ? combined_size - _max_length : 0;

                            truncated_b.at(i) = truncate_sequence(
                                encoding_pairs.at(i),
                                to_remove,
                                0);
                        }
                    });
                truncated_a = encodings;
            } // else no truncation
            batch_transform_pair(results, truncated_a, truncated_b, return_token_type_ids, f);
        }
        else
        { // did not truncate
            batch_transform_pair(results, encodings, encoding_pairs, return_token_type_ids, f);
        }
    }

    unordered_map<string, vector<vector<unsigned int>>>
    batch_encode_pairs(
        const vector<string> &texts,
        const vector<string> &text_pairs,
        bool return_attention_mask,
        bool return_special_tokens_mask,
        bool return_token_type_ids,
        unsigned int stride = 0,
        std::function<pair<vector<unsigned int>, vector<unsigned int>>(vector<unsigned int> &encoding, vector<unsigned int> &encoding_pair)> f = NULL)
    {
        unordered_map<string, vector<vector<unsigned int>>> results;
        vector<vector<unsigned int>> encodings = batch_tokenize_whole(texts);
        vector<vector<unsigned int>> encoding_pairs = batch_tokenize_whole(text_pairs);

        truncate_pairs(&results, encodings, encoding_pairs, return_token_type_ids, f);

        build_masks(
            &results,
            return_attention_mask,
            return_special_tokens_mask,
            return_token_type_ids,
            stride);

        return results;
    }

    unordered_map<string, vector<vector<unsigned int>>>
    batch_encode_pairs_presplit(
        const vector<vector<string>> &texts,
        const vector<vector<string>> &text_pairs,
        bool return_attention_mask,
        bool return_special_tokens_mask,
        bool return_token_type_ids,
        unsigned int stride = 0,
        std::function<pair<vector<unsigned int>, vector<unsigned int>>(vector<unsigned int> &encoding, vector<unsigned int> &encoding_pair)> f = NULL)
    {
        unordered_map<string, vector<vector<unsigned int>>> results;
        vector<vector<unsigned int>> encodings = batch_tokenize_presplit(texts);
        vector<vector<unsigned int>> encoding_pairs = batch_tokenize_presplit(text_pairs);

        truncate_pairs(&results, encodings, encoding_pairs, return_token_type_ids, f);

        build_masks(
            &results,
            return_attention_mask,
            return_special_tokens_mask,
            return_token_type_ids,
            stride);

        return results;
    }

    void batch_transform(
        unordered_map<string, vector<vector<unsigned int>>> *results,
        vector<vector<unsigned int>> &encodings,
        std::function<vector<unsigned int>(vector<unsigned int> &encoding)> f = NULL)
    {
        if (f != NULL)
        {
            results->emplace(
                "input_ids", vector<vector<unsigned int>>(0));
            results->at("input_ids").reserve(encodings.size());
            for (auto &e : encodings)
            {
                results->at("input_ids").emplace_back(f(e));
            }
        }
        else
        {
            results->emplace("input_ids", encodings);
        }
    }

    void truncate(
        unordered_map<string, vector<vector<unsigned int>>> *results,
        vector<vector<unsigned int>> &encodings,
        unsigned int stride,
        bool return_overflowing_tokens,
        const std::function<vector<unsigned int>(vector<unsigned int> &encodings)> f = NULL)
    {
        if (_truncation_strategy != TruncationStrategy::DO_NOT_TRUNCATE)
        {
            vector<vector<unsigned int>> truncated(encodings.size());
            vector<vector<unsigned int>> overflow(encodings.size());

            tbb::parallel_for(
                tbb::blocked_range<int unsigned>(0, encodings.size()),
                [&](tbb::blocked_range<int unsigned> r)
                {
                    for (int unsigned i = r.begin(); i < r.end(); ++i)
                    {
                        unsigned int to_remove = encodings.at(i).size() > _max_length ? encodings.at(i).size() - _max_length : 0;
                        if (to_remove == 0)
                        {
                            truncated.at(i) = encodings.at(i);
                        }
                        else if (return_overflowing_tokens)
                        {
                            auto temp = truncate_sequence_get_overflow(
                                encodings.at(i),
                                to_remove,
                                stride);
                            truncated.at(i) = temp.first;
                            overflow.at(i) = temp.second;
                        }
                        else
                        {
                            truncated.at(i) = truncate_sequence(
                                encodings.at(i),
                                to_remove,
                                stride);
                        }
                    }
                });
            batch_transform(results, truncated, f);
            if (return_overflowing_tokens)
            {
                results->emplace("overflowing_tokens", overflow);
            }
        }
        else
        {
            batch_transform(results, encodings, f);
        }
    }

    unordered_map<string, vector<vector<unsigned int>>>
    batch_encode(
        const vector<string> &texts,
        bool return_attention_mask,
        bool return_overflowing_tokens,
        bool return_special_tokens_mask,
        unsigned int stride = 0,
        const std::function<vector<unsigned int>(vector<unsigned int> &encodings)> f = NULL)
    {
        vector<vector<unsigned int>> encodings = batch_tokenize_whole(texts);
        unordered_map<string, vector<vector<unsigned int>>> results;

        // default either do not truncate or longest first for singles
        // no return overflow for pairs

        truncate(&results, encodings, stride, return_overflowing_tokens, f);

        build_masks(
            &results,
            return_attention_mask,
            return_special_tokens_mask,
            false,
            stride);

        return results;
    }

    unordered_map<string, vector<vector<unsigned int>>> batch_encode_presplit(
        const vector<vector<string>> &texts,
        bool return_attention_mask,
        bool return_overflowing_tokens,
        bool return_special_tokens_mask,
        unsigned int stride = 0,
        const std::function<vector<unsigned int>(vector<unsigned int> &encodings)> f = NULL)
    {
        vector<vector<unsigned int>> encodings = batch_tokenize_presplit(texts);
        unordered_map<string, vector<vector<unsigned int>>> results;

        // default either do not truncate or longest first for singles
        // no return overflow for pairs
        truncate(&results, encodings, stride, return_overflowing_tokens, f);

        build_masks(
            &results,
            return_attention_mask,
            return_special_tokens_mask,
            false,
            stride);

        return results;
    }
};

class PyGreedyTokenizer : public GreedyTokenizer
{
public:
    using GreedyTokenizer::batch_encode;
    using GreedyTokenizer::batch_encode_pairs;
    using GreedyTokenizer::batch_encode_pairs_presplit;
    using GreedyTokenizer::batch_encode_presplit;
    using GreedyTokenizer::batch_split_and_tokenize;
    using GreedyTokenizer::batch_tokenize_presplit;
    using GreedyTokenizer::batch_tokenize_whole;
    using GreedyTokenizer::batch_transform;
    using GreedyTokenizer::batch_transform_pair;
    using GreedyTokenizer::build_masks;
    using GreedyTokenizer::build_special_token_mask;
    using GreedyTokenizer::get_rule;
    using GreedyTokenizer::get_rules_size;
    using GreedyTokenizer::get_special_token_ids;
    using GreedyTokenizer::GreedyTokenizer;
    using GreedyTokenizer::pad;
    using GreedyTokenizer::set_post_embedding_strategy;
    using GreedyTokenizer::set_regex_pattern;
    using GreedyTokenizer::tokenize_presplit;
    using GreedyTokenizer::tokenize_text;
    using GreedyTokenizer::truncate;
    using GreedyTokenizer::truncate_pair_longest;
    using GreedyTokenizer::truncate_pairs;
    using GreedyTokenizer::truncate_sequence;
    using GreedyTokenizer::truncate_sequence_get_overflow;
};

GreedyTokenizer *build(
    vector<string> rules,
    unordered_map<string, string> special_tokens = {})
{
    return new GreedyTokenizer(rules, special_tokens);
}

PYBIND11_MODULE(greedy_encoder, var)
{
    var.doc() = "greedy module";
    py::class_<GreedyTokenizer, PyGreedyTokenizer>(var, "GreedyTokenizer")
        .def(py::init<>([](vector<string> &cover_order,
                           unordered_map<string, string> &special_tokens)
                        { return new GreedyTokenizer(cover_order, special_tokens); }))
        .def("batch_encode", &GreedyTokenizer::batch_encode,
             py::arg("texts"),
             py::arg("return_attention_mask"),
             py::arg("return_overflowing_tokens"),
             py::arg("return_special_tokens_mask"),
             py::arg("stride"),
             py::arg("f"))
        .def("batch_encode_pairs", &GreedyTokenizer::batch_encode_pairs,
             py::arg("texts"),
             py::arg("text_pairs"),
             py::arg("return_attention_mask"),
             py::arg("return_special_tokens_mask"),
             py::arg("return_token_type_ids"),
             py::arg("stride"),
             py::arg("f"))
        .def("batch_encode_presplit", &GreedyTokenizer::batch_encode_presplit,
             py::arg("texts"),
             py::arg("return_attention_mask"),
             py::arg("return_overflowing_tokens"),
             py::arg("return_special_tokens_mask"),
             py::arg("stride"),
             py::arg("f"))
        .def("batch_encode_pairs_presplit", &GreedyTokenizer::batch_encode_pairs_presplit,
             py::arg("texts"),
             py::arg("text_pairs"),
             py::arg("return_attention_mask"),
             py::arg("return_special_tokens_mask"),
             py::arg("return_token_type_ids"),
             py::arg("stride"),
             py::arg("f"))
        .def("batch_tokenize_presplit", &GreedyTokenizer::batch_tokenize_presplit)
        .def("batch_tokenize_whole", &GreedyTokenizer::batch_tokenize_whole)
        .def("batch_split_and_tokenize", &GreedyTokenizer::batch_split_and_tokenize)
        .def("get_rule", &GreedyTokenizer::get_rule)
        .def("get_rules_size", &GreedyTokenizer::get_rules_size)
        .def("set_regex_pattern", &GreedyTokenizer::set_regex_pattern)
        .def("set_post_embedding_strategy", &GreedyTokenizer::set_post_embedding_strategy)
        .def("tokenize_text", &GreedyTokenizer::tokenize_text)
        .def("tokenize_presplit", &GreedyTokenizer::tokenize_presplit)
        .def("get_special_token_ids", &GreedyTokenizer::get_special_token_ids)
        .def("truncate_sequence_get_overflow", &GreedyTokenizer::truncate_sequence_get_overflow)
        .def("truncate_sequence", &GreedyTokenizer::truncate_sequence);
    py::enum_<TruncationStrategy>(var, "TruncationStrategy")
        .value("only_first", TruncationStrategy::ONLY_FIRST)
        .value("only_second", TruncationStrategy::ONLY_SECOND)
        .value("longest_first", TruncationStrategy::LONGEST_FIRST)
        .value("do_not_truncate", TruncationStrategy::DO_NOT_TRUNCATE);
    py::enum_<PaddingStrategy>(var, "PaddingStrategy")
        .value("longest", PaddingStrategy::LONGEST)
        .value("max_length", PaddingStrategy::MAX_LENGTH)
        .value("do_not_pad", PaddingStrategy::DO_NOT_PAD);
    py::enum_<TruncationSide>(var, "TruncationSide")
        .value("right", TruncationSide::RIGHT)
        .value("left", TruncationSide::LEFT);
    py::enum_<PaddingSide>(var, "PaddingSide")
        .value("right", PaddingSide::RIGHT)
        .value("left", PaddingSide::LEFT);
    var.def(
        "build",
        &build,
        py::arg("rules") = vector<string>(),
        py::arg("special_tokens") = unordered_map<string, string>(),
        "Factory function for greedy tokenizer, use this to encode text to tokens.");
}