// Copyright 2016 Google Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.!

#include "unigram_model_trainer.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <functional>
#include <memory>
#include <numeric>
#include <queue>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "normalizer.h"
#include "third_party/esaxx/esa.hxx"  // Suffix array library.
#include "unicode_script.h"
#include "util.h"

// DOC:命名空间 sentencepiece::unigram
namespace sentencepiece {
namespace unigram {
namespace {

// DOC:
//      计算传入参数的 Digamma 函数值
// 参数:
//      x -- 传入参数
// 返回:
//      Digamma(x) 近似值
double Digamma(double x) {
  double result = 0.0;
  for (; x < 7; ++x) result -= 1 / x;
  x -= 1.0 / 2.0;
  const double xx = 1.0 / x;
  const double xx2 = xx * xx;
  const double xx4 = xx2 * xx2;
  result += log(x) + (1.0 / 24.0) * xx2 - (7.0 / 960.0) * xx4 +
            (31.0 / 8064.0) * xx4 * xx2 - (127.0 / 30720.0) * xx4 * xx4;
  return result;
}

// DOC:
//      将概率转换为对数形式
// 参数:
//      begin -- 所需实现对数形式转换的起始位置
//      end -- 所需实现对数形式转换的终止位置
template <typename IT>
void ToLogProb(IT begin, IT end) {
  float sum = 0.0;
  for (auto it = begin; it != end; ++it) {
    sum += it->second;
  }
  float logsum = log(sum);
  for (auto it = begin; it != end; ++it) {
    it->second = log(it->second) - logsum;
  }
}
}  // namespace

// DOC:
//      对象创建
TrainerModel::TrainerModel(const TrainerSpec &trainer_spec,
                           const NormalizerSpec &normalizer_spec)
    : trainer_spec_(trainer_spec), normalizer_spec_(normalizer_spec) {}

TrainerModel::~TrainerModel() {}

// DOC:
//    返回完整的句段
// 注意:
//    只包含元信息 起始符/终止符等不包含在内
const TrainerModel::SentencePieces &TrainerModel::GetSentencePieces() const {
  return sentencepieces_;
}

// DOC:
//    对 UnigramModel 设置新的句段
// 注意:
//    只包含元信息 起始符/终止符等不包含在内
void TrainerModel::SetSentencePieces(SentencePieces &&sentencepieces) {
  sentencepieces_ = std::move(sentencepieces);
  CHECK(!sentencepieces_.empty());

  min_score_ = FLT_MAX;
  model_proto_data_.Clear();
  model_proto_ = &model_proto_data_;
  std::vector<std::pair<absl::string_view, int>> pieces;

  for (size_t i = 0; i < sentencepieces_.size(); ++i) {
    const absl::string_view w = sentencepieces_[i].first;  // piece
    const float score = sentencepieces_[i].second;         // score.
    CHECK(!std::isnan(score));
    pieces.emplace_back(w, i);
    min_score_ = std::min(min_score_, score);
    auto *piece = model_proto_data_.add_pieces();
    piece->set_piece(w.data(), w.size());
    piece->set_score(score);
  }

  BuildTrie(&pieces);
  CHECK_OK(status());
}

// DOC:
//      返回种子分词块 用作 EM (期望最大化) 训练
// Returns seed sentencepieces for EM training.
TrainerModel::SentencePieces Trainer::MakeSeedSentencePieces() const {
  CHECK(!sentences_.empty());
  CHECK(!required_chars_.empty());

  // 以 '0x0000' 为分隔符合并所有句子
  // Merges all sentences into one array with 0x0000 delimiter.
  std::vector<char32> array;
  std::unordered_map<std::string, int64> all_chars;
  constexpr char32 kSentenceBoundary = 0x0000;

  for (const auto &w : sentences_) {
    for (const auto &c : string_util::UTF8ToUnicodeText(w.first)) {
      array.push_back(c);
      if (c != kUNKChar && c != kSentenceBoundary) {
        all_chars[string_util::UnicodeCharToUTF8(c)] += w.second;
      }
    }
    // 在每个句话后加入 kSentenceBoundary 作为句段结束的标记
    array.push_back(kSentenceBoundary);  // sentence boundary marker.
  }

  // DOC:
  //    SA -- 前缀数组
  //    L -- 内部结点左边界(句段起始位置)记录数组
  //    R -- 内部结点右边界(句段终止位置)记录数组
  //    D -- 内部结点深度(句段长度)记录数组
  const int n = array.size();
  std::vector<int> SA(n);  // suffix array
  std::vector<int> L(n);   // left boundaries of internal node
  std::vector<int> R(n);   // right boundaries of internal node
  std::vector<int> D(n);   // depths of internal node

  // DOC:
  //    构造前缀数组 以提取正在处理的子串
  // Makes a suffix array to extract all sub strings occurring
  // more than 2 times in the sentence.
  constexpr int kAlphabetSize = 0x110000;  // All UCS4 range.
  int node_num = 0;
  LOG(INFO) << "Making suffix array...";
  CHECK_EQ(0, esaxx(array.begin(), SA.begin(), L.begin(), R.begin(), D.begin(),
                    n, kAlphabetSize, node_num));

  LOG(INFO) << "Extracting frequent sub strings...";
  std::vector<std::pair<int, int>> substr_index;
  for (int i = 0; i < node_num; ++i) {
    const int offset = SA[L[i]];
    const int len = D[i];
    if (len <= 1) {
      continue;
    }
    const char32 *begin = &array[0] + offset;
    const char32 *end = &array[0] + offset + len;
    // 若子串包含句段分隔符 则处理完毕
    // Skips if a substring contains a sentence boundary.
    if (std::find(begin, end, kSentenceBoundary) != end) {
      continue;
    }
    const UnicodeText uw(begin, end);
    if (!IsValidSentencePiece(uw)) {
      continue;
    }

    // 单词层面的分割采用默认的score = freq * len
    // character-wise coverage is the default score.
    const int freq = R[i] - L[i];
    const int score = freq * len;
    substr_index.emplace_back(i, score);
  }

  // all_chars 必须包含在种子分词块中
  // all_chars must be included in the seed sentencepieces.
  TrainerModel::SentencePieces seed_sentencepieces;
  for (const auto &it : Sorted(all_chars)) {
    seed_sentencepieces.emplace_back(it);
  }

  // 对子串分割进行排序
  // Sort by the coverage of sub strings.
  for (const auto &p : Sorted(substr_index)) {
    const int offset = SA[L[p.first]];
    const int len = D[p.first];
    CHECK_GT(len, 0);
    const char32 *begin = &array[offset];
    const char32 *end = &array[offset + len];
    const UnicodeText uw(begin, end);
    CHECK(IsValidSentencePiece(uw));  // just in case.
    const std::string w = string_util::UnicodeTextToUTF8(uw);
    if (seed_sentencepieces.size() ==
        static_cast<size_t>(trainer_spec_.seed_sentencepiece_size())) {
      break;
    }
    CHECK(!port::ContainsKey(all_chars, w));
    seed_sentencepieces.emplace_back(w, p.second);
  }

  ToLogProb(seed_sentencepieces.begin(), seed_sentencepieces.end());

  LOG(INFO) << "Initialized " << seed_sentencepieces.size()
            << " seed sentencepieces";

  return seed_sentencepieces;
}

// DOC:
//    进行 EM (期望最大化) 算法的 E 步 -- 求期望
//
// 参数:
//    model -- 训练模型的引用
//    objective -- 当前模型的负概率值
//    num_tokens -- 训练集中的分词块总数
//
// 返回:
//    返回期望值数组
std::vector<float> Trainer::RunEStep(const TrainerModel &model, float *obj,
                                     int64 *num_tokens) const {
  std::vector<std::vector<float>> expected(trainer_spec_.num_threads());
  std::vector<float> objs(trainer_spec_.num_threads(), 0.0);
  std::vector<int64> ntokens(trainer_spec_.num_threads(), 0.0);

  auto pool = port::MakeUnique<thread::ThreadPool>();

  int64 all_sentence_freq = 0;
  for (const auto &w : sentences_) {
    all_sentence_freq += w.second;
  }

  // 同时求取期望值
  // Executes E step in parallel
  for (int n = 0; n < trainer_spec_.num_threads(); ++n) {
    pool->Schedule([&, n]() {
      Lattice lattice;
      expected[n].resize(model.GetPieceSize(), 0.0);
      for (size_t i = n; i < sentences_.size();
           i += trainer_spec_.num_threads()) {
        const std::string &w = sentences_[i].first;
        const int64 freq = sentences_[i].second;
        lattice.SetSentence(w);
        model.PopulateNodes(&lattice);
        const float Z = lattice.PopulateMarginal(freq, &expected[n]);
        ntokens[n] += lattice.Viterbi().size();
        CHECK(!std::isnan(Z))
            << "likelihood is NAN. Input sentence may be too long";
        objs[n] -= Z / all_sentence_freq;
      }
    });
  }
  pool.reset(nullptr);

  // 进行期望值的叠加
  // Merges expectations
  for (int n = 1; n < trainer_spec_.num_threads(); ++n) {
    objs[0] += objs[n];
    ntokens[0] += ntokens[n];
    for (size_t k = 0; k < expected[0].size(); ++k) {
      expected[0][k] += expected[n][k];
    }
  }

  *obj = objs[0];
  *num_tokens = ntokens[0];
  CHECK(!std::isnan(*obj));

  return expected[0];
}

// DOC:
//    进行 EM (期望最大化) 算法的 M 步 -- 求极大
//
// 参数:
//    model -- 训练模型引用
//    expected -- 经 E 步处理所得的期望值数组引用
//
// 返回:
//    返回处理后新的分词块数组
TrainerModel::SentencePieces Trainer::RunMStep(
    const TrainerModel &model, const std::vector<float> &expected) const {
  const auto &sentencepieces = model.GetSentencePieces();
  CHECK_EQ(sentencepieces.size(), expected.size());
  TrainerModel::SentencePieces new_sentencepieces;

  float sum = 0.0;
  for (size_t i = 0; i < expected.size(); ++i) {
    const float freq = expected[i];

    // Filter infrequent sentencepieces here.
    constexpr float kExpectedFrequencyThreshold = 0.5;
    if (freq < kExpectedFrequencyThreshold) {
      continue;
    }

    new_sentencepieces.emplace_back(sentencepieces[i].first, freq);
    sum += freq;
  }

  // DOC:
  //    实现EM没有采用原始的 EM 算法
  //    而是使用 Bayesianified/DPified EM
  //    此算法可以进行稀疏型先验
  // Here we do not use the original EM, but use the
  // Bayesianified/DPified EM algorithm.
  // https://cs.stanford.edu/~pliang/papers/tutorial-acl2007-talk.pdf
  // This modification will act as a sparse prior.
  const float logsum = Digamma(sum);
  for (auto &w : new_sentencepieces) {
    w.second = Digamma(w.second) - logsum;
  }

  return new_sentencepieces;
}

// DOC:
//    每次进行 EM (期望最大化) 子迭代后 对当前全部分词块进行剪枝
//    剪去冗余部分 提高算法效率
TrainerModel::SentencePieces Trainer::PruneSentencePieces(
    const TrainerModel &model) const {
  const auto &sentencepieces = model.GetSentencePieces();

  Lattice lattice;
  std::vector<bool> always_keep(sentencepieces.size(), true);
  std::vector<std::vector<int>> alternatives(sentencepieces.size());

  // DOC:
  //    处理 "在不选择当前分词块的条件下 概率值次高的路径" 问题
  //    并将处理结果保存在 sentencepiece[i].alternatives[i] 中
  // First, segments the current sentencepieces to know
  // how each sentencepiece is resegmented if this sentencepiece is removed
  // from the vocabulary.
  // To do so, we take the second best segmentation of sentencepiece[i].
  // alternatives[i] stores the sequence of second best sentencepieces.
  for (size_t i = 0; i < sentencepieces.size(); ++i) {
    const auto &w = sentencepieces[i];
    lattice.SetSentence(w.first);
    model.PopulateNodes(&lattice);
    const auto nbests = lattice.NBest(2);
    if (nbests.size() == 1) {
      // 如果无法找到次好结果 则总保留当前结点
      // No second-best result is found. always keep this sentencepiece.
      always_keep[i] = true;
      continue;
    } else if (nbests[0].size() >= 2) {
      // Can safely remove this sentencepiece if its Viterbi path is split.
      always_keep[i] = false;
    } else if (nbests[0].size() == 1) {
      always_keep[i] = true;
      for (const auto *node : nbests[1]) {
        alternatives[i].push_back(node->id);
      }
    }
  }

  // DOC:
  //    计算 unigram 语言模型下所有分词块的概率
  //    invert[i] -- 保存 sentencepieces[i] 出现的位置
  // Second, segments all sentences to compute likelihood
  // with a unigram language model. inverted[i] stores
  // the set of sentence index where the sentencepieces[i] appears.
  float vsum = 0.0;
  std::vector<float> freq(sentencepieces.size(), 0.0);
  std::vector<std::vector<int>> inverted(sentencepieces.size());
  {
    std::vector<float> vsums(trainer_spec_.num_threads(), 0.0);
    std::vector<std::vector<float>> freqs(trainer_spec_.num_threads());
    std::vector<std::vector<std::vector<int>>> inverteds(
        trainer_spec_.num_threads());

    auto pool = port::MakeUnique<thread::ThreadPool>();
    for (int n = 0; n < trainer_spec_.num_threads(); ++n) {
      freqs[n].resize(sentencepieces.size(), 0.0);
      inverteds[n].resize(sentencepieces.size());

      pool->Schedule([&, n]() {
        Lattice lattice;
        for (size_t i = n; i < sentences_.size();
             i += trainer_spec_.num_threads()) {
          const auto &w = sentences_[i];
          lattice.SetSentence(w.first);
          model.PopulateNodes(&lattice);
          vsums[n] += w.second;
          for (const auto *node : lattice.Viterbi()) {
            if (node->id >= 0) {
              freqs[n][node->id] += w.second;
              inverteds[n][node->id].push_back(i);
            }
          }
        }
      });
    }
    pool.reset(nullptr);

    for (int n = 0; n < trainer_spec_.num_threads(); ++n) {
      vsum += vsums[n];
      for (size_t i = 0; i < sentencepieces.size(); ++i) {
        freq[i] += freqs[n][i];
        std::copy(inverteds[n][i].begin(), inverteds[n][i].end(),
                  std::back_inserter(inverted[i]));
      }
    }
  }

  const float sum = std::accumulate(freq.begin(), freq.end(), 0.0);
  const float logsum = log(sum);
  std::vector<std::pair<int, float>> candidates;
  TrainerModel::SentencePieces new_sentencepieces;

  // DOC:
  //    计算如果 sentencepiece[i] 被移除后 LM (最速下降法) 的概率值
  //    由于准确计算损失较为困难 则采用以 alternatives[i] 代替 sentencepiece[i] 的方法估算损失值
  // Finally, computes how likely the LM likelihood is reduced if
  // the sentencepiece[i] is removed from the vocabulary.
  // Since the exact computation of loss is difficult, we compute the
  // loss approximately by assuming that all sentencepiece[i] in the sentences
  // are replaced with alternatives[i] when sentencepiece[i] is removed.
  for (size_t i = 0; i < sentencepieces.size(); ++i) {
    if (freq[i] == 0 || !always_keep[i]) {
      // 如果该分词块不出现在 Viterbi 路径中 则可将其安全移除
      // not found in Viterbi path. Can remove this entry safely.
      continue;
    } else if (alternatives[i].empty()) {
      // 如果该分词块不存在可替换分词块 则保留该分词块
      // no alternatives. Keeps this entry.
      new_sentencepieces.push_back(sentencepieces[i]);
    } else {
      float F = 0.0;  // the frequency of sentencepieces[i].
      for (const int n : inverted[i]) {
        F += sentences_[n].second;
      }
      F /= vsum;  // normalizes by all sentence frequency.

      // 保存 sentencepiece[i] 的概率对数值
      // The logprob with the sentencepiece[i].
      const float logprob_sp = log(freq[i]) - logsum;

      // 在移除 sentencepiece[i] 后 其 freq[i] 根据 alternatives 重新计算
      // After removing the sentencepiece[i], its frequency freq[i] is
      // re-assigned to alternatives.
      // new_sum = current_sum - freq[i] + freq[i] * alternatives.size()
      //         = current_sum + freq[i] (alternatives - 1)
      const float logsum_alt = log(sum + freq[i] * (alternatives.size() - 1));

      // The frequencies of altenatives are increased by freq[i].
      float logprob_alt = 0.0;
      for (const int n : alternatives[i]) {
        logprob_alt += (log(freq[n] + freq[i]) - logsum_alt);
      }

      // loss: the diff of likelihood after removing the sentencepieces[i].
      const float loss = F * (logprob_sp - logprob_alt);
      candidates.emplace_back(i, loss);
    }
  }

  const int pruned_size =
      std::max<int>(desired_vocab_size_,
                    trainer_spec_.shrinking_factor() * sentencepieces.size());

  // 保持新分词数组的大小是原分词数组大小 * shrinking_factor
  // shrinking_factor 默认值为 0.75
  // Keeps trainer_spec_.shrinking_factor * sentencepieces.size() pieces.
  // shrinking_factor is 0.75 by default.
  for (const auto &w : Sorted(candidates)) {
    if (new_sentencepieces.size() == static_cast<size_t>(pruned_size)) {
      break;
    }
    new_sentencepieces.emplace_back(sentencepieces[w.first]);
  }

  return new_sentencepieces;
}

// DOC:
//    通过对必要词块/控制字符/用户自定义词块的处理 决定最终分词块
// 参数:
//    model -- 训练模型的引用
TrainerModel::SentencePieces Trainer::FinalizeSentencePieces(
    const TrainerModel &model) const {
  const auto &sentencepieces = model.GetSentencePieces();
  std::unordered_map<std::string, float> final_sentencepieces;
  std::unordered_map<std::string, float> sp(sentencepieces.begin(),
                                            sentencepieces.end());

  // 必要单词必须加入到最终分词块中
  // required_chars_ must be included in the final sentencepieces.
  float min_score_penalty = 0.0;
  constexpr float kMinScorePenaltyDelta = 0.0001;
  for (const auto &w : Sorted(required_chars_)) {
    const std::string s = string_util::UnicodeCharToUTF8(w.first);
    if (port::ContainsKey(sp, s)) {
      final_sentencepieces[s] = sp[s];
    } else {
      // Add penalty to avoid required pieces from having the same score.
      // Since the required_chars_ is sorted, frequent pieces have
      // less penalties.
      final_sentencepieces[s] = model.min_score() + min_score_penalty;
      min_score_penalty += kMinScorePenaltyDelta;
    }
  }

  const int vocab_size_size = trainer_spec_.vocab_size() - meta_pieces_.size();
  CHECK_GT(vocab_size_size, 0);

  // 保持所选分词块有更高的 score
  // Then keeps sentencepieces with higher scores.
  for (const auto &w : Sorted(sentencepieces)) {
    if (port::ContainsKey(final_sentencepieces, w.first)) {
      continue;
    }
    if (static_cast<size_t>(vocab_size_size) == final_sentencepieces.size()) {
      break;
    }
    final_sentencepieces[w.first] = w.second;
  }

  return Sorted(final_sentencepieces);
}

util::Status Trainer::Train() {
  RETURN_IF_ERROR(status());

  CHECK_EQ_OR_RETURN(TrainerSpec::UNIGRAM, trainer_spec_.model_type());
  CHECK_OR_RETURN(normalizer_spec_.escape_whitespaces());

  TrainerModel model(trainer_spec_, normalizer_spec_);

  RETURN_IF_ERROR(model.status());
  RETURN_IF_ERROR(LoadSentences());

  auto seed_sentencepieces = MakeSeedSentencePieces();
  model.SetSentencePieces(std::move(seed_sentencepieces));

  if (trainer_spec_.split_by_whitespace()) {
    SplitSentencesByWhitespace();
  }

  LOG(INFO) << "Using " << sentences_.size() << " sentences for EM training";

  desired_vocab_size_ = static_cast<size_t>(trainer_spec_.vocab_size() * 1.1);

  while (true) {
    // 子 EM 的迭代过程
    // Sub-EM iteration.
    for (int iter = 0; iter < trainer_spec_.num_sub_iterations(); ++iter) {
      // 进行E步
      // Executes E step
      float objective = 0.0;
      int64 num_tokens = 0;
      const auto expected = RunEStep(model, &objective, &num_tokens);

      // 进行 M 步
      // Executes M step.
      auto new_sentencepieces = RunMStep(model, expected);
      model.SetSentencePieces(std::move(new_sentencepieces));

      LOG(INFO) << "EM sub_iter=" << iter << " size=" << model.GetPieceSize()
                << " obj=" << objective << " num_tokens=" << num_tokens
                << " num_tokens/piece="
                << 1.0 * num_tokens / model.GetPieceSize();
    }  // end of Sub EM iteration
    // 子 EM 迭代过程结束

    // DOC:
    // 若句段规模达到期望规模 则停止迭代
    // Stops the iteration when the size of sentences reaches to the
    // desired symbol size.
    if (model.GetPieceSize() <= desired_vocab_size_) {
      break;
    }

    // 对模型进行剪枝
    // Prunes pieces.
    auto new_sentencepieces = PruneSentencePieces(model);
    model.SetSentencePieces(std::move(new_sentencepieces));
  }  // end of EM iteration
  // 整个EM迭代过程结束

  // 最后调整分词数量与词库要求一致
  // Finally, adjusts the size of sentencepices to be |vocab_size|.
  final_pieces_ = FinalizeSentencePieces(model);

  return Save();
}
}  // namespace unigram
}  // namespace sentencepiece
