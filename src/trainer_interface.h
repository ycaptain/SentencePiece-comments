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

#ifndef TRAINER_INTERFACE_H_
#define TRAINER_INTERFACE_H_

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common.h"
#include "sentencepiece_model.pb.h"
#include "sentencepiece_processor.h"
#include "util.h"

namespace sentencepiece {

// 对键值对向量按照值从大到小排序。
template <typename K, typename V>
std::vector<std::pair<K, V>> Sorted(const std::vector<std::pair<K, V>> &m) {
  std::vector<std::pair<K, V>> v = m;
  std::sort(v.begin(), v.end(),
            [](const std::pair<K, V> &p1, const std::pair<K, V> &p2) {
              return (p1.second > p2.second ||
                      (p1.second == p2.second && p1.first < p2.first));
            });
  return v;
}

template <typename K, typename V>
std::vector<std::pair<K, V>> Sorted(const std::unordered_map<K, V> &m) {
  std::vector<std::pair<K, V>> v(m.begin(), m.end());
  return Sorted(v);
}

// 训练器基类。
// Base trainer class
class TrainerInterface {
 public:
  using Sentence = std::pair<std::string, int64>;
  using Sentences = std::vector<Sentence>;

  static const char32 kWSChar;
  static const char32 kUNKChar;
  static const char32 kUPPBoundaryChar;
  static const char kWSStr[];
  static const char kUNKStr[];
  static const char kUPPBoundaryStr[];

  TrainerInterface(const TrainerSpec &trainer_spec,
                   const NormalizerSpec &normalizer_spec);

  virtual ~TrainerInterface();

  virtual util::Status Train() { return status(); }

  // 返回训练器当前的异常状态。
  virtual util::Status status() const { return status_; }

  FRIEND_TEST(TrainerInterfaceTest, IsValidSentencePieceTest);
  FRIEND_TEST(TrainerInterfaceTest, OverrideSpecialPiecesTest);
  FRIEND_TEST(TrainerInterfaceTest, SerializeTest);

 protected:
  // 返回给定的词语是否有效。结果与max_sentencepiece_length，
  // split_by_whiespace，split_by_unicode_script有关。
  // Returns true if |piece| is valid sentence piece.
  // The result is affected by
  // max_sentencepiece_length, split_by_whiespace, split_by_unicode_script.
  bool IsValidSentencePiece(const string_util::UnicodeText &piece) const;

  // 从输入加载全部的句子。最多加载input_sentence_size个句子。
  // Loads all sentences from spec.input().
  // It loads at most input_sentence_size sentences.
  util::Status LoadSentences();

  // 用空格分割所有的句子并把句子替换为标志过的字符串。
  // Splits all sentencecs by whitespaces and
  // replace the |sentences_| with tokenized string.
  // e.g.,
  //  [ ["hello world ", 1], ["hi world]" ] =>
  //  [ ["hello", 1], ["hi", 1], ["world", 2] ]
  void SplitSentencesByWhitespace();

  // 将模型和词语保存至文件。
  // Save model files into spec.model_prefix().
  util::Status Save() const;

  // 必须被final_vocab_包含的字符的键值表。值存储了出现的频率。
  // Set of characters which must be included in the final vocab.
  // The value of this map stores the frequency.
  std::unordered_map<char32, int64> required_chars_;

  // 最终输出的词语。
  // Final output pieces
  std::vector<std::pair<std::string, float>> final_pieces_;

  // 全部的句子。
  // All sentences.
  Sentences sentences_;

  // 训练器特性。
  // Trainer spec.
  TrainerSpec trainer_spec_;

  // 规范器特性。
  // Normalizer spec
  NormalizerSpec normalizer_spec_;

  // 保留的控制符表。
  // 以词语为键。
  // Reserved control pieces. e.g., <unk>, <s>, </s>.
  // key is vocab id.
  std::map<int, std::pair<std::string, ModelProto::SentencePiece::Type>>
      meta_pieces_;

  // 检测初始化过程中的异常。
  // Detect errors on initialization.
  util::Status status_;

 private:
  // 将final_pieces_序列化并写入model_proto。
  // Serialize final_pieces_ to |model_proto|.
  util::Status Serialize(ModelProto *model_proto) const;

  // 以当前的调试用模型将最好的句子分割保存至文件。
  // Saves the best sentence split with the current model for debugging.
  util::Status SaveSplits(absl::string_view filename) const;

  // 保存序列化的model_proto到指定的文件。
  // Saves model file.
  util::Status SaveModel(absl::string_view filename) const;

  // 保存model_proto中的所有词语和其分数到指定的文件。
  // Saves vocabulary file for NMT.
  util::Status SaveVocab(absl::string_view filename) const;

  // 从训练器特性初始化meta_pieces_。
  // Initializes `meta_pieces_` from TrainerSpec.
  util::Status InitMetaPieces();

  // 用于自测试的随机采样原始句子。
  // Randomly sampled raw sentences for self-testing.
  std::vector<std::string> self_test_samples_;
};
}  // namespace sentencepiece
#endif  // TRAINER_INTERFACE_H_
