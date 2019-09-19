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

#ifndef MODEL_INTERFACE_H_
#define MODEL_INTERFACE_H_

#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common.h"
#include "normalizer.h"
#include "sentencepiece_model.pb.h"
#include "sentencepiece_processor.h"
#include "third_party/absl/strings/string_view.h"
#include "third_party/darts_clone/darts.h"
#include "util.h"

namespace sentencepiece {

// 将给定的文本按词分割。
// 参数：
//       add_ws_as_suffix -- 是否将空格视为后缀，否则视为前缀。
// "_this_is_a_pen" => ["_this", "_is", "_a", "_pen"]
std::vector<absl::string_view> SplitIntoWords(absl::string_view text,
                                              bool add_ws_as_suffix = false);

using EncodeResult = std::vector<std::pair<absl::string_view, int>>;
using NBestEncodeResult = std::vector<std::pair<EncodeResult, float>>;

class ModelProto;

// 底层的模型接口。
// 给定一个规范化的字符串，返回一个用id标识的句子分词序列。
// Underlying model interface.
// Given a normalized string, returns a sequence of sentence pieces with ids.
class ModelInterface {
 public:
  using PieceToIdMap =
      std::unordered_map<absl::string_view, int, string_util::string_view_hash>;

  // unknown 代表未知字的符号
  absl::string_view unk_piece() const;
  absl::string_view bos_piece() const;
  absl::string_view eos_piece() const;
  absl::string_view pad_piece() const;

  // 直到该对象被销毁前，model_proto不能销毁。
  // `model_proto` should not be deleted until ModelInterface is destroyed.
  explicit ModelInterface(const ModelProto &model_proto);
  ModelInterface() {}

  virtual ~ModelInterface();

  // 返回当前状态。
  // 只有状态为OK时Encode/Decode函数才有效。
  // Returns Status.
  // Encode/Decode functions are valid only when status is OK.
  virtual util::Status status() const { return status_; }

  virtual const ModelProto &model_proto() const { return *model_proto_; }

  virtual const normalizer::PrefixMatcher *prefix_matcher() const {
    return matcher_.get();
  }

  // 给定一个规范化的字符串，返回一个用id标识的句子分词序列。
  // Given a normalized string, returns a sequence of sentence pieces with ids.
  // The concatenation of pieces must be the same as `normalized`.
  virtual EncodeResult Encode(absl::string_view normalized) const = 0;

  // 给定一个规范化的字符串，返回一个用id标识的句子分词序列和分数。
  // The same as above, but returns nbest result with score.
  virtual NBestEncodeResult NBestEncode(absl::string_view normalized,
                                        int nbest_size) const {
    LOG(ERROR) << "Not implemented.";
    return NBestEncodeResult();
  }

  virtual EncodeResult SampleEncode(absl::string_view normalized,
                                    float alpha) const {
    LOG(ERROR) << "Not implemented.";
    return EncodeResult();
  }

  // 获取词对应的id并返回，如果词不被认识则返回UNK(0)。
  // Returns the vocab id of `piece`.
  // Returns UNK(0) if `piece` is unknown
  virtual int PieceToId(absl::string_view piece) const;

  // 返回id的文字表示。id必须在[0, GetPieceSize())区间内。
  // Returns the string representation of vocab with `id`.
  // id must be 0 <= id < GetPieceSize().
  virtual const std::string &IdToPiece(int id) const {
    return model_proto_->pieces(id).piece();
  }

  // 返回已认识的词的数目。
  // Returns the size of sentence pieces, which is the same
  // as the size of vocabulary for NMT.
  virtual int GetPieceSize() const { return model_proto_->pieces_size(); }

  // 返回id对应的词的分数。
  // Returns the score of `id`.
  // Score represents a log probability of the piece.
  // We can roughly estimate the unigram frequency of the piece.
  virtual float GetScore(int id) const {
    return model_proto_->pieces(id).score();
  }

  // Returns true if `id` is unknown symbol.
  virtual bool IsUnknown(int id) const {
    return (model_proto_->pieces(id).type() ==
            ModelProto::SentencePiece::UNKNOWN);
  }

  // 返回id对应的词是否为控制符。
  // Returns true if `id` is control symbol.
  virtual bool IsControl(int id) const {
    return (model_proto_->pieces(id).type() ==
            ModelProto::SentencePiece::CONTROL);
  }

  // Returns true if `id` is unused symbol.
  virtual bool IsUnused(int id) const {
    return (model_proto_->pieces(id).type() ==
            ModelProto::SentencePiece::UNUSED);
  }

  // Returns true if `id` is user defined symbol.
  virtual bool IsUserDefined(int id) const {
    return (model_proto_->pieces(id).type() ==
            ModelProto::SentencePiece::USER_DEFINED);
  }

 protected:
  void InitializePieces();

  // 对于快速估计的内联实现。
  // Non-virtual (inlined) implementation for faster execution.
  inline float GetScoreInlined(int id) const {
    return model_proto_->pieces(id).score();
  }

  inline bool IsUnknownInlined(int id) const {
    return (model_proto_->pieces(id).type() ==
            ModelProto::SentencePiece::UNKNOWN);
  }

  inline bool IsControlInlined(int id) const {
    return (model_proto_->pieces(id).type() ==
            ModelProto::SentencePiece::CONTROL);
  }

  inline bool IsUnusedInlined(int id) const {
    return (model_proto_->pieces(id).type() ==
            ModelProto::SentencePiece::UNUSED);
  }

  inline bool IsUserDefinedInlined(int id) const {
    return (model_proto_->pieces(id).type() ==
            ModelProto::SentencePiece::USER_DEFINED);
  }

  const ModelProto *model_proto_ = nullptr;

  // PrefixMatcher for user defined symbols.
  // 对用户定义符号的前缀匹配器。
  std::unique_ptr<normalizer::PrefixMatcher> matcher_;

  // 对于常规的词的piece->id的键值表。
  // piece -> id map for normal pieces
  PieceToIdMap pieces_;

  // 对于控制符和未知字的piece->id的键值表。
  // piece -> id map for control and unknown
  PieceToIdMap reserved_id_map_;

  // unknown id.
  int unk_id_ = 0;

  // status.
  util::Status status_;
};
}  // namespace sentencepiece
#endif  // MODEL_INTERFACE_H_
