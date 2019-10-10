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

#ifndef UNIGRAM_MODEL_TRAINER_H_
#define UNIGRAM_MODEL_TRAINER_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "sentencepiece_model.pb.h"
#include "third_party/absl/strings/string_view.h"
#include "trainer_interface.h"
#include "unigram_model.h"
#include "util.h"

// DOC:命名空间 sentencepiece::unigram
namespace sentencepiece {
namespace unigram {

using string_util::UnicodeText;

// DOC:
// TrainerModel类
// 继承Model类
class TrainerModel : public Model {
 public:
  using SentencePieces = std::vector<std::pair<std::string, float>>;

  // DOC:
  //    对象创建
  TrainerModel() {}
  TrainerModel(const ModelProto &model_proto) = delete;
  TrainerModel(const TrainerSpec &trainer_spec,
               const NormalizerSpec &normalizaiton_spec);
  ~TrainerModel() override;

  // DOC:
  //    返回完整的句段
  // 注意:
  //    只包含元信息 起始符/终止符等不包含在内
  // Returns the sentencepieces.
  // The meta symbols, e.g., </s> are NOT included.
  const SentencePieces &GetSentencePieces() const;

  // DOC:
  //    对 UnigramModel 设置新的句段
  // 注意:
  //    只包含元信息 起始符/终止符等不包含在内
  // Sets sentencepieces. The sentencepieces are moved.
  // The meta symbols, e.g., </s> are NOT included.
  void SetSentencePieces(SentencePieces &&sentencepieces);

  // DOC:
  //    对字符串进行处理计算
  //    继承Model类中的Encode方法
  // 参数:
  //    normalized -- 已规范化的字符串
  // 返回:
  //    经过Viterbi算法处理过的最佳分词序列
  EncodeResult Encode(absl::string_view normalized) const override {
    return {};
  }

  // DOC:
  //    私有成员变量:
  //        sentencepieces_ -- 保存分词块的数组
  //        trainer_spec_ --
  //        normalizer_spec_ --
  //        model_proto_data_ --
 private:
  SentencePieces sentencepieces_;
  TrainerSpec trainer_spec_;
  NormalizerSpec normalizer_spec_;
  ModelProto model_proto_data_;
};

// DOC:
// Trainer类
// 继承 TrainerInterface 类
class Trainer : public TrainerInterface {
 public:
  Trainer(const TrainerSpec &trainer_spec,
          const NormalizerSpec &normalizer_spec)
      : TrainerInterface::TrainerInterface(trainer_spec, normalizer_spec) {}

  util::Status Train() override;

 private:
  FRIEND_TEST(TrainerTest, IsValidSentencePieceTest);

  // DOC:
  //    从训练集中选取种子分词块
  //    所选分词块对数量由 seed_sentencepiece_size 决定
  // Makes seed pieces from the training corpus.
  // The size of seed pieces is determined by seed_sentencepiece_size.
  TrainerModel::SentencePieces MakeSeedSentencePieces() const;

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
  // Executes the E step of EM and returns expected count.
  // The index of return array is the vocab id.
  // |objective| is a negative likelihood of the current model.
  // |num_token| is the number of total tokens to tokenize
  // training corpus.
  std::vector<float> RunEStep(const TrainerModel &model, float *objective,
                              int64 *num_tokens) const;

  // DOC:
  //    进行 EM (期望最大化) 算法的 M 步 -- 求极大
  //
  // 参数:
  //    model -- 训练模型引用
  //    expected -- 经E步处理所得的期望值数组引用
  //
  // 返回:
  //    返回处理后新的分词块数组
  // Executes the M step of EM with the expected frequency and
  // returns new pieces.
  TrainerModel::SentencePieces RunMStep(
      const TrainerModel &model, const std::vector<float> &expected) const;

  // DOC:
  //    每次进行 EM (期望最大化) 子迭代后 对当前全部分词块进行剪枝
  //    剪去冗余部分 提高算法效率
  // Heuristically prunes the current pieces.
  // This is called after each EM sub-iteration.
  TrainerModel::SentencePieces PruneSentencePieces(
      const TrainerModel &model) const;

  // DOC:
  //    通过对必要词块/控制字符/用户自定义词块的处理 决定最终分词块
  // 参数:
  //    model -- 训练模型的引用
  // Makes the final sentence pieces by incorporating the required characters
  // and control/user defined symbols.
  TrainerModel::SentencePieces FinalizeSentencePieces(
      const TrainerModel &model) const;

  // DOC:
  //    词库期望规模
  //    当分词数量小于词库期望规模 则跳出主训练循环
  //    当前词库期望规模 = 词库规模 * 1.1
  // When the size of SentencePieces becomes less than desired_vocab_size_,
  // break the main training loop. desired_vocab_size_ = 1.1 * vocab_size_
  // for now.
  int desired_vocab_size_;
};
}  // namespace unigram
}  // namespace sentencepiece
#endif  // UNIGRAM_MODEL_TRAINER_H_
