// Copyright 2018 Google Inc.
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

#ifndef SENTENCEPIECE_TRAINER_H_
#define SENTENCEPIECE_TRAINER_H_

#include <string>
#include "sentencepiece_processor.h"

namespace sentencepiece {

class TrainerSpec;
class NormalizerSpec;

class SentencePieceTrainer {
 public:
  // Trains SentencePiece model with `trainer_spec`.
  // Default `normalizer_spec` is used.用“教练规格”训练哨兵模型。              //使用默认的“normalizer\u spec”。
  //DOC：
//  用默认的规范化器设置与指定的训练方法训练SentencePiece模型
// 参数:
//      const TrainerSpec &trainer_spec -- 用以训练SentencePiece模型的指定训练方法
// 返回:
//      util::Status  -- 用于表示训练的状态与结果
  static util::Status Train(const TrainerSpec &trainer_spec);
  // 传入参数训练模型。
  // Trains SentencePiece model with `trainer_spec` and
  // `normalizer_spec`.
  // DOC：
//  用指定的规范化器设置与指定的训练方法训练SentencePiece模型
// 参数:
//      const TrainerSpec &trainer_spec -- 用以训练SentencePiece模型的指定训练方法
//      const NormalizerSpec &normalizer_spec -- 用以训练SentencePiece模型的指定规范化器设置
// 返回:
//      util::Status  -- 用于表示训练的状态与结果
  static util::Status Train(const TrainerSpec &trainer_spec,
                            const NormalizerSpec &normalizer_spec);

  // Trains SentencePiece model with command-line string in `args`,
  // e.g.,
  // '--input=data --model_prefix=m --vocab_size=8192 model_type=unigram'
  //DOC:
//  根据输入的命令行字符串指令训练sentencepiece模型
// 参数:
//      util::min_string_view args -- 指定如何训练SentencePiece模型的命令行字符串
// 返回:
//      util::Status  -- 用于表示训练的状态与结果
  static util::Status Train(util::min_string_view args);
  // 从预编译的Normalizer名称创建一个Normalizer对象。
  // Handy function to make a normalizer spec from the pre-compiled
  // normalization name. Do not use this method in production as it crashes
  // when `name` is invalid. Useful for unittesting.
  //DOC：
// 用于从已有的的规范化器名称生成规范化器设置
// 参数:
//      util::min_string_view name -- 用以生成规范化器设置的已有的的规范化器名称字符串
// 返回:
//      NormalizerSpec spec -- 生成的规范化器设置
  static NormalizerSpec GetNormalizerSpec(util::min_string_view name);
  // 从名称或normalization_rule_tsv为normalizer_spec输入必要的字段数据。
  // Populates necessary fields (precompiled_charmap) from
  // `NormalizerSpec::name` or `NormalizerSpec::normalization_rule_tsv`.
  //DOC:
  // 用于对规范化器的规范增添数据与信息
// 参数:
//      NormalizerSpec &normalizer_spec -- 用以加入规范化器的规范中的数据与信息
// 返回:
//      util::Status  -- 用于表示操作的状态与结果
  static util::Status PopulateNormalizerSpec(NormalizerSpec *normalizer_spec);

  // Overrides `trainer_spec` and `normalizer_spec` with the
  // command-line string in `args`.
  //DOC:
//  根据输入的命令行字符串指令重新配置SentencePiece模型的训练方法的设置与规范化器的设置
// 参数:
//      util::min_string_view args -- 输入的命令行字符串
//      TrainerSpec *trainer_spec -- 需要被重新配置的SentencePiece模型的训练方法的设置
//      NormalizerSpec *normalizer_spec -- 需要被重新配置的SentencePiece模型的规范化器的设置
// 返回:
//      util::Status  -- 用于表示操作的状态与结果
  static util::Status MergeSpecsFromArgs(util::min_string_view args,
                                         TrainerSpec *trainer_spec,
                                         NormalizerSpec *normalizer_spec);
  // 设置Protobuf中的字段值。
  // Helper function to set `field_name=value` in `message`.
  // When `field_name` is repeated, multiple values can be passed
  // with comma-separated values. `field_name` must not be a nested message.
  // The body of these functions are automatically generated with
  // data/gen_spec_parser.pl
  //DOC:
//  用于生成模型的训练方法的设置的信息
// 参数:
//      const std::string &name -- 模型的训练方法的名称
//      const std::string &value -- 模型的训练方法的值
//      TrainerSpec *message -- 用于存储生成的模型的训练方法设置的信息
// 返回:
//      util::Status  -- 用于表示操作的状态与结果
  static util::Status SetProtoField(const std::string &name,
                                    const std::string &value,
                                    TrainerSpec *message);
  // 设置Protobuf中的字段值。
  //DOC：
//    用于生成规范化器的规范的信息
// 参数:
//      const std::string &name -- 模型的规范化器规范的名称
//      const std::string &value -- 模型的规范化器规范的值
//      TrainerSpec *message -- 用于存储生成的模型的规范化器规范的信息
// 返回:
//      util::Status  -- 用于表示操作的状态与结果
  static util::Status SetProtoField(const std::string &name,
                                    const std::string &value,
                                    NormalizerSpec *message);

  SentencePieceTrainer() = delete;
  ~SentencePieceTrainer() = delete;
};
}  // namespace sentencepiece
#endif  // SENTENCEPIECE_TRAINER_H_
