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

#include "char_model_trainer.h"
#include <string>
#include <vector>
#include "filesystem.h"
#include "sentencepiece_processor.h"
#include "testharness.h"
#include "util.h"

// 命名空间 sentencepiece
namespace sentencepiece {
// 命名空间 character
namespace character {
// 无名命名空间 仅限于本文件内
namespace {

// Space symbol (U+2581)
// DOC: 空格 Unicode 编码宏定义
#define WS "\xe2\x96\x81"

// DOC:
// 训练器执行函数
//
// 参数:
//		input -- 输入文本
//		size -- 词汇表大小
//
// 返回:
//		将拆分得到的文本块以空格连接的文本结果
std::string RunTrainer(const std::vector<std::string> &input, int size) {
  test::ScopedTempFile input_scoped_file("input");
  test::ScopedTempFile model_scoped_file("model");
  const std::string input_file = input_scoped_file.filename();
  const std::string model_prefix = model_scoped_file.filename();
  {
    auto output = filesystem::NewWritableFile(input_file);
    for (const auto &line : input) {
      output->WriteLine(line);
    }
  }

  TrainerSpec trainer_spec;
  trainer_spec.set_model_type(TrainerSpec::CHAR);
  trainer_spec.add_input(input_file);
  trainer_spec.set_vocab_size(size);
  trainer_spec.set_model_prefix(model_prefix);

  NormalizerSpec normalizer_spec;
  normalizer_spec.set_name("identity");

  Trainer trainer(trainer_spec, normalizer_spec);
  EXPECT_OK(trainer.Train());

  SentencePieceProcessor processor;
  EXPECT_OK(processor.Load(model_prefix + ".model"));

  const auto &model = processor.model_proto();
  std::vector<std::string> pieces;

  // remove <unk>, <s>, </s>
// DOC: 移除 <unk>, <s>, </s>
  for (int i = 3; i < model.pieces_size(); ++i) {
    pieces.emplace_back(model.pieces(i).piece());
  }

  return string_util::Join(pieces, " ");
}

TEST(TrainerTest, BasicTest) {
// DOC: 测试文本 " a e p n I h l v" 与 训练后得到的文本块结果文本是否相同，不同则触发异常
  EXPECT_EQ(WS " a e p n I h l v",
            RunTrainer({"I have a pen", "I have an apple", "apple pen"}, 100));
  EXPECT_EQ(WS " a",  // <unk>, <s>, </s>, _, a
            RunTrainer({"I have a pen", "I have an apple", "apple pen"}, 5));
}

}  // namespace
}  // namespace character
}  // namespace sentencepiece
