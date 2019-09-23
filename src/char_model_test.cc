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

#include "char_model.h"

#include <string>
#include <unordered_map>
#include <unordered_set>

#include "testharness.h"
#include "util.h"

// 命名空间 sentencepiece
namespace sentencepiece {
// 命名空间 character
namespace character {
// 无名命名空间 仅限于本文件内
namespace {

// Space symbol (U+2581)
// DOC: 空格宏定义
#define WS "\xe2\x96\x81"

// DOC:
// 创建基本模型原型函数
//
// 返回:
//		创建好的基本模型原型
ModelProto MakeBaseModelProto() {
  ModelProto model_proto;
  auto *sp1 = model_proto.add_pieces();
  auto *sp2 = model_proto.add_pieces();
  auto *sp3 = model_proto.add_pieces();

  sp1->set_type(ModelProto::SentencePiece::UNKNOWN);
  sp1->set_piece("<unk>");
  sp2->set_type(ModelProto::SentencePiece::CONTROL);
  sp2->set_piece("<s>");
  sp3->set_type(ModelProto::SentencePiece::CONTROL);
  sp3->set_piece("</s>");

  return model_proto;
}

// DOC:
// 为模型原型增加文本块 (Piece) 函数
//
// 参数:
//		model_proto -- 目标模型原型指针
//		piece -- 增加的文本块 (piece)
//		score -- 模型分数（参考BLEU SCORE），默认值为0.0
void AddPiece(ModelProto *model_proto, const std::string &piece,
              float score = 0.0) {
  auto *sp = model_proto->add_pieces();
  sp->set_piece(piece);
  sp->set_score(score);
}

TEST(ModelTest, EncodeTest) {
  ModelProto model_proto = MakeBaseModelProto();

// DOC: 向模型原型增加文本块 " abcABC"
  AddPiece(&model_proto, WS, 0.0);
  AddPiece(&model_proto, "a", 0.1);
  AddPiece(&model_proto, "b", 0.2);
  AddPiece(&model_proto, "c", 0.3);
  AddPiece(&model_proto, "d", 0.4);
  AddPiece(&model_proto, "ABC", 0.4);
  model_proto.mutable_pieces(8)->set_type(
      ModelProto::SentencePiece::USER_DEFINED);

  const Model model(model_proto);

  EncodeResult result;

  result = model.Encode("");
  EXPECT_TRUE(result.empty());

 // DOC: 测试 " a b c" 编码结果是否与测试文本相同，不同则触发异常
  result = model.Encode(WS "a" WS "b" WS "c");
  EXPECT_EQ(6, result.size());
  EXPECT_EQ(WS, result[0].first);
  EXPECT_EQ("a", result[1].first);
  EXPECT_EQ(WS, result[2].first);
  EXPECT_EQ("b", result[3].first);
  EXPECT_EQ(WS, result[4].first);
  EXPECT_EQ("c", result[5].first);

 // DOC: 测试 " ab cd abc" 编码结果是否与测试文本相同，不同则触发异常
  result = model.Encode(WS "ab" WS "cd" WS "abc");
  EXPECT_EQ(10, result.size());
  EXPECT_EQ(WS, result[0].first);
  EXPECT_EQ("a", result[1].first);
  EXPECT_EQ("b", result[2].first);
  EXPECT_EQ(WS, result[3].first);
  EXPECT_EQ("c", result[4].first);
  EXPECT_EQ("d", result[5].first);
  EXPECT_EQ(WS, result[6].first);
  EXPECT_EQ("a", result[7].first);
  EXPECT_EQ("b", result[8].first);
  EXPECT_EQ("c", result[9].first);

  // makes a broken utf-8
 // DOC: 使用一个损坏的utf-8编码字符 "あ" 进行测试编码是否正确，不正确则触发异常
  const std::string broken_utf8 = std::string("あ").substr(0, 1);
  result = model.Encode(broken_utf8);
  EXPECT_EQ(1, result.size());
  EXPECT_EQ(broken_utf8, result[0].first);

  // "ABC" is treated as one piece, as it is USER_DEFINED.
// DOC: "ABC" 被当作一个词划分，与用户定义相同

// DOC: 测试 " abABCcd" 编码结果是否与测试文本相同，不同则触发异常
  result = model.Encode(WS "abABCcd");
  EXPECT_EQ(6, result.size());
  EXPECT_EQ(WS, result[0].first);
  EXPECT_EQ("a", result[1].first);
  EXPECT_EQ("b", result[2].first);
  EXPECT_EQ("ABC", result[3].first);
  EXPECT_EQ("c", result[4].first);
  EXPECT_EQ("d", result[5].first);
}


TEST(CharModelTest, NotSupportedTest) {
// DOC: 测试编码结果和NBest编码结果是否正确，不正确则触发异常
  ModelProto model_proto = MakeBaseModelProto();
  const Model model(model_proto);
  EXPECT_EQ(NBestEncodeResult(), model.NBestEncode("test", 10));
  EXPECT_EQ(EncodeResult(), model.SampleEncode("test", 0.1));
}

}  // namespace
}  // namespace character
}  // namespace sentencepiece
