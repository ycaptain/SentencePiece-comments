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
#include "util.h"

// 命名空间 sentencepiece
namespace sentencepiece {
// 命名空间 character
namespace character {

// DOC:
// 初始化模型（文本块）函数
//
// 参数:
//		model_proto -- 欲初始化模型原型
Model::Model(const ModelProto &model_proto) {
  model_proto_ = &model_proto;
  InitializePieces();
}

Model::~Model() {}

// DOC:
// 为 normalized 文本(string_view) 编码函数
// 
// 参数:
//		normalized -- 欲编码 normalized 文本
// 返回:
//		编码结果
EncodeResult Model::Encode(absl::string_view normalized) const {
  if (!status().ok() || normalized.empty()) {
    return {};
  }

  // Splits the input into character sequence
// DOC: 将输入文本拆分成字符序列
  EncodeResult output;
  while (!normalized.empty()) {
    const int mblen = matcher_->PrefixMatch(normalized);
    absl::string_view w(normalized.data(), mblen);
    output.emplace_back(w, PieceToId(w));
    normalized.remove_prefix(mblen);
  }

  return output;
}

}  // namespace character
}  // namespace sentencepiece
