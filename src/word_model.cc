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

#include "word_model.h"
#include "util.h"

// 命名空间 sentencepiece
namespace sentencepiece {
namespace word {

// 定义一个以 model_proto 为参数的分词模型
// 并初始化分词工具
Model::Model(const ModelProto &model_proto) {
  model_proto_ = &model_proto;
  InitializePieces();
}

Model::~Model() {}

// 对已经规范化的字符串 normalized 进行编码，并返回编码后的结果
// 若字符串 normalized 为空或者状态异常，则直接返回空。否则进行编码并输出
EncodeResult Model::Encode(absl::string_view normalized) const {
  if (!status().ok() || normalized.empty()) {
    return {};
  }

  EncodeResult output;
  for (const auto &w : SplitIntoWords(normalized)) {
    output.emplace_back(w, PieceToId(w));
  }

  return output;
}

}  // namespace word
}  // namespace sentencepiece
