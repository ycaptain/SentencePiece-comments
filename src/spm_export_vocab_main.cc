

// Copyright 2016 Google Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// n//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.!

#include <sstream>
#include "common.h"
#include "filesystem.h"
#include "flags.h"
#include "sentencepiece_model.pb.h"
#include "sentencepiece_processor.h"

// DOC:
// 初始化flags，定义模型输出词组的参数的数据类型，名称，默认值和帮助信息
DEFINE_string(output, "", "Output filename");
DEFINE_string(model, "", "input model file name");
DEFINE_string(output_format, "txt", "output format. choose from txt or proto");

// DOC:
// 输出词组的主函数，载入命令行参数，输出词组
int main(int argc, char *argv[]) {
  // DOC:
  // 解析命令行参数并修改对应Flag
  sentencepiece::flags::ParseCommandLineFlags(argc, argv);
  // sentencepiece处理器
  sentencepiece::SentencePieceProcessor sp;
  // 检测Flag参数model是否为空
  CHECK_OK(sp.Load(FLAGS_model));

  // DOC:
  // 获取输出
  auto output = sentencepiece::filesystem::NewWritableFile(FLAGS_output);
  // 检测输出状态是否正常
  CHECK_OK(output->status());

  // 如果flags的输出格式为txt文件，将结果输出到文件中
  if (FLAGS_output_format == "txt") {
    for (const auto &piece : sp.model_proto().pieces()) {
      std::ostringstream os;
      os << piece.piece() << "\t" << piece.score();
      output->WriteLine(os.str());
    }
    // 如果flags的输出格式为proto文件，不进行操作
  } else if (FLAGS_output_format == "proto") {
    //    output->Write(sp.model_proto().Utf8DebugString());
  }

  return 0;
}
