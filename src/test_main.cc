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

#include "flags.h"
#include "testharness.h"

// DOC:
// 如果操作系统是windows，保存数据的路径为..\\data
#ifdef OS_WIN
DEFINE_string(data_dir, "..\\data", "Data directory");
// 如果操作系统不是windows，保存数据的路径为../data
#else
DEFINE_string(data_dir, "../data", "Data directory");
#endif

// DOC:
// 执行所有测试文件
int main(int argc, char **argv) {
  // DOC:
  // 读取额外的参数
  std::vector<std::string> rest_args;
  // 解析命令行参数并修改对应Flag
  sentencepiece::flags::ParseCommandLineFlags(argc, argv, &rest_args);
  // 执行所有的测试文件
  sentencepiece::test::RunAllTests();
  return 0;
}
