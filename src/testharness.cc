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

#include "testharness.h"

#ifndef OS_WIN
#include <unistd.h>
#endif

#include <memory>
#include <string>
#include <vector>

#include "common.h"
#include "util.h"

// 命名空间:sentencepiece::test
namespace sentencepiece {
namespace test {

namespace {
// 定义结构体
// 变量:
//      base -- 保存库名
//      name -- 保存文件名
//      func -- 函数指针
struct Test {
  const char *base;
  const char *name;
  void (*func)();
};
std::vector<Test> *tests;
}  // namespace

bool RegisterTest(const char *base, const char *name, void (*func)()) {
  if (tests == nullptr) {
    tests = new std::vector<Test>;
  }
  Test t;
  t.base = base;
  t.name = name;
  t.func = func;
  tests->emplace_back(t);
  return true;
}

// 进行各项测试
int RunAllTests() {
  int num = 0;
  //当测试类指针为空时 输出报错信息
  if (tests == nullptr) {
    std::cerr << "No tests are found" << std::endl;
    return 0;
  }

  // 在每一项测试完成后 输出提示信息
  for (const Test &t : *(tests)) {
    std::cerr << "[ RUN      ] " << t.base << "." << t.name << std::endl;
    (*t.func)();
    std::cerr << "[       OK ] " << t.base << "." << t.name << std::endl;
    ++num;
  }
  // 全部测试通过后输出相应信息
  std::cerr << "==== PASSED " << num << " tests" << std::endl;

  return 0;
}

// 检测当前文件名
ScopedTempFile::ScopedTempFile(absl::string_view filename) {
  char pid[64];
  snprintf(pid, sizeof(pid), "%u",
#ifdef OS_WIN
           static_cast<uint32>(::GetCurrentProcessId())
#else
           ::getpid()
#endif
  );
  filename_ = string_util::StrCat(".XXX.tmp.", filename, ".", pid);
}

// 在检测相关完成后删除临时文件
ScopedTempFile::~ScopedTempFile() {
#ifdef OS_WIN
  ::DeleteFile(WPATH(filename_.c_str()));
#else
  ::unlink(filename_.c_str());
#endif
}
}  // namespace test
}  // namespace sentencepiece
