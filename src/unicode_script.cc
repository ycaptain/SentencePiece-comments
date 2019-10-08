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

#include "unicode_script.h"
#include <unordered_map>
#include "unicode_script_map.h"
#include "util.h"

// DOC: 命名空间 sentencepiece
namespace sentencepiece {
// DOC: 命名空间 sentencepiece::unicode_script
namespace unicode_script {
namespace {
// DOC:
// 语料所属字符集检测类
class GetScriptInternal {
 public:
// DOC:
// GetScriptInternal 类构造函数 初始化字符集表
// 
// 参数:
//      smap_ -- 字符集表存储变量的引用
  GetScriptInternal() { InitTable(&smap_); }

  // DOC:
  // 获取字符归属字符集类型
  // 
  // 参数:
  //      c -- 待判断的字符
  ScriptType GetScript(char32 c) const {
	// DOC:
	// 调用针对 STL map 的通用查找方法
    return port::FindWithDefault(smap_, c, ScriptType::U_Common);
  }

 private:
  // 语言表
  std::unordered_map<char32, ScriptType> smap_;
};
}  // namespace

// DOC:
// 获取字符归属字符集类型
// 
// 参数:
//      c -- 待判断的字符
ScriptType GetScript(char32 c) {
  static GetScriptInternal sc;
  return sc.GetScript(c);
}
}  // namespace unicode_script
}  // namespace sentencepiece
