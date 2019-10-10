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

#ifndef FREELIST_H_
#define FREELIST_H_

#include <string.h>
#include <vector>

// DOC:
// 命名空间sentencepiece::model
namespace sentencepiece {
namespace model {

// 定义FreeList类
// 一次释放大量空间
// Simple FreeList that allocates a chunk of T at once.
template <class T>
class FreeList {
 public:
  FreeList() = delete;
  explicit FreeList(size_t chunk_size) : chunk_size_(chunk_size) {}
  virtual ~FreeList() {
    for (auto& chunk : freelist_) delete[] chunk;
  }

  // 实现对大量空间进行复用
  // `Free` doesn't free the object but reuse the allocated memory chunks.
  void Free() {
    const int size = std::min<int>(chunk_index_ + 1, freelist_.size());
    for (int i = 0; i < size; ++i) {
      T* chunk = freelist_[i];
      memset(chunk, 0, sizeof(*chunk) * chunk_size_);
    }
    chunk_index_ = 0;
    element_index_ = 0;
  }

  // 返回分配对元素数目
  // Returns the number of allocated elements.
  size_t size() const { return chunk_size_ * chunk_index_ + element_index_; }

  // 以数组的方式返回所需元素
  //
  // 参数:
  //    index -- 所需元素的下标
  //
  // 返回:
  //    存储元素数组的头指针
  // Returns the element as an array.
  T* operator[](size_t index) const {
    return freelist_[index / chunk_size_] + index % chunk_size_;
  }

  // 为新元素分配空间
  //
  // 返回:
  //    新元素类型的指针
  // Allocates new element.
  T* Allocate() {
    if (element_index_ >= chunk_size_) {
      ++chunk_index_;
      element_index_ = 0;
    }

    if (chunk_index_ == freelist_.size()) {
      T* chunk = new T[chunk_size_];
      memset(chunk, 0, sizeof(*chunk) * chunk_size_);
      freelist_.push_back(chunk);
    }

    T* result = freelist_[chunk_index_] + element_index_;
    ++element_index_;

    return result;
  }

  // 私有变量:
  //    freelist_ -- 储存各类型变量
  //    element_index_ -- 元素在其所处chunk中元素的序号
  //    chunk_index_ -- chunk在freelist_中的序号
  //    chunk_size_ -- chunk的数据规模
 private:
  std::vector<T*> freelist_;

  // The last element is stored at freelist_[chunk_index_][element_index_]
  size_t element_index_ = 0;
  size_t chunk_index_ = 0;
  const size_t chunk_size_ = 0;
};
}  // namespace model
}  // namespace sentencepiece
#endif  // FREELIST_H_
