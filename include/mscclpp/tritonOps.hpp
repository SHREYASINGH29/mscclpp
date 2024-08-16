// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef RELAXED_SIGNAL_HPP
#define RELAXED_SIGNAL_HPP

#include <cstdint>
#include <iostream>

// If a spin is stuck, print a warning and keep spinning.
#define POLL_MAYBE_JAILBREAK(__cond, __max_spin_cnt)                     \
  do {                                                                   \
    int64_t __spin_cnt = 0;                                              \
    while (__cond) {                                                     \
      if (__max_spin_cnt >= 0 && __spin_cnt++ == __max_spin_cnt) {       \
        __assert_fail(#__cond, __FILE__, __LINE__, __PRETTY_FUNCTION__); \
      }                                                                  \
    }                                                                    \
  } while (0);

#include <cuda/atomic>

namespace mscclpp {

constexpr cuda::memory_order memoryOrderRelaxed = cuda::memory_order_relaxed;
constexpr cuda::memory_order memoryOrderAcquire = cuda::memory_order_acquire;

template <typename T>
__device__ void atomicStore(T* ptr, const T& val, cuda::memory_order memoryOrder) {
  cuda::atomic_ref<T, cuda::thread_scope_system>{*ptr}.store(val, memoryOrder);
}

template <typename T>
__device__ T atomicLoad(T* ptr, cuda::memory_order memoryOrder) {
  return cuda::atomic_ref<T, cuda::thread_scope_system>{*ptr}.load(memoryOrder);
}

struct DeviceSyncer {
 public:
  DeviceSyncer() = default;
  ~DeviceSyncer() = default;

  __device__ void sync(int blockNum, int64_t maxSpinCount = 100000000) {
    // printf("RT sync\n");
    unsigned int maxOldCnt = blockNum - 1;
    __syncthreads();
    if (blockNum == 1) return;
    if (threadIdx.x == 0) {
      __threadfence();
      unsigned int tmp = preFlag_ ^ 1;
      if (atomicInc(&count_, maxOldCnt) == maxOldCnt) {
        atomicStore(&flag_, tmp, memoryOrderRelaxed);
      } else {
        POLL_MAYBE_JAILBREAK((atomicLoad(&flag_, memoryOrderRelaxed) != tmp), maxSpinCount);
      }
      preFlag_ = tmp;
    }
    __syncthreads();
  }

 private:
  unsigned int flag_;
  unsigned int count_;
  unsigned int preFlag_;
};

struct SmDevice2DeviceSemaphoreDeviceHandle {
  __device__ void relaxedSignal() {
    semaphoreIncrement();
    atomicStore(remoteInboundSemaphoreId, semaphoreGetLocal(), memoryOrderRelaxed);
    // printf("RT relaxed signal end\n");
  }

  __device__ void wait(int64_t maxSpinCount = 100000000) {
    (*expectedInboundSemaphoreId) += 1;
    POLL_MAYBE_JAILBREAK((atomicLoad(inboundSemaphoreId, memoryOrderAcquire) < (*expectedInboundSemaphoreId)),
                         maxSpinCount);
    // printf("RT wait end\n");
  }

  __device__ void semaphoreIncrement() { *outboundSemaphoreId += 1; }
  __device__ uint64_t semaphoreGetLocal() const { return *outboundSemaphoreId; }

  uint64_t* inboundSemaphoreId;
  uint64_t* outboundSemaphoreId;
  uint64_t* remoteInboundSemaphoreId;
  uint64_t* expectedInboundSemaphoreId;
};

struct SmChannelDeviceHandle {
  SmDevice2DeviceSemaphoreDeviceHandle semaphore_;
  void* src_;
  void* dst_;
  void* getPacketBuffer_;

  __device__ void relaxedSignal() { 
      // printf("RT relaxed signal start %p\n", this);
      semaphore_.relaxedSignal(); 
  }

  __device__ void wait(int64_t maxSpinCount = 10000000) { 
      // printf("RT wait start %p\n", this);
      semaphore_.wait(maxSpinCount); 
  }

  __device__ int4 read_int4(uint64_t index) {
    return *(reinterpret_cast<int4*>(dst_) + index);
  }
  
  __device__ float read(uint64_t index) {
    /* printf("RT read on %lu got %f at %lu \n", index, *(reinterpret_cast<float*>(dst_) + index), dst_); */
    return *(reinterpret_cast<float*>(dst_) + index);
  }

  __device__ void write_int4(uint64_t index, const int4& v) {
    *(reinterpret_cast<int4*>(dst_) + index) = v;
  }

  __device__ void write(uint64_t index, const float v) {
    /* printf("RT write %lu <= %f at %lu\n", index, v, dst_); */
    *(reinterpret_cast<float*>(dst_) + index) = v;
  }

  __device__ void writeWrapper(uint64_t index, const int v1, const int v2, const int v3, const int v4) {
      write_int4(index, (int4) {v1, v2, v3, v4});
  }

  template <typename T>
  __device__ void element_copy(T* dst, T* src, uint64_t numElems, uint32_t threadId, uint32_t numThreads) {
    T reg;
    for (size_t i = threadId; i < numElems; i += numThreads) {
      // Load to register first.
      reg = src[i];
      // Then store to destination.
      dst[i] = reg;
    }
  }

  template <typename T, bool CopyRemainder = true>
  __device__ void copy_helper(void* dst, void* src, uint64_t bytes, uint32_t threadId, uint32_t numThreads) {
    int* dstInt = reinterpret_cast<int*>(dst);
    int* srcInt = reinterpret_cast<int*>(src);
    const uintptr_t dstPtr = reinterpret_cast<uintptr_t>(dst);
    const uintptr_t srcPtr = reinterpret_cast<uintptr_t>(src);
    const uint64_t numInt = bytes / sizeof(int);
    T* dstElem = reinterpret_cast<T*>((dstPtr + sizeof(T) - 1) / sizeof(T) * sizeof(T));
    T* srcElem = reinterpret_cast<T*>((srcPtr + sizeof(T) - 1) / sizeof(T) * sizeof(T));
    uint64_t nFirstInt = (reinterpret_cast<uintptr_t>(dstElem) - dstPtr) / sizeof(int);
    if (CopyRemainder) {
      // Copy the remainder integers at the beginning.
      element_copy<int>(dstInt, srcInt, nFirstInt, threadId, numThreads);
    }
    // Copy elements.
    constexpr uint64_t nIntPerElem = sizeof(T) / sizeof(int);
    uint64_t nElem = (numInt - nFirstInt) / nIntPerElem;
    element_copy<T>(dstElem, srcElem, nElem, threadId, numThreads);
    if (CopyRemainder && nIntPerElem > 1) {
      // Copy the remainder integers at the end.
      uint64_t nLastInt = (numInt - nFirstInt) % nIntPerElem;
      element_copy<int>(dstInt + nFirstInt + nElem * nIntPerElem, srcInt + nFirstInt + nElem * nIntPerElem, nLastInt,
                         threadId, numThreads);
    }
  }

  template <int Alignment = 16, bool CopyRemainder = true>
  __device__ void copy(void* dst, void* src, uint64_t bytes, uint32_t threadId, uint32_t numThreads) {
      copy_helper<longlong2, CopyRemainder>(dst, src, bytes, threadId, numThreads);
  }


  __device__ void get_helper(uint64_t targetOffset, uint64_t originOffset, uint64_t originBytes, uint32_t threadId,
                                 uint32_t numThreads) {
    // Note that `dst` and `src` are swapped for `get()`.
    copy<16, true>((char*)src_ + originOffset, (char*)dst_ + targetOffset, originBytes, threadId,
                                   numThreads);
  }

  __device__ void get(uint64_t offset, uint64_t bytes, uint32_t threadId, uint32_t numThreads) {
    get_helper(offset, offset, bytes, threadId, numThreads);
  }


};

__device__ mscclpp::DeviceSyncer DS;
__device__ int foo(){
    SmChannelDeviceHandle handle;
    handle.relaxedSignal();
    handle.wait();
    DS.sync(99);
    bool mask[10];
    if (mask[0]) {
        handle.relaxedSignal();
    }
}

__global__ void bar(SmChannelDeviceHandle* SmChans){
    SmChans[6].relaxedSignal();
    SmChans[3].wait();
    /* int4 test = SmChans[9].read(3); */
    SmChans[8].write(5, SmChans[9].read(3));
    SmChans[8].writeWrapper(5, 1, 2, 3, 4);
    SmChans[8].get(1, 2, 3, 4);
}  

} // namespace mscclpp
#endif  // MSCCLPP_SM_CHANNEL_DEVICE_HPP_
