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
__device__ __attribute__((always_inline)) void atomicStore(T* ptr, const T& val, cuda::memory_order memoryOrder) {
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
  __device__ __attribute__((always_inline)) void relaxedSignal() {
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

  __device__ void __attribute__((always_inline)) semaphoreIncrement() { *outboundSemaphoreId += 1; }
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

  __device__ __attribute__((used)) void relaxedSignal() { 
      // printf("RT relaxed signal start %p\n", this);
      semaphore_.relaxedSignal(); 
  }

  __device__ __attribute((used)) void wait(int64_t maxSpinCount = 10000000) { 
      // printf("RT wait start %p\n", this);
      semaphore_.wait(maxSpinCount); 
  }

  __device__ int4 read_int4(uint64_t index) {
    return *(reinterpret_cast<int4*>(dst_) + index);
  }
  
  __device__ __attribute((used)) float read(uint64_t index) {
    /* printf("RT read on %lu got %f at %lu \n", index, *(reinterpret_cast<float*>(dst_) + index), dst_); */
    return *(reinterpret_cast<float*>(dst_) + index);
  }

  __device__ void write_int4(uint64_t index, const int4& v) {
    *(reinterpret_cast<int4*>(dst_) + index) = v;
  }

  __device__ __attribute((used)) void write(uint64_t index, const float v) {
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

  __device__ __attribute((used)) void get(uint64_t offset, uint64_t bytes, uint32_t threadId, uint32_t numThreads) {
    get_helper(offset, offset, bytes, threadId, numThreads);
  }


};

union alignas(16) LL16Packet {
  struct {
    uint32_t data1;
    uint32_t flag1;
    uint32_t data2;
    uint32_t flag2;
  };
  using Payload = uint2;

  ulonglong2 raw_;

  __device__ LL16Packet() {}

  __device__ LL16Packet(uint2 val, uint32_t flag) {
    data1 = val.x;
    flag1 = flag;
    data2 = val.y;
    flag2 = flag;
  }

  __device__ void write(uint32_t val1, uint32_t val2, uint32_t flag) {
    asm volatile("st.volatile.global.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(&raw_), "r"(val1), "r"(flag), "r"(val2),
                 "r"(flag));
  }

  __device__ void write(uint64_t val, uint32_t flag) { write((uint32_t)val, (uint32_t)(val >> 32), flag); }

  __device__ bool readOnce(uint32_t flag, uint2& data) const {
    uint32_t flag1, flag2;
    asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(data.x), "=r"(flag1), "=r"(data.y), "=r"(flag2)
                 : "l"(&raw_));
    return (flag1 != flag) || (flag2 != flag);
  }

  __device__ __attribute((used)) uint2 read(uint32_t flag, int64_t maxSpinCount = 100000000) const {
    uint2 data;
    POLL_MAYBE_JAILBREAK(readOnce(flag, data), maxSpinCount);
    return data;
  }

  __device__ void clear() { raw_ = make_ulonglong2(0, 0); }
};


__device__ mscclpp::DeviceSyncer DS;
__device__ uint2 foo(){
    SmChannelDeviceHandle handle;
    handle.relaxedSignal();
    handle.wait();
    handle.write(5, handle.read(3));
    DS.sync(99);
    bool mask[10];
    if (mask[0]) {
        handle.relaxedSignal();
    }
	LL16Packet packet;
	packet.write(1, 2, 3);
	LL16Packet packet2;
    packet.data1 = 1;
    packet.flag1 = 2;
    packet.data2 = 3;
    packet.flag2 = 4;
	return packet.read(3);
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
