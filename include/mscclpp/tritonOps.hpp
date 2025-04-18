// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef RELAXED_SIGNAL_HPP
#define RELAXED_SIGNAL_HPP

#include <cstdint>
#include <iostream>

#define MSCCLPP_BITS_SIZE 32
#define MSCCLPP_BITS_OFFSET 32
#define MSCCLPP_BITS_REGMEM_HANDLE 9
#define MSCCLPP_BITS_TYPE 3
#define MSCCLPP_BITS_CONNID 10
#define MSCCLPP_BITS_FIFO_RESERVED 1

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

#define OR_POLL_MAYBE_JAILBREAK(__cond1, __cond2, __max_spin_cnt)                  \
  do {                                                                             \
    int64_t __spin_cnt = 0;                                                        \
    while (true) {                                                                 \
      if (!(__cond1)) {                                                            \
        break;                                                                     \
      } else if (!(__cond2)) {                                                     \
        break;                                                                     \
      }                                                                            \
      if (__max_spin_cnt >= 0 && __spin_cnt++ == __max_spin_cnt) {                 \
        __assert_fail(#__cond1 #__cond2, __FILE__, __LINE__, __PRETTY_FUNCTION__); \
      }                                                                            \
    }                                                                              \
  } while (0);

#include <cuda/atomic>

namespace mscclpp {

constexpr cuda::memory_order memoryOrderRelaxed = cuda::memory_order_relaxed;
constexpr cuda::memory_order memoryOrderAcquire = cuda::memory_order_acquire;

const uint64_t TriggerData = 0x1; 
const uint64_t TriggerFlag = 0x2;
const uint64_t TriggerSync = 0x4;

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
    // POLL_MAYBE_JAILBREAK((atomicLoad(inboundSemaphoreId, memoryOrderAcquire) < (*expectedInboundSemaphoreId)),
                         // maxSpinCount);
    // printf("RT wait end\n");
  }

  __device__ void __attribute__((always_inline)) semaphoreIncrement() { *outboundSemaphoreId += 1; }
  __device__ uint64_t semaphoreGetLocal() const { return *outboundSemaphoreId; }

  uint64_t* inboundSemaphoreId;
  uint64_t* outboundSemaphoreId;
  uint64_t* remoteInboundSemaphoreId;
  uint64_t* expectedInboundSemaphoreId;
};

struct alignas(16) ProxyTrigger {
  uint64_t fst, snd;
};


__device__ uint64_t atomicFetchAdd(uint64_t* ptr, const uint64_t& val, cuda::memory_order memoryOrder) {
  return cuda::atomic_ref<uint64_t, cuda::thread_scope_system>{*ptr}.fetch_add(val, memoryOrder);
}

struct FifoDeviceHandle {
  __device__ uint64_t push(ProxyTrigger trigger, int64_t maxSpinCount = 1000000) {
    uint64_t curFifoHead = atomicFetchAdd(this->head, (uint64_t)1, memoryOrderRelaxed);

    trigger.snd ^= ((uint64_t)1 << (uint64_t)63);
    if (curFifoHead >= size + *(this->tailReplica)) {
      OR_POLL_MAYBE_JAILBREAK((curFifoHead >= size + atomicLoad(this->tailReplica, memoryOrderRelaxed)),
                              (atomicLoad(&(this->triggers[curFifoHead % size].fst), memoryOrderRelaxed) != 0),
                              maxSpinCount);
    }

    ProxyTrigger* triggerPtr = &(this->triggers[curFifoHead % size]);

    asm volatile("st.global.relaxed.sys.v2.u64 [%0], {%1,%2};" ::"l"(triggerPtr), "l"(trigger.fst), "l"(trigger.snd));
    return curFifoHead;
  }
  __device__ void sync(uint64_t curFifoHead, int64_t maxSpinCount = 1000000) {
    OR_POLL_MAYBE_JAILBREAK((curFifoHead >= atomicLoad(this->tailReplica, memoryOrderRelaxed)),
                            (atomicLoad(&(this->triggers[curFifoHead % size].fst), memoryOrderRelaxed) != 0),
                            maxSpinCount);
  }
  ProxyTrigger* triggers;
  uint64_t* tailReplica;
  uint64_t* head;
  int size;
};


struct Host2DeviceSemaphoreDeviceHandle {
  __device__ bool poll() {
    bool signaled = (atomicLoad(inboundSemaphoreId, memoryOrderAcquire) > (*expectedInboundSemaphoreId));
    if (signaled) (*expectedInboundSemaphoreId) += 1;
    return signaled;
  }

  __device__ void wait(int64_t maxSpinCount = 100000000) {
    (*expectedInboundSemaphoreId) += 1;
    POLL_MAYBE_JAILBREAK((atomicLoad(inboundSemaphoreId, memoryOrderAcquire) < (*expectedInboundSemaphoreId)),
                         maxSpinCount);
  }

  uint64_t* inboundSemaphoreId;
  uint64_t* expectedInboundSemaphoreId;
};

union ChannelTrigger {
  ProxyTrigger value;
  struct {
    uint64_t size : MSCCLPP_BITS_SIZE;
    uint64_t srcOffset : MSCCLPP_BITS_OFFSET;
    uint64_t : (64 - MSCCLPP_BITS_SIZE - MSCCLPP_BITS_OFFSET);  // ensure 64-bit alignment
    uint64_t dstOffset : MSCCLPP_BITS_OFFSET;
    uint64_t srcMemoryId : MSCCLPP_BITS_REGMEM_HANDLE;
    uint64_t dstMemoryId : MSCCLPP_BITS_REGMEM_HANDLE;
    uint64_t type : MSCCLPP_BITS_TYPE;
    uint64_t chanId : MSCCLPP_BITS_CONNID;
    uint64_t : (64 - MSCCLPP_BITS_OFFSET - MSCCLPP_BITS_REGMEM_HANDLE - MSCCLPP_BITS_REGMEM_HANDLE - MSCCLPP_BITS_TYPE -
                MSCCLPP_BITS_CONNID - MSCCLPP_BITS_FIFO_RESERVED);  // ensure 64-bit alignment
    uint64_t reserved : MSCCLPP_BITS_FIFO_RESERVED;
  } fields;

  __device__ ChannelTrigger() {}

  __device__ ChannelTrigger(ProxyTrigger value) : value(value) {}

  __device__ ChannelTrigger(uint64_t type, uint64_t dst, uint64_t dstOffset, uint64_t src,
                                       uint64_t srcOffset, uint64_t bytes, int semaphoreId) {
    constexpr uint64_t maskSize = (1ULL << MSCCLPP_BITS_SIZE) - 1;
    constexpr uint64_t maskSrcOffset = (1ULL << MSCCLPP_BITS_OFFSET) - 1;
    constexpr uint64_t maskDstOffset = (1ULL << MSCCLPP_BITS_OFFSET) - 1;
    constexpr uint64_t maskSrcMemoryId = (1ULL << MSCCLPP_BITS_REGMEM_HANDLE) - 1;
    constexpr uint64_t maskDstMemoryId = (1ULL << MSCCLPP_BITS_REGMEM_HANDLE) - 1;
    constexpr uint64_t maskType = (1ULL << MSCCLPP_BITS_TYPE) - 1;
    constexpr uint64_t maskChanId = (1ULL << MSCCLPP_BITS_CONNID) - 1;
    value.fst = (((srcOffset & maskSrcOffset) << MSCCLPP_BITS_SIZE) + (bytes & maskSize));
    value.snd = (((((((((semaphoreId & maskChanId) << MSCCLPP_BITS_TYPE) + ((uint64_t)type & maskType))
                      << MSCCLPP_BITS_REGMEM_HANDLE) +
                     (dst & maskDstMemoryId))
                    << MSCCLPP_BITS_REGMEM_HANDLE) +
                   (src & maskSrcMemoryId))
                  << MSCCLPP_BITS_OFFSET) +
                 (dstOffset & maskDstOffset));
  }
};

struct ProxyChannelDeviceHandle {
  uint32_t semaphoreId_;

  Host2DeviceSemaphoreDeviceHandle semaphore_;
  FifoDeviceHandle fifo_;

  __device__ void putWithSignal(uint64_t dst, uint64_t dstOffset, uint64_t src, uint64_t srcOffset,
                                           uint64_t size) {
    fifo_.push(ChannelTrigger(TriggerData | TriggerFlag, dst, dstOffset, src, srcOffset, size, semaphoreId_).value);
  }

  __device__ void putWithSignal(uint64_t dst, uint64_t src, uint64_t offset, uint64_t size) {
    putWithSignal(dst, offset, src, offset, size);
  }

  __device__ void wait(int64_t maxSpinCount = 10000000) { semaphore_.wait(maxSpinCount);  }

  __device__ void flush() {
    uint64_t curFifoHead = fifo_.push(ChannelTrigger(TriggerSync, 0, 0, 0, 0, 1, semaphoreId_).value);
    fifo_.sync(curFifoHead);
  }

  __device__ void put(uint64_t dst, uint64_t dstOffset, uint64_t src, uint64_t srcOffset, uint64_t size) {
    fifo_.push(ChannelTrigger(TriggerData, dst, dstOffset, src, srcOffset, size, semaphoreId_).value);
  }

  __device__ void put(uint64_t dst, uint64_t src, uint64_t offset, uint64_t size) {
    put(dst, offset, src, offset, size);
  }

  __device__ bool poll() { return semaphore_.poll();  }

  __device__ void signal() { fifo_.push(ChannelTrigger(TriggerFlag, 0, 0, 0, 0, 1, semaphoreId_).value);  }

  __device__ void putWithSignalAndFlush(uint64_t dst, uint64_t dstOffset, uint64_t src, uint64_t srcOffset,
                                                   uint64_t size) {
    uint64_t curFifoHead = fifo_.push(
        ChannelTrigger(TriggerData | TriggerFlag | TriggerSync, dst, dstOffset, src, srcOffset, size, semaphoreId_)
            .value);
    fifo_.sync(curFifoHead);
  }

  __device__ void putWithSignalAndFlush(uint64_t dst, uint64_t src, uint64_t offset, uint64_t size) {
    putWithSignalAndFlush(dst, offset, src, offset, size);
  }

};

struct SimpleProxyChannelDeviceHandle {
  ProxyChannelDeviceHandle proxyChan_;
  uint32_t dst_;
  uint32_t src_;

  __device__  void putWithSignal(uint64_t dstOffset, uint64_t srcOffset, uint64_t size) {
    proxyChan_.putWithSignal(dst_, dstOffset, src_, srcOffset, size);
  }
  __device__ __attribute__((used))  void putWithSignal(uint64_t offset, uint64_t size) { putWithSignal(offset, offset, size); }

  __device__ void wait(int64_t maxSpinCount = 10000000) { proxyChan_.wait(maxSpinCount);  }

  __device__ void flush() { proxyChan_.flush();  }

  __device__ void put(uint64_t dstOffset, uint64_t srcOffset, uint64_t size) {
    proxyChan_.put(dst_, dstOffset, src_, srcOffset, size);
  }

  __device__ void put(uint64_t offset, uint64_t size) { put(offset, offset, size);  }

  __device__ bool poll() { return proxyChan_.poll();  }

  __device__ void signal() { proxyChan_.signal();  }

  __device__ void putWithSignalAndFlush(uint64_t dstOffset, uint64_t srcOffset, uint64_t size) {
    proxyChan_.putWithSignalAndFlush(dst_, dstOffset, src_, srcOffset, size);
  }

  __device__ void putWithSignalAndFlush(uint64_t offset, uint64_t size) {
    putWithSignalAndFlush(offset, offset, size);
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

  __device__ __attribute__((used)) int64_t createPacketPtr(uint32_t val1, uint32_t val2, uint32_t flag) {
    LL16Packet packet(make_uint2(val1, val2), flag);
    return (int64_t)&packet;
  }

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

  __device__ __attribute((used)) float4 read_float4(uint64_t index) {
    //printf("RT read float4 on %lu\n", index);
    index = index / 4;
    // printf("YYY %lu\n", index);
    // printf("YYY %lu\n", index + 1);
    // printf("YYY %lu\n", index + 2);
    // printf("YYY %lu\n", index + 3);
    return *(reinterpret_cast<float4*>(dst_) + index);
  }
  
  __device__ __attribute((used)) float read(uint64_t index) {
    //printf("RT read on %lu got %f at %lu \n", index, *(reinterpret_cast<float*>(dst_) + index), dst_);
    return *(reinterpret_cast<float*>(dst_) + index);
  }

  __device__ __attribute((used)) void write_float4(uint64_t index, const float4& v) {
    // index = index / 4;
    // printf("WWW %lu\n", index);
    // printf("WWW %lu\n", index + 1);
    // printf("WWW %lu\n", index + 2);
    // printf("WWW %lu\n", index + 3);
    *(reinterpret_cast<float4*>(dst_) + index) = v;
  }

  __device__ __attribute((used)) void writeWrapper1(uint64_t index, const float v1, const float v2, const float v3, const float v4) {
      //printf("RT write wrapper on %lu\n", index);
      float4 t; 
      t.x = v1; 
      t.y = v2; 
      t.z = v3;
      t.w = v4;

      index = index / 4;
      *(reinterpret_cast<float4*>(dst_) + index) = t;
      assert(0);
      // const float4 value = make_float4(v1, v2, v3, v4);  // Ensure correct conversion to float4
      // *(reinterpret_cast<float*>(dst_) + index) = v1;
      // *(reinterpret_cast<float*>(dst_) + index+1) = v2;
      // *(reinterpret_cast<float*>(dst_) + index+2) = v3;
      // *(reinterpret_cast<float*>(dst_) + index+3) = v4;
      // write_float4(index, t);
  }

  __device__ __attribute((used)) void writeWrapper(uint64_t index, const float v1, const float v2, const float v3, const float v4) {
    // Ensure index is aligned for float4 (multiple of 4 floats)
    if (index % 4 != 0) return; // Handle error or adjust index
    
    float4* dst_ptr = reinterpret_cast<float4*>(dst_);
    const uint64_t f4_index = index / 4; // Convert to float4 position
    // printf("Debug: f4_index = %lu, v1 = %f, v2 = %f, v3 = %f, v4 = %f\n", f4_index, v1, v2, v3, v4);
    dst_ptr[f4_index] = make_float4(v1, v2, v3, v4);
  }

__device__ __attribute((used)) void writeWrapperStride(uint32_t index, uint32_t stride, float val1, float val2, float val3, float val4) {
    float4 vec;
    vec.x = val1;
    vec.y = val2;
    vec.z = val3;
    vec.w = val4;

    index = index / 4;
    //printf("Index: %u, Stride: %u\n", index, stride);
    float4* arr_vec4 = reinterpret_cast<float4*>(dst_);
    arr_vec4[index] = vec;
    
    // #pragma unroll
    // for (int i = 0; i < 4; ++i) {
    //     arr_vec4[i * stride / 4] = vec;
    // }
}
  
  __device__ __attribute((used)) void write1(uint64_t index, const float v) {
   // printf("RT write %lu <= %f at %lu\n", index, v, dst_);
    *(reinterpret_cast<float*>(dst_) + index) = v;
    // printf("WWW %lu\n", index);
  }
  
  __device__ __attribute((used)) void write(uint32_t index, const float v) {
  //   if (index != index2) {
  //       printf("Index1: %lu, not equal to Index2: %lu\n", index, index2);
  //   }
  //   printf("III\t%lu\t%lu\t%lu\n", index, index2, index2 > index ? index2 - index : index - index2);
  //   printf("Index2: %lu\n", index2);
  //   assert(index == index2 && "Index mismatch");
  //  printf("RT write %lu <= %f at %lu\n", index, v, dst_);
    *(reinterpret_cast<float*>(dst_) + index) = v;
    // printf("WWW %lu\n", index2);
  }
  
  __device__ __attribute((used)) void write_scratch_packets(uint64_t index, float val1, float val2, int flag) {
    LL16Packet packet(make_uint2(val1, val2), flag);
    *(reinterpret_cast<LL16Packet*>(dst_) + index) = packet;
  }

  __device__ __attribute((used)) void write_packet_wrapper(uint64_t index, uint64_t packet) {
    LL16Packet* pkt = reinterpret_cast<LL16Packet*>(packet);
    write_packet(index, pkt);
  }

  __device__ __attribute((used)) void write_packet(uint64_t index, const LL16Packet *v) {
    /* printf("RT write %lu <= %f at %lu\n", index, v, dst_); */
    *(reinterpret_cast<LL16Packet*>(dst_) + index) = *v;
    // printf("WWW %lu\n", index);
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
    // printf("Triton => get %lu bytes from %lu\n", bytes, offset);
    get_helper(offset, offset, bytes, threadId, numThreads);
  }

  __device__ void putLL16Packets(void* targetPtr, uint64_t targetOffset, const void* originPtr,
                                            uint64_t originOffset, uint64_t originBytes, uint32_t threadId,
                                            uint32_t numThreads, uint32_t flag) {
    // Offsets should be aligned to 8 bytes & size should be a multiple of 8 bytes
    const uint32_t* originBase = (const uint32_t*)((const char*)originPtr + originOffset);
    LL16Packet* targetBase = (LL16Packet*)((char*)targetPtr + targetOffset);
    size_t nElem = originBytes / sizeof(uint64_t);
    for (size_t i = threadId; i < nElem; i += numThreads) {
      LL16Packet* pkt = &targetBase[i];
      pkt->write(originBase[2 * i], originBase[2 * i + 1], flag);
    }
  }

  __device__ __attribute((used)) void putPackets(uint64_t targetOffset, uint64_t originOffset, uint64_t originBytes,
                                        uint32_t threadId, uint32_t numThreads, uint32_t flag) {
    putLL16Packets(dst_, targetOffset, src_, originOffset, originBytes, threadId, numThreads, flag);
  }

};

__device__ mscclpp::DeviceSyncer DS;
__device__ uint2 foo(){
    SmChannelDeviceHandle handle;
    handle.relaxedSignal();
    handle.wait();
    handle.write(6, handle.read(3));
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
    // int64_t pkt = createPacket(1, 2, 3);
    handle.write_scratch_packets(0, 1, 2, 3);
    return packet.read(3);
}

__global__ void bar(SmChannelDeviceHandle* SmChans){
    SmChans[6].relaxedSignal();
    SmChans[9].writeWrapper(5, 1, 2, 3, 4);
    SmChans[9].writeWrapperStride(5, 1, 1, 2, 3, 4);
    float4 temp = SmChans[8].read_float4(5);
    SmChans[3].write_float4(5, make_float4(1, 2, 3, 4));
    SmChans[3].wait();
    /* int4 test = SmChans[9].read(3); */
    SmChans[8].write(7, SmChans[9].read(3));
    SmChans[8].writeWrapper(5, 1, 2, 3, 4);
    SmChans[8].get(1, 2, 3, 4);
    SmChans[8].write_scratch_packets(0, 1, 2, 3);
  
}  

} // namespace mscclpp
#endif  // MSCCLPP_SM_CHANNEL_DEVICE_HPP_
