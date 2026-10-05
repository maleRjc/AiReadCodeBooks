# Chapter 06: Collective Dispatch: Converting ncclAllReduce into Executable Tasks


上一章我们走完了 tuning 模块，知道 NCCL 会在微秒级内为一次集合通信选定 (算法, 协议, channel, warp) 组合。但选型结果本身只是一堆数字——它需要被“翻译”成 GPU kernel 能读懂的任务描述对象，才能被真正执行。本章进入 src/enqueue/enqueue.cc 的主干，回答一个核心问题：当用户调用 ncclAllReduce 时，host 侧到底发生了什么？从 ncclAllReduce 到 ncclEnqueueCheck，经过参数校验、算法/协议确定、channel 切分，最终生成 ncclInfo 与 ncclTaskColl 结构。这是全书从“用户视角”切换到“引擎视角”的关键一章。如果把 NCCL 比作一家餐厅，那么 enqueue 模块就是“前台点单系统”：用户（应用层）说“我要一份 AllReduce”，前台把它翻译成厨房（GPU kernel）能执行的工单——几号灶台、用什么锅、分几批做。没有这个翻译层，厨房根本不知道要做什么菜。

## 一、入口：ncclAllReduce 如何构造 ncclInfo

### Intuitive Architectural Model

`ncclAllReduce` 是用户直接调用的 API 函数。它的职责极其单一：**把用户传入的裸参数打包成一个 `ncclInfo` 结构体，然后交给 `ncclEnqueueCheck`**。这就像你去银行柜台办业务，柜员先把你的需求填进一张标准表单，再转交给后台系统。

如果没有这一层，每个集合通信 API 都要自己处理参数校验、group 语义、profiler 埋点——代码会重复到无法维护。

### 数据结构：ncclInfo 的内存布局

`ncclInfo` 是贯穿整个 enqueue 流程的核心载体。它的定义在 `src/include/info.h`：

[FACT:src/include/info.h:17-44]

这个结构体有 20+ 个字段，我们可以按功能分成四组：

| 字段组 | 字段 | 作用 |
|--------|------|------|
| 集合通信参数 | `coll`, `sendbuff`, `recvbuff`, `count`, `datatype`, `op`, `root` | 描述"做什么" |
| 通信域与流 | `comm`, `stream` | 描述"在哪做" |
| 算法细节 | `chunkSteps`, `sliceSteps` | 描述"怎么切分" |
| 单边操作 | `peerWinOffset`, `peerWin`, `sigIdx`, `ctx`, `flags`, `nDesc`, `signalDescs` | RMA 专用 |
| 用户配置 | `collConfig` | 从用户 config 拷贝的私有副本 |

注意 `collConfig` 的注释：**"A config copied from config passed by user so older user config can be safely accessed during synchronous host scheduling (never at launch/replay)"** [FACT:src/include/info.h:41-43]。这是一个关键设计——用户传入的 config 指针可能在 `ncclGroupEnd` 之前就被销毁，所以 NCCL 在 `ncclInfo` 里做了一份拷贝。

### Step-by-Step：ncclAllReduce 的调用链

我们以 `ncclAllReduce` 为例，追踪从用户调用到 `ncclInfo` 构造的完整路径。

**第 1 步：用户调用 ncclAllReduce。** 入口在 `src/collectives.cc`：

[FACT:src/collectives.cc:206-211]

这里做了三件事：
1. `NVTX3_FUNC_WITH_PARAMS` 打 NVTX 标记（用于 Nsight 等工具可视化）
2. 调用 `ncclAllReduceConfigImpl`，传入 `config = nullptr`
3. 返回结果

**第 2 步：ncclAllReduceConfigImpl 构造 ncclInfo。** 这是关键的一步：

[FACT:src/collectives.cc:192-202]

注意这里用了 C 风格的聚合初始化：

```c
struct ncclInfo info = {ncclFuncAllReduce, "AllReduce",
                        sendbuff, recvbuff, count, datatype, op, 0, comm, stream,
                        ALLREDUCE_CHUNKSTEPS, ALLREDUCE_SLICESTEPS};
```

字段按 `ncclInfo` 的声明顺序一一对应。`ALLREDUCE_CHUNKSTEPS` 和 `ALLREDUCE_SLICESTEPS` 定义在 `src/include/collectives.h`：

[FACT:src/include/collectives.h:19-20]

`NCCL_STEPS` 是环形缓冲区里的步数（通常为 8 或 16），所以 AllReduce 的 chunkSteps 是 `NCCL_STEPS/2`，sliceSteps 是 `NCCL_STEPS/4`。这意味着一个 chunk 包含 2 个 slice。

**第 3 步：解析用户 config。** `ncclParseCollConfig` 把用户传入的 `ncclCollConfig_t*` 解析进 `info.collConfig`。如果 `config == nullptr`，这个字段保持零初始化。

**第 4 步：交给 ncclEnqueueCheck。** 这是 enqueue 模块的真正入口。

### 设计思考：为什么用聚合初始化而不是逐字段赋值？

[INFERENCE] 聚合初始化有两个好处：一是编译器会检查字段数量是否匹配（少一个字段会警告），二是代码更紧凑。但缺点是**字段顺序必须与结构体声明严格一致**——如果有人在 `ncclInfo` 中间插入一个字段，所有聚合初始化点都会静默错位。这是 NCCL 代码里一个隐含的维护风险。

### 生产踩坑：config 生命周期

一个真实的踩坑场景：用户这样写代码：

```c
ncclCollConfig_t config = {...};
ncclAllReduceConfig(..., &config);
// config 在这里被销毁（比如是栈变量，函数返回了）
```

如果 NCCL 没有在 `ncclInfo` 里拷贝 config，那么 `ncclGroupEnd` 时访问 `info.collConfig` 就会读到已释放的内存。`src/include/info.h:41-43` 的注释正是为了说明这个设计——**config 在 task append 阶段就被解析并拷贝，之后不再依赖用户指针**。

---

## 二、ncclEnqueueCheck：参数校验与 group 语义

### Intuitive Architectural Model

`ncclEnqueueCheck` 是 enqueue 模块的"总闸门"。所有集合通信 API 最终都汇聚到这里。它的职责是：**校验参数合法性、处理 group 语义、调用 taskAppend 生成任务**。如果把它比作机场安检，那么每个 API 函数就是值机柜台——值机只是收行李，真正的安检在 `ncclEnqueueCheck`。

如果没有这一层，每个 API 都要自己写一遍参数校验和 group 处理，代码会膨胀数倍，而且容易漏掉某个校验。

### Step-by-Step：ncclEnqueueCheck 的执行流程

[FACT:src/enqueue/enqueue.cc:3478-3527]

我们逐步拆解：

**第 1 步：CommCheck 校验通信域。** `CommCheck(info->comm, info->opName, "comm")` 检查 comm 指针是否非空、是否已初始化。如果 comm 被 revoke（比如某个 rank 出错），直接返回错误：

[FACT:src/enqueue/enqueue.cc:3480-3485]

**第 2 步：处理 profiler 深度。** 如果已经在 group 内部（`profilerGroupDepth > 0`），递增深度计数。这是为了正确处理隐式的 `ncclGroupStartInternal`/`ncclGroupEndInternal` 调用。

**第 3 步：进入内部 group。** `ncclGroupStartInternal()` 是 NCCL 内部的 group 机制。**关键点**：即使用户没有显式调用 `ncclGroupStart`，NCCL 也会为每次 API 调用创建一个隐式 group。这保证了单次调用的原子性。

**第 4 步：确保 comm 就绪。** `ncclCommEnsureReady(info->comm)` 等待通信域初始化完成（比如 bootstrap 完成、连接建立）。

**第 5 步：ArgsCheck 参数校验。** 这是最复杂的校验步骤：

[FACT:src/enqueue/enqueue.cc:3497-3503]

注意 `checkMode` 的处理：如果是 `ncclCheckModeDebugGlobal`，`ArgsCheck` 会把 info 入队，等 `ncclGroupEnd` 时做全局校验（比如检查所有 rank 的 count 是否一致）。

**第 6 步：调用 taskAppend。** 这是核心转换步骤：

[FACT:src/enqueue/enqueue.cc:3513]

**第 7 步：递增 opCount。** 每次成功入队后，`comm->opCount++`。这个计数器用于匹配 send/recv 操作，也是 profiler 的时间线依据。

**第 8 步：退出 group。** `ncclGroupEndInternal()` 如果 depth 降到 0，会触发真正的 group 操作（调度、启动 kernel）。

### 并发控制：group 语义与线程安全

[INFERENCE] `ncclGroupStartInternal`/`ncclGroupEndInternal` 使用线程局部存储（TLS）来维护 group 状态。这意味着**同一个线程内的多次 API 调用会被合并成一个 group**，但不同线程的调用是独立的。这是 NCCL 支持多线程调用的基础。

一个容易踩的坑：如果用户在 `ncclGroupStart` 和 `ncclGroupEnd` 之间调用了非 NCCL 的 CUDA API（比如 `cudaMemcpy`），可能会导致 stream 顺序问题。NCCL 的 group 机制假设 group 内的操作都在同一组 stream 上。

### 错误恢复链

`ncclEnqueueCheck` 的错误处理有一个精巧的设计：

[FACT:src/enqueue/enqueue.cc:3524-3526]

如果 `taskAppend` 失败，且 comm 是非阻塞模式，会调用 `ncclCommSetAsyncError` 记录错误。这样后续的 API 调用会立即返回错误，而不是继续尝试。这是异步错误传播机制。

---

## 三、taskAppend：任务分发的十字路口

### Intuitive Architectural Model

`taskAppend` 是 enqueue 模块的"交通枢纽"。它根据 `info->coll` 的值，把任务分发到不同的处理路径：P2P、RMA、CE、或者普通集合通信。这就像一个邮局分拣中心——根据信封上的地址，把信件投到不同的邮筒。

如果没有这个分发层，所有类型的操作都要挤在一个巨大的 if-else 里，代码会难以维护。

### Step-by-Step：taskAppend 的分发逻辑

[FACT:src/enqueue/enqueue.cc:3337-3476]

**第 1 步：判断是否启用新架构。** `ncclParamEnqueueRearchEnable()` 是一个环境变量开关（默认 0）。如果启用，走 `rawTaskAppend` 路径——这是 NCCL 正在开发的新任务模型。

**第 2 步：P2P 分发。** 如果是 Send/Recv，调用 `p2pTaskAppend`：

[FACT:src/enqueue/enqueue.cc:3343-3345]

**第 3 步：RMA 分发。** 如果是 PutSignal/Signal/WaitSignal，调用 `rmaTaskAppend`：

[FACT:src/enqueue/enqueue.cc:3346-3347]

**第 4 步：空集合通信提前返回。** `if (info->count == 0) return ncclSuccess;`——count 为 0 的集合通信直接丢弃。

**第 5 步：算法选择校验。** `ncclCollConfigGetAlgMask` 校验用户传入的算法选择是否合法：

[FACT:src/enqueue/enqueue.cc:3357-3358]

**第 6 步：FP8 类型检查。** FP8 归约需要 sm90+：

[FACT:src/enqueue/enqueue.cc:3360-3366]

**第 7 步：归约操作转换。** `hostToDevRedOp` 把 host 侧的 `ncclRedOp_t` 转换成设备侧的 `ncclDevRedOpFull`：

[FACT:src/enqueue/enqueue.cc:3370-3371]

**第 8 步：单 rank 提前返回。** 如果 `comm->nRanks == 1`，直接调用 `ncclLaunchOneRank` 执行本地归约，不需要生成任务：

[FACT:src/enqueue/enqueue.cc:3373-3377]

**第 9 步：多 rank 路径。** 这是最复杂的分支，包含 CE 路由、AllToAll/Gather/Scatter 降级、以及普通集合通信：

[FACT:src/enqueue/enqueue.cc:3378-3470]

### 数据结构：ncclTaskColl 的字段

`collTaskAppend` 是生成 `ncclTaskColl` 的地方。我们看它的核心逻辑：

[FACT:src/enqueue/enqueue.cc:2757-2851]

关键字段赋值：

| 字段 | 来源 | 含义 |
|------|------|------|
| `func` | `info->coll` | 集合通信类型 |
| `sendbuff`/`recvbuff` | `info->sendbuff`/`recvbuff` | 缓冲区指针 |
| `count` | `info->count` | 元素数量 |
| `datatype` | `info->datatype` | 数据类型 |
| `trafficBytes` | `count * elementSize * ncclFuncTrafficPerByte` | 流量估算 |
| `opHost`/`opDev` | `info->op`/`opDev` | 归约操作 |
| `chunkSteps`/`sliceSteps` | `info->chunkSteps`/`sliceSteps` | 切分步数 |
| `minCTAs`/`maxCTAs`/`nvlsCTAs` | 配置解析 | 资源上限 |
| `algMask` | `ncclCollConfigGetAlgMask` | 算法选择掩码 |

注意 `trafficBytes` 的计算：

[FACT:src/enqueue/enqueue.cc:2813]

`ncclFuncTrafficPerByte` 返回每种集合通信的流量倍数：

[FACT:src/enqueue/enqueue.cc:123-134]

AllReduce 返回 2（因为要 reduce + broadcast），AllGather/ReduceScatter 返回 nRanks，其他返回 1。

### 设计思考：为什么 AllGather/Broadcast 要转成 int8？

[FACT:src/enqueue/enqueue.cc:2808-2812]

AllGather 和 Broadcast 把 count 乘以 elementSize，然后把 datatype 改成 `ncclInt8`。这是一个优化：**这两种操作不涉及归约，所以不需要关心数据类型，统一按字节处理可以简化 kernel 逻辑**。

### 生产踩坑：CTAPolicy 的解析顺序

[FACT:src/enqueue/enqueue.cc:3390-3397]

CTAPolicy 的解析有一个微妙的优先级：**env > per-call > comm**。而且 `NCCL_CTA_POLICY_ZERO` 优先于 `NCCL_CTA_POLICY_EFFICIENCY`。如果用户同时设置了这两个标志，ZERO 会生效。

一个真实的踩坑场景：用户设置了 `NCCL_CTA_POLICY=EFFICIENCY`，但发现 CE 路径没有被使用。原因是 CE 路由要求 `CTAPolicy & NCCL_CTA_POLICY_ZERO` 为真，而 EFFICIENCY 不满足这个条件。

---

## 四、ncclPrepareTasks：从任务列表到调度队列

### Intuitive Architectural Model

`ncclPrepareTasks` 是 enqueue 模块的"预处理器"。它把散乱的任务列表按 (func, op, datatype) 分桶，然后为每个桶计算算法和协议。这就像一个图书馆管理员——先把还回来的书按类别分好，再决定每类书放在哪个书架。

如果没有这一步，后续的 `scheduleCollTasksToPlan` 就要为每个任务单独计算算法，效率极低。

### Step-by-Step：ncclPrepareTasks 的分桶逻辑

[FACT:src/enqueue/enqueue.cc:423-642]

**第 1 步：Broadcast 任务转换。** 如果只有一个 broadcast peer，把 broadcast 任务转成 coll 任务：

[FACT:src/enqueue/enqueue.cc:430-461]

注意这里把 `bcastTask` 的字段拷贝到新的 `ncclTaskColl`，并计算 `trafficBytes`。然后从 `memPool_ncclTaskBcast` 释放原任务。

**第 2 步：按 (func, op, datatype) 分桶。** 任务从 sorter 出来是按 size 降序的，然后被分到 `tasksByFnOpTy` 数组：

[FACT:src/enqueue/enqueue.cc:464-487]

索引计算：`((int)task->func * ncclNumDevRedOps + (int)task->opDev.op) * ncclNumTypes + (int)task->datatype`。这是一个三维数组的线性化。

**第 3 步：聚合与算法选择。** 对每个桶，聚合大小相近的任务（4 倍以内），然后调用 `ncclGetAlgoInfo`：

[FACT:src/enqueue/enqueue.cc:503-547]

**第 4 步：按 (collnet, nvls) 分桶。** 根据算法类型，把任务分到 `collBins[2][2]`：

[FACT:src/enqueue/enqueue.cc:517-544]

**第 5 步：拼接最终队列。** 把四个桶拼接成 `planner->collTaskQueue`：

[FACT:src/enqueue/enqueue.cc:553-557]

### 数据结构：ncclTaskCollSorter

`ncclTaskCollSorter` 是一个按 `trafficBytes` 排序的插入式排序器。`ncclTaskCollSorterInsert` 把任务插入到正确位置，`ncclTaskCollSorterDequeueAll` 按顺序取出所有任务。

[INFERENCE] 这个排序器的设计动机是：**大任务优先调度**。因为大任务的传输时间长，先启动它们可以更好地重叠计算和通信。

### 并发控制：runtimeConn 与连接建立

[FACT:src/enqueue/enqueue.cc:572-583]

如果 `comm->runtimeConn` 为真（运行时连接模式），且某个算法的 channel 还没初始化，就标记 `algoNeedConnect`。这会在后续触发连接建立。

### 生产踩坑：聚合的边界条件

[FACT:src/enqueue/enqueue.cc:507-508]

聚合条件是 `aggEnd->trafficBytes < 4 * aggBeg->trafficBytes`，且两个任务都不设置 `aggIsolate`。如果用户设置了 per-call config（比如 `maxCTAs`），`aggIsolate` 会被设为 true，这个任务就不会被聚合。

一个真实的踩坑场景：用户为某个 AllReduce 设置了 `maxCTAs=4`，期望它只用 4 个 CTA。但由于聚合逻辑，这个任务可能和相邻任务合并，导致实际使用的 CTA 数量不符合预期。解决方案是设置 `aggIsolate`——NCCL 在 `collTaskAppend` 里已经处理了这一点：

[FACT:src/enqueue/enqueue.cc:2821-2822]

---

## 五、scheduleCollTasksToPlan：channel 切分与预算控制

### Intuitive Architectural Model

`scheduleCollTasksToPlan` 是 enqueue 模块的"调度器"。它把任务分配到具体的 channel，并计算每个 channel 的数据切分。这就像一个工厂的排产系统——决定每条生产线做什么、做多少。

如果没有这一步，GPU kernel 就不知道自己要处理哪部分数据。

### Step-by-Step：channel 切分算法

[FACT:src/enqueue/enqueue.cc:644-947]

**第 1 步：预算估算。** 先估算能放进这个 plan 的任务数量：

[FACT:src/enqueue/enqueue.cc:648-689]

`ncclTestBudget` 检查工作字节数是否超出预算：

[FACT:src/enqueue/enqueue.cc:343-349]

**第 2 步：计算每个 channel 的流量。** 根据 kind（collnet/nvls）计算 `trafficPerChannel`：

[FACT:src/enqueue/enqueue.cc:701-707]

**第 3 步：Collnet 路径。** 如果是 collnet 算法，channel 分配比较简单：

[FACT:src/enqueue/enqueue.cc:709-739]

**第 4 步：普通路径的 cell 切分。** 这是最复杂的部分。NCCL 把数据切成 "cell"，每个 cell 是一个最小传输单元：

[FACT:src/enqueue/enqueue.cc:740-845]

关键变量：
- `cellSize`：每个 cell 的字节数，至少 `MinTrafficPerChannel`（32KB）
- `cells`：总 cell 数
- `cellsPerChannel`：每个 channel 处理的 cell 数
- `cellsLo`/`cellsHi`：首尾 channel 的 cell 数（可能不满）

**第 5 步：计算 chunkGrains。** 对每个 channel 段调用 `calcCollChunking`：

[FACT:src/enqueue/enqueue.cc:811-825]

**第 6 步：生成 proxyOp。** 为每个 channel 生成 proxy 操作：

[FACT:src/enqueue/enqueue.cc:844-894]

### 数据结构：ncclDevWorkColl

`ncclDevWorkColl` 是设备侧的工作描述符。它的关键字段：

| 字段 | 含义 |
|------|------|
| `sendbuff`/`recvbuff` | 缓冲区指针 |
| `channelLo`/`channelHi` | channel 范围 |
| `cbd.countLo`/`countMid`/`countHi` | 各段元素数 |
| `cbd.chunkGrainsLo`/`Mid`/`Hi` | 各段 chunk 粒度 |
| `direct` | 直接标志 |

### 并发控制：channelMask 的位运算

[FACT:src/enqueue/enqueue.cc:897]

这行代码用位运算设置 channelMask：`(2ull << channelHi) - (1ull << channelLo)`。比如 channelLo=2, channelHi=5，结果是 `(2<<5) - (1<<2) = 64 - 4 = 60 = 0b111100`，即 bit 2-5 被设置。

### 生产踩坑：预算溢出

[FACT:src/enqueue/enqueue.cc:792-794]

如果预算不够，直接返回 `ncclSuccess`，让外层循环创建新的 plan。这是一个优雅的降级策略——**不报错，只是分批处理**。

一个真实的踩坑场景：如果 `NCCL_WORK_FIFO_BYTES` 设置得太小，会导致每个 plan 只能容纳很少的任务，增加 kernel 启动次数，降低性能。

---

## 六、finishPlan：从任务到 kernel 参数

### Intuitive Architectural Model

`finishPlan` 是 enqueue 模块的"打包器"。它把任务、batch、proxyOp 打包成 kernel 能直接读取的参数结构。这就像快递打包——把散件装进箱子，贴上运单，等待发货。

### Step-by-Step：finishPlan 的打包逻辑

[FACT:src/enqueue/enqueue.cc:236-330]

**第 1 步：决定存储类型。** 如果所有工作都能放进 kernel args，用 `ncclDevWorkStorageTypeArgs`：

[FACT:src/enqueue/enqueue.cc:244-250]

**第 2 步：分配 kernelArgs。** 从内存栈分配：

[FACT:src/enqueue/enqueue.cc:251-255]

**第 3 步：Round-robin 放置 batch。** 每个 channel 的第一个 batch 必须放在 `batchZero[blockIdx.x]`：

[FACT:src/enqueue/enqueue.cc:257-280]

**第 4 步：合并 proxyOp 队列。** 按 opCount 归并排序：

[FACT:src/enqueue/enqueue.cc:282-329]

### 数据结构：ncclDevKernelArgs

`ncclDevKernelArgs` 是传给 kernel 的参数结构。它包含：
- `comm`：设备侧通信器
- `channelMask`：channel 位掩码
- `workStorageType`：工作存储类型
- `workBuf`：工作缓冲区指针
- `workMask`：工作缓冲区掩码

### 生产踩坑：batch 顺序

[FACT:src/enqueue/enqueue.cc:257-259]

注释说得很清楚："The first batch for each channel must be located at batchZero[blockIdx.x]"。如果这个顺序错了，kernel 会读到错误的 batch，导致数据损坏。

---

## 本章Summary

本章我们追踪了从 `ncclAllReduce` 到 `ncclTaskColl` 的完整路径：

1. **ncclAllReduce** 构造 `ncclInfo`，打包用户参数
2. **ncclEnqueueCheck** 校验参数、处理 group 语义
3. **taskAppend** 根据操作类型分发到不同路径
4. **collTaskAppend** 生成 `ncclTaskColl`，解析配置
5. **ncclPrepareTasks** 按 (func, op, datatype) 分桶，计算算法
6. **scheduleCollTasksToPlan** 切分 channel，生成 `ncclDevWorkColl`
7. **finishPlan** 打包成 kernel 参数

关键设计思想：
- **分层解耦**：每个函数只做一件事，通过 `ncclInfo` 和 `ncclTaskColl` 传递状态
- **预算控制**：通过 `ncclTestBudget` 控制每个 plan 的大小
- **聚合优化**：大小相近的任务会被聚合，减少 kernel 启动次数
- **配置优先级**：env > per-call > comm

下一章我们将进入 `task_sched`，看 NCCL 如何编排多 channel 多 kernel 的执行顺序。

## 本章思考与自测

<details><summary>Q1: 如果把 `collTaskAppend` 中的 `aggIsolate` 判断去掉（即 `src/enqueue/enqueue.cc:2821-2822` 永远返回 false），在什么场景下会导致用户设置的 `maxCTAs` 失效？为什么？</summary>

**参考解析**：`aggIsolate` 的作用是标记"这个任务不能被聚合"。如果去掉这个判断，设置了 per-call config 的任务会和相邻任务合并。在 `ncclPrepareTasks` 的聚合循环中（`src/enqueue/enqueue.cc:507-508`），聚合条件是 `aggEnd->trafficBytes < 4 * aggBeg->trafficBytes && !aggBeg->aggIsolate && !aggEnd->aggIsolate`。如果 `aggIsolate` 永远为 false，那么即使任务设置了 `maxCTAs=4`，它也可能和一个 `maxCTAs=32` 的任务合并。合并后的 `agg` 会取两者的某种组合（具体取决于 `ncclGetAlgoInfo` 的实现），导致实际使用的 CTA 数量不符合用户预期。

更严重的是，在 `scheduleCollTasksToPlan` 中（`src/enqueue/enqueue.cc:665-666`），`taskAggIsolate` 用于确保配置了 per-call 资源的任务单独占一个 plan。如果这个判断失效，多个任务会共享 plan 的 channel 预算，导致资源分配不符合预期。

</details>

<details><summary>Q2: 在 `ncclEnqueueCheck` 中，如果 `ncclGroupEndInternal()` 返回错误（比如某个 rank 的 ArgsCheck 失败），但 `taskAppend` 已经成功执行了，会发生什么？NCCL 如何保证状态一致性？</summary>

**参考解析**：看 `src/enqueue/enqueue.cc:3513-3519` 的控制流：

```c
NCCLCHECKGOTO(taskAppend(info->comm, info), ret, fail);
info->comm->opCount++;
exit:
  if (devOld != -1) CUDACHECK(cudaSetDevice(devOld));
  ncclGroupErrCheck(ret);
  NCCLCHECK(ncclGroupEndInternal());
```

如果 `taskAppend` 成功但 `ncclGroupEndInternal` 失败，`opCount` 已经递增了。这会导致后续操作的 opCount 与对端不匹配，可能触发 hang。

NCCL 的处理方式是：`ncclGroupErrCheck(ret)` 会检查是否有错误，如果有，会设置 comm 的错误状态。后续的 API 调用会通过 `ncclCommGetAsyncError` 检测到这个错误并立即返回。这是一种"快速失败"策略——一旦出错，整个 comm 进入错误状态，不再尝试恢复。

在生产环境中，这意味着一旦出现 group 错误，用户需要销毁并重建 communicator。

</details>

<details><summary>Q3: `scheduleCollTasksToPlan` 中的 cell 切分算法（`src/enqueue/enqueue.cc:740-845`）有一个边界条件：当 `cellsLo == 0` 时，会跳过最少的 channel。如果这个跳过逻辑有 bug（比如 `channelId` 没有正确递增），会导致什么后果？</summary>

**参考解析**：看 `src/enqueue/enqueue.cc:770-780`：

```c
if (cellsLo == 0) {
  // Least channel skipped. Make the next channel the new least.
  channelId += 1;
  if (nMidChannels == 0) {
    cellsLo = cellsHi;
    cellsHi = 0;
  } else {
    cellsLo = cellsPerChannel;
    nMidChannels -= 1;
  }
}
```

如果 `channelId` 没有正确递增，那么下一个任务会从错误的 channel 开始分配。这会导致：
1. **channel 重叠**：两个任务可能分配到同一个 channel 的同一段数据
2. **数据损坏**：kernel 会重复处理或遗漏数据
3. **性能下降**：channel 负载不均衡

更隐蔽的是，这种 bug 可能只在特定消息大小下触发（当 `cellsLo == 0` 时），难以复现。NCCL 通过 `plan->channelMask |= (2ull << devWork->channelHi) - (1ull << devWork->channelLo)` 来跟踪已使用的 channel，但这只是记录，不能防止重叠。

</details>

至此，我们已经看清 ncclAllReduce 如何从用户调用变成一串可执行的 kernel 任务：参数校验、算法/协议确定、channel 切分，最终生成 ncclInfo 与 ncclTaskColl。但任务被创建出来只是第一步——它们还需要被调度到多个 channel 上，生成 kernel 启动参数，并在 group 语义下处理批量提交与依赖排序。下一章将深入 src/enqueue/task_sched 与 src/enqueue/task_prep，回答“为什么一次 AllReduce 会启动多个 kernel，它们之间的顺序和依赖是怎么保证的”，同时揭示 src/group.cc 中 ncclGroupStart/ncclGroupEnd 如何把多次 API 调用合并成一次提交。