/*
 * 分配策略：显式空闲列表 + 最佳适配 + 分离适配 + 优化的mm_realloc
 *
 * 内存块的数据结构：（每一行代表一个word，从上往下为低地址到高地址）
 * |头部(size | allocated)       |
 * |指向下一个空闲块的指针          |
 * |指向上一个空闲块的指针          |
 * |负载                         |
 * |脚部(size | allocated)       | 
 * 
 * 其他细节请见REPORT.md
 */
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "memlib.h"
#include "mm.h"

/*********************************************************
 * NOTE TO STUDENTS: Before you do anything else, please
 * provide your team information in the following struct.
 ********************************************************/
team_t team = {
    /* Team name */
    "LTNS XD",
    /* First member's full name */
    "Junli Wang",
    /* First member's email address */
    "2021012957",
    /* Second member's full name (leave blank if none) */
    "",
    /* Second member's email address (leave blank if none) */
    ""};

/* 16 bytes alignment */
#define ALIGNMENT         16

/* rounds up to the nearest multiple of ALIGNMENT */
#define ALIGN(size)       (((size) + (ALIGNMENT - 1)) & ~(ALIGNMENT - 1))

#define SIZE_T_SIZE       (ALIGN(sizeof(size_t)))

#define WSIZE             8
#define DSIZE             16
#define CHUNKSIZE         928

#define MAX(x, y)         ((x) > (y) ? (x) : (y))

/* 将大小与占用信息整合到一个word中 */
#define PACK(size, alloc) ((size) | (alloc))

/* 对位于地址p的一个word进行读写 */
#define GET(p)            (*(unsigned long *)(p))
#define PUT(p, val)       (*(unsigned long *)(p) = (val))

/* 获取指针p的首字信息 */
#define GET_SIZE(p)       (GET(p) & ~0xf)
#define GET_ALLOC(p)      (GET(p) & 0x1)

/* 显式空闲列表：获取一个空闲块的pred或succ */
#define SUCC_FREE(bp)     (GET(bp))
#define PRED_FREE(bp)     (GET((char *)(bp + WSIZE)))

/* 设置空闲块的pred或succ */
#define SET_SUCC(bp, val) PUT(bp, (unsigned long)val)
#define SET_PRED(bp, val) PUT((char *)(bp + WSIZE), (unsigned long)val)

/* 计算一个块的header或footer地址 */
#define HDRP(bp)          ((char *)(bp) - WSIZE)
#define FTRP(bp)          ((char *)(bp) + GET_SIZE(HDRP(bp)) - DSIZE)

/* 计算前一个或后一个块的地址 */
#define NEXT_BLKP(bp)     ((char *)(bp) + GET_SIZE(((char *)(bp) - WSIZE)))
#define PREV_BLKP(bp)     ((char *)(bp) - GET_SIZE(((char *)(bp) - DSIZE)))    

/* 分离适配：类别总数 */
#define CATEGORIES        13

/* 指向序言块的指针 */
static char *heap_listp;

/* 指向空闲列表超头部的指针 */
static char *free_listp;

/* 针对binary数据的设计：_rand */
static size_t _rand;

#pragma region 函数注册区

/* 扩展堆 */
static void *extend_heap(size_t words);

/* 合并空闲块 */
static void *coalesce(void *);

/* 搜索空闲块 */
static void *find_fit(size_t asize);

/* 放置请求块 */
static void *place(void *bp, size_t asize);

/* 显式空闲链表：插入 */
static void freelist_insert(void *bp, size_t size);

/* 显式空闲列表：删除 */
static void freelist_remove(void *bp);

/* Mapping size to the corresponding category */
static int size2offset(size_t size);

#pragma endregion

int size2offset(size_t size) {
  if (size <= 32) {
    return 0;
  }
  else if (size <= 64) {
    return 1;
  }
  else if (size <= 128) {
    return 2;
  }
  else if (size <= 192) {
    return 3;
  }
  else if (size <= 256) {
    return 4;
  }
  else if (size <= 320) {
    return 5;
  }
  else if (size <= 384) {
    return 6;
  }
  else if (size <= 448) {
    return 7;
  }
  else if (size <= 512) {
    return 8;
  }
  else if (size <= 1024) {
    return 9;
  }
  else if (size <= 2048) {
    return 10;
  }
  else if (size <= 4096) {
    return 11;
  }
  else {
    return 12;
  }
}

static void *extend_heap(size_t words) {
  /* 参考CSAPP：Page600 */
  char* bp;
  size_t size;

  /* 页数向偶数取上整 */
  size = (words % 2) ? (words + 1) * WSIZE : words * WSIZE;
  if ((long)(bp = mem_sbrk(size)) == -1)
    return NULL;
  
  /* 更新结尾块信息 */
  PUT(HDRP(bp), PACK(size, 0));
  PUT(FTRP(bp), PACK(size, 0));
  PUT(HDRP(NEXT_BLKP(bp)), PACK(0, 1));

  /* 合并空闲块 */
  return coalesce(bp);
}

static void *coalesce(void *bp) {
  size_t prev_alloc = GET_ALLOC(FTRP(PREV_BLKP(bp)));
  size_t next_alloc = GET_ALLOC(HDRP(NEXT_BLKP(bp)));
  size_t size = GET_SIZE(HDRP(bp));

  void *prev = PREV_BLKP(bp);
  void *next = NEXT_BLKP(bp);

  if (prev_alloc && next_alloc) { /* Case 1 */
    freelist_insert(bp, size);
  }

  else if (prev_alloc && !next_alloc) { /* Case 2 */
    size += GET_SIZE(HDRP(NEXT_BLKP(bp)));
    PUT(HDRP(bp), PACK(size, 0));
    PUT(FTRP(bp), PACK(size, 0));
    freelist_remove(next);
    freelist_insert(bp, size);
  }

  else if (!prev_alloc && next_alloc) { /* Case 3 */
    size += GET_SIZE(HDRP(PREV_BLKP(bp)));
    PUT(FTRP(bp), PACK(size, 0));
    PUT(HDRP(PREV_BLKP(bp)), PACK(size, 0));
    freelist_remove(prev);
    bp = prev;
    freelist_insert(bp, size);
  }

  else { /* Case 4 */
    size += GET_SIZE(HDRP(PREV_BLKP(bp))) + GET_SIZE(FTRP(NEXT_BLKP(bp)));
    PUT(HDRP(PREV_BLKP(bp)), PACK(size, 0));
    PUT(FTRP(NEXT_BLKP(bp)), PACK(size, 0));
    freelist_remove(prev);
    freelist_remove(next);
    bp = prev;
    freelist_insert(bp, size);
  }
  return bp;
}

static void *find_fit(size_t asize) {
  int offset = size2offset(asize);
  for (int i = offset; i < CATEGORIES; i++) { /* 从当前的大小类开始寻找 */
    void *bp = free_listp + i * WSIZE;
    for (bp = SUCC_FREE(bp); bp != NULL; bp = SUCC_FREE(bp)) {
      if (asize <= GET_SIZE(HDRP(bp))) { /* 显式空闲列表：大小符合的块一定满足需求 */
        return bp;
      }
    }
  }
  return NULL;
}

static void *place(void *bp, size_t asize) {
  size_t csize = GET_SIZE(HDRP(bp));
  void *new_ptr = bp;

  freelist_remove(bp);
  if ((csize - asize) >= (2 * DSIZE)) { /* 当剩余部分的大小等于或超出最小块大小时，进行分割 */
    /* 针对binary.rep */
    if (_rand == 0) { /* 空闲块在后 */
      PUT(HDRP(bp), PACK(asize, 1));
      PUT(FTRP(bp), PACK(asize, 1));
      PUT(HDRP(NEXT_BLKP(bp)), PACK(csize - asize, 0));
      PUT(FTRP(NEXT_BLKP(bp)), PACK(csize - asize, 0));
      coalesce(NEXT_BLKP(bp));
      _rand = 1;
    }
    else { /* 空闲块在前 */
      PUT(HDRP(bp), PACK(csize - asize, 0));
      PUT(FTRP(bp), PACK(csize - asize, 0));
      new_ptr = NEXT_BLKP(bp);
      PUT(HDRP(new_ptr), PACK(asize, 1));
      PUT(FTRP(new_ptr), PACK(asize, 1));
      coalesce(bp);
      _rand = 0;
    }
  }
  else {
    PUT(HDRP(bp), PACK(csize, 1));
    PUT(FTRP(bp), PACK(csize, 1));
  }
  return new_ptr;
}

static void freelist_insert(void *bp, size_t size) {
  int offset = size2offset(size);
  void *header = free_listp + offset * WSIZE;
  void *prev = header;
  void *next = SUCC_FREE(header);

  /* bestfit : 找到插入的合适位置 */
  for ( ; next != NULL; prev = next, next = SUCC_FREE(next)) {
    if (size <= GET_SIZE(HDRP(next)))
      break;
  }
  SET_SUCC(prev, bp);
  SET_PRED(bp, prev);
  SET_SUCC(bp, next);
  if (next) {
    SET_PRED(next, bp);
  }
}

static void freelist_remove(void *bp) {
  void *pred = PRED_FREE(bp);
  void *succ = SUCC_FREE(bp);
  SET_SUCC(pred, succ);
  if (succ != NULL)
    SET_PRED(succ, pred);
}

/*
 * mm_init -- 初始化堆结构
 * 1. 分离的空闲链表的头部
 * 2. 序言块
 * 3. 结尾块
 */
int mm_init(void) { 
  _rand = 0;

  if ((heap_listp = mem_sbrk((CATEGORIES + 5) * WSIZE)) == (void *)-1) {
    return -1;
  }

  PUT(heap_listp, 0);
  heap_listp += WSIZE;
  free_listp = heap_listp;
  for (int i = 0; i < CATEGORIES; i++) {
    PUT(free_listp + i * WSIZE, 0);
  }
  /* 对齐 */
  PUT(heap_listp + 13 * WSIZE, 0);
  /* 序言块 */
  PUT(heap_listp + 14 * WSIZE, PACK(DSIZE, 1));
  PUT(heap_listp + 15 * WSIZE, PACK(DSIZE, 1));
  /* 结尾块 */
  PUT(heap_listp + 16 * WSIZE, PACK(0, 1));
  
  return 0;  
}

/*
 * mm_malloc -- 为请求分配空间
 * 完成的工作：
 * 1. 将空间包装、对齐
 * 2. 寻找合适的位置、安置
 */
void *mm_malloc(size_t size) {
  size_t asize;
  size_t extendsize;
  char *bp;

  if (size == 0)
    return NULL;
  
  if (size <= DSIZE) /* 最小块（32 Bytes） */
    asize = 2 * DSIZE;
  else 
    asize = ALIGN(size + DSIZE);

  if ((bp = find_fit(asize)) != NULL) {
    return place(bp, asize);
  }

  extendsize = MAX(asize, CHUNKSIZE);
  if ((bp = extend_heap(extendsize / WSIZE)) == NULL)
    return NULL;
  return place(bp, asize);
}

/*
 * mm_free - 释放目标块的内存
 * 目标块的header与footer，并将其插入回空闲链表中
 */
void mm_free(void *ptr) {
  size_t size = GET_SIZE(HDRP(ptr));

  PUT(HDRP(ptr), PACK(size, 0));
  PUT(FTRP(ptr), PACK(size, 0));

  coalesce(ptr);
}

/*
 * mm_realloc - 重新分配块的大小
 * 优化：检查是否能与邻近的前后两块进行合并
 */
void *mm_realloc(void *ptr, size_t size) {
  /* 简单调用mm_malloc与mm_free的情况 */
  if (ptr == NULL)
    return mm_malloc(size);

  if (size == 0) {
    mm_free(ptr);
    return NULL;
  }

  /* ptr != NULL && size > 0 */
  void *newptr;
  void *prev;
  void *next;
  size_t asize;
  size_t old_size = GET_SIZE(HDRP(ptr));
  size_t prev_size = GET_SIZE(HDRP(PREV_BLKP(ptr)));
  size_t next_size = GET_SIZE(HDRP(NEXT_BLKP(ptr)));
  size_t sum_size = old_size + prev_size + next_size;

  if (size <= DSIZE)
    asize = 2 * DSIZE;
  else 
    asize = ALIGN(size + DSIZE);

  if (asize <= old_size) { /* 新块大小比原先要小 */
    PUT(HDRP(ptr), PACK(old_size, 1));
    PUT(FTRP(ptr), PACK(old_size, 1));
    return ptr;
  }
  else { /* 新块的大小比原先要大 */
    prev = PREV_BLKP(ptr);
    next = NEXT_BLKP(ptr);
    if (GET_SIZE(HDRP(next)) == 0) { /* 当前块位于堆顶 */
      size_t extendsize = asize - old_size;
      if ((long)(mem_sbrk(extendsize)) == -1) { /* 堆顶空间不足以使用 */
          newptr = mm_malloc(size);
          memmove(newptr, ptr, old_size - WSIZE);
          mm_free(ptr);
          return newptr;
      }
      PUT(HDRP(ptr), PACK(asize, 1));
      PUT(FTRP(ptr), PACK(asize, 1));
      PUT(HDRP(NEXT_BLKP(ptr)), PACK(0, 1));
      return ptr;
    }
    else if (!GET_ALLOC(HDRP(next)) && old_size + next_size >= asize) { /* 与下一个块合并 */
      freelist_remove(next);

      PUT(HDRP(ptr), PACK(old_size + next_size, 1));
      PUT(FTRP(ptr), PACK(old_size + next_size, 1));        
      return ptr;
    }
    else if (!GET_ALLOC(HDRP(prev)) && old_size + prev_size >= asize) { /* 与前一个块合并 */
      freelist_remove(prev);

      PUT(HDRP(prev), PACK(old_size + prev_size, 1));
      PUT(FTRP(prev), PACK(old_size + prev_size, 1));        
      memmove(prev, ptr, old_size - WSIZE);
      return prev;
    }
    else if (!GET_ALLOC(HDRP(prev)) && !GET_ALLOC(HDRP(next)) && sum_size >= asize) { /* 与前后两个块一同合并 */
      freelist_remove(prev);
      freelist_remove(next);

      PUT(HDRP(prev), PACK(sum_size, 1));
      PUT(FTRP(prev), PACK(sum_size, 1));
      memmove(prev, ptr, old_size - WSIZE);
      return prev;       
    }
    /* 合并块的大小不足以使用 */
    newptr = mm_malloc(size);
    memmove(newptr, ptr, old_size - WSIZE);
    mm_free(ptr);
    return newptr;
  }
}
