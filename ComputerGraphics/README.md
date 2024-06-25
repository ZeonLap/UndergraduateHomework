# 使用方式

使用的代码框架完全基于小作业。由于中间使用了服务器渲染，而服务器现在到期了，所以有些testcase好像丢失了，导致文件夹中的testcase文件可能不完全。

由于最后运行的环境是自己的Macbook，所以关于omp库的使用需要修改。具体为：

- 修改CMakeLists.txt，删除以下三行(g++应当是自带openmp库的)

```
SET(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I/opt/homebrew/Cellar/libomp/16.0.6/include")
SET(OpenMP_CXX_LIB_NAMES "libomp")
SET(OpenMP_libomp_LIBRARY "/opt/homebrew/Cellar/libomp/16.0.6/lib/libomp.dylib")
```

- 将smallpt.hpp中的

```C++
#include </opt/homebrew/Cellar/libomp/16.0.6/include/omp.h>
```

改为

```C++
#include <omp.h>
```

(因为本地是用homebrew装的omp)

# 图片结果

在Result文件夹下。