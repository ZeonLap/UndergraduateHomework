$计14\ 汪隽立\ 2021012957$

# Lab1

## 基本公式

$$
a_0 = \frac{1}{2\pi} \int_0^{2\pi}f(t)dt \\
a_n = \frac{1}{\pi} \int_0^{2\pi}f(t)\sin(nt)dt \\
b_n = \frac{1}{\pi} \int_0^{2\pi}f(t)\cos(nt)dt \\
$$



## 实现定积分

主要使用了`scipy.integrate`库来实现定积分：

```python
def fourier_coefficient(n):
    if n == 0:
        return quad(function, 0, 2 * math.pi)[0] / (2 * math.pi)
    elif n % 2 == 1:
        m = (n + 1) / 2
        return quad(lambda t: function(t) * math.sin(m * t), 0, 2 * math.pi)[0] / math.pi
    elif n % 2 == 0:
        m = n / 2
        return quad(lambda t: function(t) * math.cos(m * t), 0, 2 * math.pi)[0] / math.pi
    else:
        raise Exception("n must be an integer")
```

## 方波与半圆波

```python
def square_wave(t):
    return 1 if math.sin(t) > 0 else 0
```

```python
def semi_circle_wave(t):
    return math.sqrt(math.pi ** 2 - (t - math.pi) ** 2) if 0 <= t <= 2 * math.pi else semi_circle_wave(t - 2 * math.pi)
```

## 代码说明

本机的实验环境为Mac，所以在使用`ffmpeg`包时需要额外更改环境变量：

```python
import os
os.environ["IMAGEIO_FFMPEG_EXE"] = "your ffmpeg path"
```

另外，为了方便批量测试，故使用了`argparse`包和批量处理脚本，一并上传至网络学堂。