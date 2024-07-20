$计14\ 汪隽立\ 2021012957$

# HW2

## 6. “三权”间的权限控制

新的网络拓扑图如下所示：

<img src="/Users/wangjuanli/Codefield/2023Fall/NetworkSecurity/hw2/REPORT/image-20231210225227461.png" alt="image-20231210225227461" style="zoom:33%;" />

各终端设备的IP地址及权限级别如下表所示：

| 设备名称 | 部门       | 备注     | IP          | Gateway     |
| -------- | ---------- | -------- | ----------- | ----------- |
| PC1      | 元老院     | 领导人   | 192.168.1.2 | 192.168.1.1 |
| Laptop1  | 元老院     | 联络人   | 192.168.1.4 | 192.168.1.1 |
| PC4      | 元老院     | /        | 192.168.1.5 | 192.168.1.1 |
| Server1  | 元老院     | 机密管理 | 192.168.4.2 | 192.168.4.1 |
| Laptop2  | 执政官首府 | 领导人   | 192.168.2.3 | 192.168.2.1 |
| PC2      | 执政官首府 | 联络人   | 192.168.2.2 | 192.168.2.1 |
| Laptop4  | 执政官首府 | /        | 192.168.2.4 | 192.168.2.1 |
| PC3      | 部族会议所 | 领导人   | 192.168.3.2 | 192.168.3.1 |
| Laptop3  | 部族会议所 | 联络人   | 192.168.3.3 | 192.168.3.1 |
| PC5      | 部族会议所 | /        | 192.168.3.4 | 192.168.3.1 |

需要满足的访问权限为：

1. 各权力机构内部所有成员可以相互通信

2. 权力机构之间的相互通信只能通过联络人完成
3. 领导人之间可以相互通信
4. 只有PC1可以访问Server1

采用如下的思路实现：

- 对于1，各权力机构内部的成员通过交换机进行通信
- 对于2，在每个路由器，增加它对于它的子网的**Outbound ACL**：
  - **permit"src为另外两个子网的联络人"的情况**
  - **permit"dst为当前子网的联络人"的情况**
- 对于2，在每个路由器，增加它对于它的子网的**Inbound ACL**：
  - **permit"src为当前子网的联络人"的情况**
  - **permit"dst为另外两个子网的联络人"的情况**

- 对于3，在每个路由器，增加它对于它的子网的**Outbound ACL**
  - **permit"src为另外两个子网的领导人，dst为当前子网的领导人"的情况**
- 对于3，在每个路由器，增加它对于它的子网的**Inbound ACL**：
  - **permit"src为当前子网的领导人，dst为另外两个领导人"的情况**
- 对于4，对router1进行特殊处理：
  - 对于192.168.1.0子网的端口，增加Outbound ACL，permit"src为192.168.4.2，dst为192.168.1.2"的情况
  - 对于192.168.4.0子网的端口，增加Outbound ACL，permit "src为192.168.1.2，dst为192.168.4.2"的情况

采取“先特殊后一般”的思想，路由器设置的ACL（以Router1为例）为：

```
(config)# access-list 101 permit ip host 192.168.2.3 host 192.168.1.2
(config)# access-list 101 permit ip host 192.168.3.2 host 192.168.1.2
(config)# access-list 101 permit ip host 192.168.2.2 192.168.1.0 0.0.0.255
(config)# access-list 101 permit ip host 192.168.3.3 192.168.1.0 0.0.0.255
(config)# access-list 101 permit ip any host 192.168.1.4
(config)# access-list 101 permit ip host 192.168.4.2 host 192.168.1.2

(config-if)# ip access-group 101 out

(config)# access-list 103 permit ip host 192.168.1.2 host 192.168.2.3
(config)# access-list 103 permit ip host 192.168.1.2 host 192.168.3.2
(config)# access-list 103 permit ip host 192.168.1.4 any
(config)# access-list 103 permit ip 192.168.1.0 0.0.0.255 host 192.168.2.2
(config)# access-list 103 permit ip 192.168.1.0 0.0.0.255 host 192.168.3.3
(config)# access-list 103 permit ip host 192.168.1.2 host 192.168.4.2

(config-if)# ip access-group 103 in

(config)# access-list 102 permit ip host 192.168.1.2 host 192.168.4.2

(config-if)# ip access-group 102 out
```

Router2，Router3的设置类似，只不过不需要处理有关192.168.4.0的内容。

- Router2（可以先忽略ICMP的部分）

<img src="/Users/wangjuanli/Codefield/2023Fall/NetworkSecurity/hw2/REPORT/image-20231211033117288.png" alt="image-20231211033117288" style="zoom:33%;" />

- Router3:

<img src="/Users/wangjuanli/Codefield/2023Fall/NetworkSecurity/hw2/REPORT/image-20231211034151361.png" alt="image-20231211034151361" style="zoom:33%;" />

下面是结果展示：

- 192.168.1.2（凯撒）：

<img src="/Users/wangjuanli/Codefield/2023Fall/NetworkSecurity/hw2/REPORT/image-20231211035234875.png" alt="image-20231211035234875" style="zoom:33%;" />

- 192.168.1.4（子网1的联络人）：

<img src="/Users/wangjuanli/Codefield/2023Fall/NetworkSecurity/hw2/REPORT/image-20231211035335479.png" alt="image-20231211035335479" style="zoom:33%;" />

- 192.168.1.5（子网1的普通人）：

<img src="/Users/wangjuanli/Codefield/2023Fall/NetworkSecurity/hw2/REPORT/image-20231211035500960.png" alt="image-20231211035500960" style="zoom:33%;" />

## 7. 凯撒给予的最高权限

只有PC1可以随意地ping其他设备，我们在已有的ACL上进行修改：

- Router1:

```
(config)# access-list 103 permit icmp host 192.168.1.2 any
```

- Router2:

```
(config)# access-list 101 permit icmp host 192.168.1.2 any
```

- Router3:

```
(config)# access-list 101 permit icmp host 192.168.1.2 any
```

同时，我们设置CBAC如下：

- Router1的Fa0/1（连接他自己的子网），由于对于PC1来说，ICMP报文是进入Fa0/1，所以最后我们要用in

<img src="/Users/wangjuanli/Codefield/2023Fall/NetworkSecurity/hw2/REPORT/image-20231211040459997.png" alt="image-20231211040459997" style="zoom:33%;" />

- 对其他两个路由器则是out

<img src="/Users/wangjuanli/Codefield/2023Fall/NetworkSecurity/hw2/REPORT/image-20231211040828115.png" alt="image-20231211040828115" style="zoom:33%;" />

- 实验结果：

192.168.1.2现在能ping任何地方

<img src="/Users/wangjuanli/Codefield/2023Fall/NetworkSecurity/hw2/REPORT/image-20231211040902645.png" alt="image-20231211040902645" style="zoom:33%;" />

但别人不一定能ping到它

<img src="/Users/wangjuanli/Codefield/2023Fall/NetworkSecurity/hw2/REPORT/image-20231211040948237.png" alt="image-20231211040948237" style="zoom:33%;" />

（另外，在这次试验中我发现Ethernet接口是没有CBAC功能的。我不得不换了一个模块（NM-2FE2W））。

## 8. 新的远征

新的网络拓扑图如下（所有之前的ACL都已经被删除）：

<img src="/Users/wangjuanli/Codefield/2023Fall/NetworkSecurity/hw2/REPORT/image-20231211051337824.png" alt="image-20231211051337824" style="zoom:33%;" />

对于router1，我们将其连接外网（router4）的端口设置为1.1.1.1，router4连接router1的端口设置为1.1.1.2

对称地，我们将router2连接外网的端口设置为1.1.2.1，router4连接router2的端口设置为1.1.2.2

然后我们给边界路由添加静态路由转发（以router1为例）：

<img src="/Users/wangjuanli/Codefield/2023Fall/NetworkSecurity/hw2/REPORT/image-20231211051654349.png" alt="image-20231211051654349" style="zoom: 33%;" />

接着我们进行ISAKMP的配置。以Router1为例，这经历了以下五个步骤：

1. 配置ISAKMP

<img src="/Users/wangjuanli/Codefield/2023Fall/NetworkSecurity/hw2/REPORT/image-20231211051845920.png" alt="image-20231211051845920" style="zoom:33%;" />

2. 设置ACL

（这里偷了个懒，直接保护了192.168.0.0到192.168.0.0的所有流量）

<img src="/Users/wangjuanli/Codefield/2023Fall/NetworkSecurity/hw2/REPORT/image-20231211052004192.png" alt="image-20231211052004192" style="zoom:33%;" />

3. 设置transform-set

<img src="/Users/wangjuanli/Codefield/2023Fall/NetworkSecurity/hw2/REPORT/image-20231211052103912.png" alt="image-20231211052103912" style="zoom:33%;" />

4. 创建MAP映射表

<img src="/Users/wangjuanli/Codefield/2023Fall/NetworkSecurity/hw2/REPORT/image-20231211052123413.png" alt="image-20231211052123413" style="zoom:33%;" />

5. 绑定端口

<img src="/Users/wangjuanli/Codefield/2023Fall/NetworkSecurity/hw2/REPORT/image-20231211052137723.png" alt="image-20231211052137723" style="zoom:33%;" />

可以看出，我们ISAKMP的key为**2023fall**，index为1，transform-set的名称为**2023set**，MAP映射表的名字为**2023map**，index为1

对于router2，用完全对称的方式配置即可，只不过address和peer要改为1.1.1.1

这时从192.168.1.2可以ping通192.168.2.2，而我们没有对router4进行任何设置

<img src="/Users/wangjuanli/Codefield/2023Fall/NetworkSecurity/hw2/REPORT/image-20231211052419118.png" alt="image-20231211052419118" style="zoom:33%;" />

可以看到router1的isakmp状态也发生改变：

<img src="/Users/wangjuanli/Codefield/2023Fall/NetworkSecurity/hw2/REPORT/image-20231211052553224.png" alt="image-20231211052553224" style="zoom:33%;" />

### 在搬迁之后，使用配置静态路由的方法将无法让各个权力机构正常通信，请简述原因。

- 首先，我们只能改变边界路由器的设置，而不能改变公网路由器的设置，也就是说公网路由器拿到来自（例如）192.168.1.0子网的数据，并不知道它的下一跳是哪里。
- 其次，私网地址段（可能主要为）192.168.0.0-192.168.255.255。若内网的计算机想要与外网通信，其必须将ip经过NAT转换。

### 通过仿真抓包分析，如上配置的IPSec VPN使用了传输模式还是隧道模式，为什么？

<img src="/Users/wangjuanli/Codefield/2023Fall/NetworkSecurity/hw2/REPORT/image-20231211053324754.png" alt="image-20231211053324754" style="zoom:33%;" />

使用了隧道模式。原因如下：

<img src="/Users/wangjuanli/Codefield/2023Fall/NetworkSecurity/hw2/REPORT/image-20231211053404484.png" alt="image-20231211053404484" style="zoom:33%;" />

可以看到，传输模式是不改变原始IP包头的。然而这里，经过Router转发以后，报文的IP header中的src和dst都发生了变化，这说明ip包头被改变了。说明只能使用了隧道模式，附加了一个新IP包头上去。

## 凯撒的馈赠：NAT转换

### NAT简介

NAT (Network Address Translation，网络地址转换)，允许将**私有的IP地址**（10.0.0.0-10.255.255.255；172.16.0.0-172.31.255.255；192.168.0.0-192.168.255.255）映射到**合法的Internet IP地址（公网IP地址）**。

当**本地内网没有公网ip地址**，或是**需要合并多个一样的内网时**，NAT可以发挥作用。

### 静态NAT

- 每台主机**得到一个真实的IP地址**，一一映射

<img src="/Users/wangjuanli/Codefield/2023Fall/NetworkSecurity/hw2/REPORT/image-20231211060717430.png" alt="image-20231211060717430" style="zoom:33%;" />

以这样一个网络拓扑结构为例。假如router1左端口为192.168.1.1，PC0的地址为192.168.1.2。router1右端口为137.0.1.1，router2左端口为137.0.1.2。router2右端口为202.0.1.1，server0的地址为202.0.1.2。其中switch0，PC0为内网设备，Router1位边界路由器，Router2和Server0为公网设备。

我们再为Router1（边界路由器）和Router2（公网路由器）设置静态路由：

<img src="/Users/wangjuanli/Codefield/2023Fall/NetworkSecurity/hw2/REPORT/image-20231211063744805.png" alt="image-20231211063744805" style="zoom: 33%;" />

<img src="/Users/wangjuanli/Codefield/2023Fall/NetworkSecurity/hw2/REPORT/image-20231211063808914.png" alt="image-20231211063808914" style="zoom:33%;" />

这时我们在私网的192.168.1.2，去ping服务器202.0.1.2，是ping不通的。因为公网路由器不知道怎么转发192.168.1.2这个私网字段的地址。

<img src="/Users/wangjuanli/Codefield/2023Fall/NetworkSecurity/hw2/REPORT/image-20231211063902904.png" alt="image-20231211063902904" style="zoom:33%;" />

这时，我们在router1中为192.168.1.2手动指定公网地址：

```
// 左端为内部端口
int fa0/0
ip nat inside
exit

// 右端为外部端口
int fa0/1
ip nat outside

ip nat inside source static 192.168.1.2 131.0.1.3
```

<img src="/Users/wangjuanli/Codefield/2023Fall/NetworkSecurity/hw2/REPORT/image-20231211064049554.png" alt="image-20231211064049554" style="zoom:33%;" />

这时，我们再ping服务器：

<img src="/Users/wangjuanli/Codefield/2023Fall/NetworkSecurity/hw2/REPORT/image-20231211064121469.png" alt="image-20231211064121469" style="zoom:33%;" />

可以ping通了！我们观察一下报文：

<img src="/Users/wangjuanli/Codefield/2023Fall/NetworkSecurity/hw2/REPORT/image-20231211064157569.png" alt="image-20231211064157569" style="zoom:33%;" />

可以发现IP头的src被改变了。

### 动态NAT

静态NAT只是简单的一一映射，可能不能很好地解决公网ip不够用的情况。

我们先取消之前的静态NAT配置：

<img src="/Users/wangjuanli/Codefield/2023Fall/NetworkSecurity/hw2/REPORT/image-20231211064344111.png" alt="image-20231211064344111" style="zoom:33%;" />

然后配置动态NAT：

1. 设置ACL

<img src="/Users/wangjuanli/Codefield/2023Fall/NetworkSecurity/hw2/REPORT/image-20231211064519434.png" alt="image-20231211064519434" style="zoom:33%;" />

2. 定义公网地址池

<img src="/Users/wangjuanli/Codefield/2023Fall/NetworkSecurity/hw2/REPORT/image-20231211064655133.png" alt="image-20231211064655133" style="zoom:33%;" />

3. 绑定

<img src="/Users/wangjuanli/Codefield/2023Fall/NetworkSecurity/hw2/REPORT/image-20231211064710162.png" alt="image-20231211064710162" style="zoom:33%;" />

<img src="/Users/wangjuanli/Codefield/2023Fall/NetworkSecurity/hw2/REPORT/image-20231211065416520.png" alt="image-20231211065416520" style="zoom:33%;" />

可以ping通了，看一下报文：

<img src="/Users/wangjuanli/Codefield/2023Fall/NetworkSecurity/hw2/REPORT/image-20231211065450958.png" alt="image-20231211065450958" style="zoom:33%;" />

可以看到分配到的公网地址为137.0.1.3