## CPU

# OpenMP
```
#pragma omp parallel for
for (int i = 0; i < n; ++i) {
  for (int j = 0; j < n; ++j) {
    int sum = 0;
    for (int k = 0; k < n; ++k) {
      sum += a[i * n + k] * bt[j * n + k];
    }
    c[i * n + j] = sum;
  }
}
```

# Linux

## grep
grep 用来按模式搜索文本；手册页写得很明确：grep searches for patterns in each FILE。

grep "error" app.log
grep -n "main" main.cpp
grep -r "TODO" .
grep -i "warning" app.log

# ps

ps 用来查看进程；man page 说明它显示活动进程的信息，并指出如果要反复更新，应该用 top。

ps
ps -ef
ps aux

常见用法：

ps -ef：全格式
ps aux：BSD 风格，面试和实战都很常见
top

动态查看系统进程。

# top

适合看：

谁最吃 CPU
谁最吃内存
系统 load 高不高

# ssh

ssh 是远程登录和远程执行命令的标准工具；OpenSSH 手册和 Red Hat 文档都把它描述为加密的远程登录协议/客户端。

ssh user@host
ssh -p 2222 user@host
ssh user@host "hostname"
-p 指定端口
最后一种写法是在远程直接执行命令
# scp

scp 用于在主机之间安全复制文件，它使用 ssh 进行数据传输。man page 明确这样写。

scp file.txt user@host:/tmp/
scp user@host:/tmp/file.txt .
scp -r dir user@host:/tmp/

# gdb

调试程序。

gdb ./app
run
bt
break main
next
step
print x

最常用几个：

run 运行
bt 看栈回溯
break 下断点
next 单步越过
step 单步进入
print 打印变量
# strace

strace 的作用是跟踪进程的系统调用和信号；man page 明确写了它会拦截并记录系统调用。

strace ./app
strace -p 1234
strace -o trace.log ./app

什么时候用：

程序卡住了，不知道在干嘛
想看是不是卡在 open/read/write/futex
想知道某个文件到底有没有被访问

# perf

Linux 上非常常用的性能工具族。perf-trace 的 man page 说它可以显示 syscalls、pagefaults、调度等系统事件；perf-top 则实时生成并显示性能计数器 profile。

几个最常见的：

perf stat ./app
perf record ./app
perf report
perf top