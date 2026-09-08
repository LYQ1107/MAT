# M0 网络审计（2026-09-08）

本审计只记录是否存在配置和可观察证据，不记录代理 URL、用户名、Cookie、token 或签名查询参数。

| 项目 | 结果 | 证据/边界 |
|---|---|---|
| 应用层代理变量 | `http_proxy/https_proxy/all_proxy` 及大小写均 `SET` | 值未输出；父 shell 未修改 |
| 命令路径 | curl `/home/lwr/anaconda3/bin/curl` 8.7.1；git `/usr/bin/git` 2.17.1；wget `/usr/bin/wget` | 通过 `type -a` 核验 |
| curl/wget 配置 | `/etc/wgetrc` 存在；用户 curlrc/wgetrc 未发现 | 内容未执行修改；curl 子进程必须 `-q` |
| URL 级 Git proxy | 当前审计命令未发现 | 仍用 `-c http.proxy= -c https.proxy=` 覆盖子进程 |
| LD_PRELOAD/proxychains | `LD_PRELOAD=UNSET`；未发现 proxychains/torsocks 命令 | 不能排除内核/出口透明代理 |
| 网卡/路由 | `lo, eno1, eno2`；默认 `202.205.84.1 dev eno1`；无可见 TUN | `ip addr/route/rule` 只读检查 |
| 本地代理监听 | 127.0.0.1:7890/7891 等存在 | 这些是现有控制环境的一部分，未停止 |
| 直连小测试 | 直连 `git ls-remote` 30s 超时，无响应 | 不能据此切换到代理下载大文件 |
| 控制通道小核验 | 通过已有源代码/网页控制通道读取了小文本 | 不用于数据/权重传输 |

结论：`application_proxy_bypass=true` 只适用于 `DirectOnlyPolicy.child_env` 子进程；`route_status=DIRECT_ROUTE_UNVERIFIED`，不是“绝对直连”。因此 Rat ID、PigReID、PigTracking、MultiCamCows 和所有未知模型权重均为 `BLOCKED_DIRECT_ROUTE`，直到管理员/用户提供经核验的直连出口、可信镜像或授权本地文件。续执行指令仅对本轮五个官方 SLEAP gerbil 对象显式授权 `AuthorizedProxyPolicy(allowed_hosts={storage.googleapis.com})`，其 receipts 标记 `AUTHORIZED_PROXY`；不会偷偷把其他资产切换到代理，也不会改变父进程代理/路由配置。
