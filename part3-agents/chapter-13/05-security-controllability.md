<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">

<span>[&larr; 13.4 数据库工具](04-database-tools.md)</span>

<span>[14.1 任务分解 &rarr;](../chapter-14/01-task-decomposition.md)</span>

</div>
---

# 13.5 安全性与可控性

> "智能体的安全是底线——能力越强，责任越大。"

## 安全风险分析

### 主要风险

```
智能体工具使用的安全风险：

1. 代码执行风险
   - 恶意代码执行
   - 系统命令注入
   - 资源耗尽攻击

2. 数据安全风险
   - 敏感数据泄露
   - 未授权数据访问
   - SQL注入

3. 权限提升风险
   - 越权操作
   - 权限绕过
   - 横向移动

4. 业务逻辑风险
   - 重复操作
   - 错误决策
   - 不可逆操作
```

## 安全设计原则

### 纵深防御

```
多层防护：

Layer 1: 输入验证
  - 参数类型检查
  - 格式验证
  - 长度限制

Layer 2: 权限控制
  - 用户身份验证
  - 操作权限检查
  - 资源访问控制

Layer 3: 沙箱隔离
  - 代码沙箱执行
  - 网络隔离
  - 资源限制

Layer 4: 审计日志
  - 操作记录
  - 异常监控
  - 追溯能力
```

## Java安全实现

### 沙箱执行

```java
package com.example.agent.security;

import java.security.*;
import java.io.*;

/**
 * 安全沙箱执行器
 */
public class SandboxExecutor {
    
    private final long maxMemory;
    private final long timeoutMs;
    
    public SandboxExecutor(long maxMemory, long timeoutMs) {
        this.maxMemory = maxMemory;
        this.timeoutMs = timeoutMs;
    }
    
    /**
     * 在沙箱中执行代码
     */
    public ExecutionResult execute(String code) {
        // 创建安全策略
        Policy policy = new Policy() {
            @Override
            public PermissionCollection getPermissions(
                    CodeSource codesource) {
                Permissions perms = new Permissions();
                // 最小权限原则
                perms.add(new RuntimePermission("accessDeclaredMembers"));
                return perms;
            }
        };
        
        // 创建受限线程
        ExecutorService executor = Executors.newSingleThreadExecutor();
        Future<ExecutionResult> future = executor.submit(() -> {
            // 设置安全管理器
            System.setSecurityManager(new SecurityManager() {
                @Override
                public void checkPermission(Permission perm) {
                    // 禁止文件系统访问
                    if (perm instanceof FilePermission) {
                        throw new SecurityException("文件访问被拒绝");
                    }
                    // 禁止网络访问
                    if (perm instanceof NetPermission ||
                        perm instanceof SocketPermission) {
                        throw new SecurityException("网络访问被拒绝");
                    }
                }
            });
            
            // 在受限环境中执行
            return runInSandbox(code);
        });
        
        try {
            return future.get(timeoutMs, TimeUnit.MILLISECONDS);
        } catch (TimeoutException e) {
            future.cancel(true);
            return ExecutionResult.failure("执行超时");
        } catch (Exception e) {
            return ExecutionResult.failure(e.getMessage());
        } finally {
            executor.shutdown();
        }
    }
}
```

### 权限控制

```java
/**
 * 工具权限管理器
 */
public class ToolPermissionManager {
    
    private final Map<String, Set<String>> userPermissions;
    
    /**
     * 检查用户是否有权限使用工具
     */
    public boolean checkPermission(String userId, String toolName, 
                                    Map<String, Object> params) {
        Set<String> allowedTools = userPermissions.get(userId);
        
        if (allowedTools == null || !allowedTools.contains(toolName)) {
            return false;
        }
        
        // 特殊参数检查
        if ("file_write".equals(toolName)) {
            String path = (String) params.get("path");
            return isPathAllowed(userId, path);
        }
        
        return true;
    }
    
    /**
     * 基于角色的权限
     */
    public boolean checkRolePermission(String userId, String toolName) {
        User user = userService.getUser(userId);
        
        return switch (user.getRole()) {
            case ADMIN -> true;  // 管理员可用所有工具
            case USER -> !isDangerousTool(toolName);
            case GUEST -> isReadOnlyTool(toolName);
        };
    }
}
```

## 可控性设计

### 人工确认

```java
/**
 * 需要确认的操作
 */
public class ConfirmationRequiredTool extends AbstractTool {
    
    private final Tool delegate;
    private final ConfirmationService confirmationService;
    
    @Override
    public ToolResult execute(Map<String, Object> params) {
        // 生成操作描述
        String description = describeOperation(params);
        
        // 请求用户确认
        Confirmation confirmation = confirmationService.request(
            description,
            Duration.ofMinutes(5)
        );
        
        if (!confirmation.isApproved()) {
            return ToolResult.failure("用户取消操作");
        }
        
        // 执行原工具
        return delegate.execute(params);
    }
    
    private String describeOperation(Map<String, Object> params) {
        return String.format(
            "即将执行 %s，参数: %s",
            delegate.getName(),
            params
        );
    }
}
```

### 操作限流

```java
/**
 * 限流工具包装器
 */
public class RateLimitedTool extends AbstractTool {
    
    private final Tool delegate;
    private final RateLimiter rateLimiter;
    
    @Override
    public ToolResult execute(Map<String, Object> params) {
        if (!rateLimiter.tryAcquire()) {
            return ToolResult.failure("操作过于频繁，请稍后再试");
        }
        
        return delegate.execute(params);
    }
}

/**
 * 基于用户限流
 */
public class UserRateLimiter {
    
    private final Map<String, RateLimiter> userLimiters = 
        new ConcurrentHashMap<>();
    
    public boolean tryAcquire(String userId) {
        RateLimiter limiter = userLimiters.computeIfAbsent(userId,
            k -> RateLimiter.create(10.0)); // 每秒10次
        
        return limiter.tryAcquire();
    }
}
```

## 审计与监控

### 审计日志

```java
/**
 * 审计日志服务
 */
@Service
public class AuditService {
    
    private final AuditLogRepository repository;
    
    /**
     * 记录工具调用
     */
    public void logToolInvocation(ToolInvocation invocation) {
        AuditLog log = AuditLog.builder()
            .timestamp(Instant.now())
            .userId(invocation.getUserId())
            .toolName(invocation.getToolName())
            .parameters(maskSensitiveData(invocation.getParams()))
            .result(invocation.getResult().isSuccess() ? "SUCCESS" : "FAILURE")
            .executionTime(invocation.getExecutionTime())
            .build();
        
        repository.save(log);
    }
    
    /**
     * 异常告警
     */
    public void checkAndAlert() {
        // 检测异常模式
        List<AuditLog> suspicious = repository.findSuspiciousActivities(
            Duration.ofMinutes(5),
            10  // 5分钟内超过10次调用
        );
        
        if (!suspicious.isEmpty()) {
            alertService.sendAlert("检测到可疑工具调用活动", suspicious);
        }
    }
}
```

## 小结

本章我们学习了：

1. **安全风险**：代码执行、数据安全、权限提升
2. **安全设计**：纵深防御、多层防护
3. **沙箱执行**：隔离代码执行环境
4. **权限控制**：基于角色和资源的权限
5. **可控性**：人工确认、操作限流
6. **审计监控**：日志记录、异常检测

**关键认识：**
安全是智能体系统的基石，必须在设计之初就考虑，而不是事后修补。

**下一步：** 我们将学习规划与推理。

---

**练习题：**

1. 设计一个完整的工具安全框架
2. 如何实现代码执行的完全隔离？
3. 设计智能

---

<div style="display: flex; justify-content: space-between; align-items: center; margin-top: 20px;">

<span>[&larr; 13.4 数据库工具](04-database-tools.md)</span>

<span>[14.1 任务分解 &rarr;](../chapter-14/01-task-decomposition.md)</span>

</div>