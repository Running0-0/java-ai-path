<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">

<span>[&larr; 13.3 构建自定义工具](03-build-custom-tools.md)</span>

<span>[13.5 安全性与可控性 &rarr;](05-security-controllability.md)</span>

</div>
---

# 13.4 数据库工具

> "让智能体能够查询和操作数据库——连接AI能力与企业数据。"

## 数据库工具设计

### 设计思路

```
智能体操作数据库的挑战：

1. 安全性
   - 防止SQL注入
   - 权限控制
   - 敏感数据保护

2. 可靠性
   - 事务管理
   - 错误处理
   - 连接池管理

3. 可用性
   - 结果格式化
   - 分页支持
   - 自然语言转SQL

设计原则：
- 只读操作默认允许
- 写操作需要确认
- 危险操作（DELETE/DROP）禁止或受控
```

## 只读查询工具

### 自然语言转SQL

```java
package com.example.agent.tools;

import dev.langchain4j.agent.tool.P;
import dev.langchain4j.agent.tool.Tool;

import javax.sql.DataSource;
import java.sql.*;
import java.util.*;

/**
 * 数据库查询工具
 */
public class DatabaseQueryTool {
    
    private final DataSource dataSource;
    private final ChatLanguageModel llm;
    private final SchemaInfo schemaInfo;
    
    @Tool("""
        查询数据库信息。
        可以描述你想要的数据，系统会自动生成SQL查询。
        支持：统计、筛选、排序、聚合等操作。
        注意：只支持只读查询。
        """)
    public QueryResult queryDatabase(
        @P("用自然语言描述想要查询的数据，例如：'查询今日新注册的用户数量'") 
        String query
    ) {
        // 1. 自然语言转SQL
        String sql = translateToSQL(query);
        
        // 2. 验证SQL（只允许SELECT）
        validateSQL(sql);
        
        // 3. 执行查询
        return executeQuery(sql);
    }
    
    /**
     * 自然语言转SQL
     */
    private String translateToSQL(String naturalQuery) {
        String prompt = String.format("""
            将以下自然语言查询转换为SQL语句。
            
            数据库结构：
            %s
            
            查询：%s
            
            规则：
            1. 只生成SELECT语句
            2. 不使用子查询（尽量）
            3. 添加LIMIT限制结果数量（默认100）
            4. 只输出SQL，不要解释
            """, schemaInfo.describe(), naturalQuery);
        
        return llm.generate(prompt).trim();
    }
    
    /**
     * SQL安全验证
     */
    private void validateSQL(String sql) {
        String upperSQL = sql.toUpperCase().trim();
        
        // 只允许SELECT
        if (!upperSQL.startsWith("SELECT")) {
            throw new SecurityException("只允许SELECT查询");
        }
        
        // 禁止危险关键字
        List<String> forbidden = List.of(
            "DROP", "DELETE", "UPDATE", "INSERT", 
            "ALTER", "CREATE", "EXEC"
        );
        
        for (String keyword : forbidden) {
            if (upperSQL.contains(keyword)) {
                throw new SecurityException("SQL包含不允许的操作: " + keyword);
            }
        }
    }
    
    /**
     * 执行查询
     */
    private QueryResult executeQuery(String sql) {
        try (Connection conn = dataSource.getConnection();
             PreparedStatement stmt = conn.prepareStatement(sql);
             ResultSet rs = stmt.executeQuery()) {
            
            // 获取列名
            ResultSetMetaData meta = rs.getMetaData();
            int colCount = meta.getColumnCount();
            List<String> columns = new ArrayList<>();
            for (int i = 1; i <= colCount; i++) {
                columns.add(meta.getColumnName(i));
            }
            
            // 获取数据
            List<Map<String, Object>> rows = new ArrayList<>();
            while (rs.next()) {
                Map<String, Object> row = new LinkedHashMap<>();
                for (String col : columns) {
                    row.put(col, rs.getObject(col));
                }
                rows.add(row);
            }
            
            return new QueryResult(columns, rows, sql);
            
        } catch (SQLException e) {
            throw new ToolException("查询失败: " + e.getMessage());
        }
    }
}
```

## 写操作工具

### 受控写操作

```java
/**
 * 数据库写操作工具（受控）
 */
public class DatabaseWriteTool {
    
    private final DataSource dataSource;
    private final AuditLogger auditLogger;
    
    @Tool("""
        向数据库插入数据。
        操作将被记录到审计日志。
        适用于：添加新记录。
        """)
    public String insertRecord(
        @P("目标表名") String tableName,
        @P("要插入的数据（JSON格式）") String jsonData
    ) {
        // 验证表名白名单
        validateTableName(tableName);
        
        // 解析数据
        Map<String, Object> data = parseJson(jsonData);
        
        // 构建INSERT语句
        String sql = buildInsertSQL(tableName, data);
        
        // 执行并记录审计
        try (Connection conn = dataSource.getConnection();
             PreparedStatement stmt = conn.prepareStatement(sql)) {
            
            fillParameters(stmt, data);
            int affected = stmt.executeUpdate();
            
            // 审计日志
            auditLogger.log("INSERT", tableName, data, affected);
            
            return String.format("成功插入 %d 条记录到 %s", 
                affected, tableName);
                
        } catch (SQLException e) {
            auditLogger.logError("INSERT", tableName, e.getMessage());
            throw new ToolException("插入失败: " + e.getMessage());
        }
    }
    
    /**
     * 白名单验证
     */
    private void validateTableName(String tableName) {
        Set<String> allowedTables = Set.of(
            "user_notes", "task_records", "agent_logs"
        );
        
        if (!allowedTables.contains(tableName.toLowerCase())) {
            throw new SecurityException("不允许操作的表: " + tableName);
        }
    }
}
```

## Schema信息工具

### 数据库元数据

```java
/**
 * 数据库结构查询工具
 */
public class SchemaInfoTool {
    
    private final DataSource dataSource;
    
    @Tool("获取数据库表结构信息，帮助理解数据库设计")
    public String getDatabaseSchema(
        @P("表名（可选，不填则返回所有表）") String tableName
    ) {
        try (Connection conn = dataSource.getConnection()) {
            DatabaseMetaData meta = conn.getMetaData();
            
            StringBuilder result = new StringBuilder();
            
            if (tableName != null && !tableName.isEmpty()) {
                // 获取指定表结构
                result.append(getTableStructure(meta, tableName));
            } else {
                // 获取所有表
                ResultSet tables = meta.getTables(null, null, "%", 
                    new String[]{"TABLE"});
                while (tables.next()) {
                    String table = tables.getString("TABLE_NAME");
                    result.append(getTableStructure(meta, table));
                    result.append("\n");
                }
            }
            
            return result.toString();
            
        } catch (SQLException e) {
            throw new ToolException("获取表结构失败: " + e.getMessage());
        }
    }
    
    private String getTableStructure(DatabaseMetaData meta, 
                                      String tableName) throws SQLException {
        StringBuilder sb = new StringBuilder();
        sb.append("表名: ").append(tableName).append("\n");
        sb.append("列：\n");
        
        ResultSet columns = meta.getColumns(null, null, tableName, "%");
        while (columns.next()) {
            sb.append(String.format("  - %s (%s)%s\n",
                columns.getString("COLUMN_NAME"),
                columns.getString("TYPE_NAME"),
                "YES".equals(columns.getString("IS_NULLABLE")) ? 
                    "" : " NOT NULL"
            ));
        }
        
        return sb.toString();
    }
}
```

## 使用示例

### 完整对话示例

```java
/**
 * 智能数据库助手使用示例
 */
public class DatabaseAssistantDemo {
    
    public void demonstrate() {
        // 注册工具
        Agent agent = AgentBuilder.create()
            .withTool(new DatabaseQueryTool(dataSource, llm, schema))
            .withTool(new SchemaInfoTool(dataSource))
            .build();
        
        // 对话示例
        System.out.println(
            agent.chat("我们今天有多少新用户注册？"));
        // LLM自动调用 queryDatabase("今日新注册用户数量")
        // 自动转换SQL: SELECT COUNT(*) FROM users WHERE created_at >= CURDATE()
        
        System.out.println(
            agent.chat("过去一周每天的订单量是多少？"));
        // 自动生成并执行相应SQL
        
        System.out.println(
            agent.chat("解释一下orders表的结构"));
        // 调用 getDatabaseSchema("orders")
    }
}
```

## 小结

本章我们学习了：

1. **设计思路**：安全、可靠、可用的数据库工具
2. **只读工具**：自然语言转SQL，安全验证
3. **写操作**：受控写操作，审计日志
4. **Schema工具**：数据库元数据查询
5. **实际应用**：智能数据库助手

**关键认识：**
数据库工具让智能体能够直接与企业数据交互，但安全性必须放在首位。

**下一步：** 我们将学习安全性与可控性。

---

**练习题：**

1. 实现一个支持分页的数据库查询工具
2. 如何防止SQL注入攻击？
3. 设计数据库工具的访问控制策略。

---

<div style="display: flex; justify-content: space-between; align-items: center; margin-top: 20px;">

<span>[&larr; 13.3 构建自定义工具](03-build-custom-tools.md)</span>

<span>[13.5 安全性与可控性 &rarr;](05-security-controllability.md)</span>

</div>